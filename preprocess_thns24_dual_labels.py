import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple


ACT_DICT = {
    "logon": 1,
    "logoff": 2,
    "http": 3,
    "email": 4,
    "file": 5,
    "connect": 6,
    "disconnect": 7,
}


class SemanticFeatureExtractor:
    def __init__(self):
        self.semantic_map = {
            "None": 0,
            "Http_Normal": 1,
            "Http_Job": 2,
            "Http_Leak": 3,
            "Http_Cloud": 4,
            "Http_Hack": 5,
            "Http_Social": 6,
            "File_Other": 11,
            "File_Doc": 12,
            "File_Zip": 13,
            "File_Exe": 14,
            "File_Img": 15,
            "Email_Int": 21,
            "Email_Ext": 22,
        }

        self.http_domains = {
            "Http_Job": [
                "indeed.com",
                "monster.com",
                "careerbuilder.com",
                "simplyhired.com",
                "aol.com",
                "job",
            ],
            "Http_Leak": [
                "wikileaks.org",
                "freedom.press",
                "theintercept.com",
            ],
            "Http_Cloud": [
                "dropbox.com",
                "drive.google.com",
                "mega.co.nz",
                "live.com",
                "box.com",
            ],
            "Http_Hack": [
                "keylog",
                "best-spy-soft",
                "webwatchernow",
                "spectorsoft",
            ],
            "Http_Social": [
                "facebook.com",
                "twitter.com",
                "linkedin.com",
                "reddit.com",
                "instagram.com",
            ],
        }

        self.file_exts = {
            "File_Zip": ["zip", "rar", "7z", "tar", "gz"],
            "File_Img": ["jpg", "png", "bmp", "jpeg", "gif"],
            "File_Doc": ["doc", "docx", "pdf", "ppt", "pptx", "xls", "xlsx", "txt", "rtf"],
            "File_Exe": ["exe", "sh", "bat", "cmd", "dll", "msi"],
        }

    def _http_sem(self, url: str) -> int:
        if not url:
            return self.semantic_map["Http_Normal"]
        s = url.lower()
        for tag, kws in self.http_domains.items():
            if any(kw in s for kw in kws):
                return self.semantic_map[tag]
        return self.semantic_map["Http_Normal"]

    def _file_sem(self, filename: str) -> int:
        if not filename or "." not in filename:
            return self.semantic_map["File_Other"]
        ext = filename.rsplit(".", 1)[-1].lower()
        for tag, exts in self.file_exts.items():
            if ext in exts:
                return self.semantic_map[tag]
        return self.semantic_map["File_Other"]

    def _email_sem(self, detail: str) -> int:
        if not detail:
            return self.semantic_map["Email_Int"]
        s = detail.lower()
        return self.semantic_map["Email_Ext"] if "dtaa.com" not in s else self.semantic_map["Email_Int"]

    def semantic_id(self, act_type: str, detail: str) -> int:
        if act_type == "http":
            return self._http_sem(detail)
        if act_type == "file":
            return self._file_sem(detail)
        if act_type == "email":
            return self._email_sem(detail)
        return self.semantic_map["None"]


def parse_dt(date_str: str) -> Optional[datetime]:
    try:
        return datetime.strptime(date_str, "%m/%d/%Y %H:%M:%S")
    except Exception:
        return None


def collect_insider_labels(insider_dir: str) -> Tuple[set, set]:
    anomaly_user_day = set()
    insider_users = set()
    files = glob.glob(os.path.join(insider_dir, "r4.2-*.csv"))
    for fp in files:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 4:
                    continue
                user = row[3].strip()
                dt = parse_dt(row[2].strip())
                if not user or dt is None:
                    continue
                day = dt.strftime("%m/%d/%Y")
                anomaly_user_day.add((user, day))
                insider_users.add(user)
    return anomaly_user_day, insider_users


def detect_action(file_name: str, row: List[str]) -> Optional[str]:
    base = file_name.lower()
    if base == "logon.csv":
        activity = row[4].lower() if len(row) > 4 else ""
        return "logon" if "logon" in activity else "logoff"
    if base == "device.csv":
        activity = row[4].lower() if len(row) > 4 else ""
        return "connect" if "connect" in activity else "disconnect"
    if base == "http.csv":
        return "http"
    if base == "email.csv":
        return "email"
    if base == "file.csv":
        return "file"
    return None


def detail_for_sem(file_name: str, row: List[str], mode: str) -> str:
    base = file_name.lower()
    if base == "http.csv":
        return row[4] if len(row) > 4 else ""
    if base == "file.csv":
        return row[4] if len(row) > 4 else ""
    if base == "email.csv":
        # compat: mimic old data_finall preprocessing (use only "to" field).
        if mode == "compat":
            return row[4] if len(row) > 4 else ""
        # enhanced: merge to/cc/bcc/from for richer external-contact signal.
        fields = []
        for idx in (4, 5, 6, 7):
            if len(row) > idx and row[idx]:
                fields.append(row[idx])
        return ";".join(fields)
    return ""


def load_sessions(
    log_dir: str,
    extractor: SemanticFeatureExtractor,
    mode: str,
    max_rows_per_file: int = 0,
) -> Dict[str, List[Tuple[datetime, int, str]]]:
    files = ["logon.csv", "http.csv", "email.csv", "file.csv", "device.csv"]
    data_map: Dict[str, List[Tuple[datetime, int, str]]] = defaultdict(list)
    for name in files:
        path = os.path.join(log_dir, name)
        if not os.path.exists(path):
            continue
        print(f"[load] {name}")
        line_idx = 0
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            _ = next(reader, None)
            for row in reader:
                if len(row) < 3:
                    continue
                dt = parse_dt(row[1].strip())
                user = row[2].strip()
                if dt is None or not user:
                    continue
                act = detect_action(name, row)
                if act is None:
                    continue
                act_id = ACT_DICT.get(act, 0)
                sem_detail = detail_for_sem(name, row, mode=mode)
                sem_id = extractor.semantic_id(act, sem_detail)
                token = f"{act_id}_{dt.hour}_{sem_id}"

                key = user
                data_map[key].append((dt, line_idx, token))
                line_idx += 1

                if max_rows_per_file > 0 and line_idx >= max_rows_per_file:
                    break
        print(f"[load] {name} done, rows used={line_idx}")
    return data_map


def write_rows(path: str, rows: List[List[str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def maybe_pad_fixed_width(rows: List[List[str]], target_width: Optional[int]) -> List[List[str]]:
    if not target_width or target_width <= 0:
        return rows
    padded: List[List[str]] = []
    for row in rows:
        if len(row) < target_width:
            padded.append(row + [""] * (target_width - len(row)))
        else:
            padded.append(row[:target_width])
    return padded


def main():
    parser = argparse.ArgumentParser(description="Build THNS24 data_finall-style data with extra user-level labels.")
    parser.add_argument("--log_dir", default="data/r4.2", type=str)
    parser.add_argument("--insider_dir", default="data/insider", type=str)
    parser.add_argument("--out_dir", default="data/itdbert", type=str)
    parser.add_argument(
        "--mode",
        default="enhanced",
        choices=["compat", "enhanced"],
        help="compat: closer to old data_finall; enhanced: richer email semantics.",
    )
    parser.add_argument(
        "--min_events",
        default=1,
        type=int,
        help="Minimum events per user-day session; default 1 to avoid dropping sessions.",
    )
    parser.add_argument(
        "--fixed_width_pad",
        default=0,
        type=int,
        help="Optional target column width for fixed-width csv (0 means variable length).",
    )
    parser.add_argument(
        "--max_rows_per_file",
        default=0,
        type=int,
        help="Debug mode: cap rows per raw log file; 0 means full.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    extractor = SemanticFeatureExtractor()
    anomaly_user_day, insider_users = collect_insider_labels(args.insider_dir)
    print(f"[insider] anomaly_user_day={len(anomaly_user_day)}, insider_users={len(insider_users)}")

    data_map = load_sessions(
        args.log_dir,
        extractor,
        mode=args.mode,
        max_rows_per_file=args.max_rows_per_file,
    )
    print(f"[sessions] total users={len(data_map)}")

    train_log_rows: List[List[str]] = []
    test_log_rows: List[List[str]] = []
    test_user_rows: List[List[str]] = []
    test_dual_rows: List[List[str]] = []

    skipped_short = 0
    for user, events in data_map.items():
        
        events_2010 = [e for e in events if e[0].year == 2010]
        events_2011 = [e for e in events if e[0].year == 2011]
        
        # Training Set: 2010
        if len(events_2010) >= args.min_events:
            ordered_2010 = [x[2] for x in sorted(events_2010, key=lambda t: (t[0], t[1]))]
            # Check if this user had any anomaly in 2010
            is_anom_2010 = 1 if any((user, e[0].strftime("%m/%d/%Y")) in anomaly_user_day for e in events_2010) else 0
            train_log_rows.append([str(is_anom_2010)] + ordered_2010)
            
        # Test Set: 2011
        if len(events_2011) >= args.min_events:
            ordered_2011 = [x[2] for x in sorted(events_2011, key=lambda t: (t[0], t[1]))]
            
            # Anomaly label based on user
            user_label = 1 if user in insider_users else 0
            
            test_log_rows.append([str(user_label)] + ordered_2011)
            test_user_rows.append([str(user_label)] + ordered_2011)
            test_dual_rows.append([str(user_label), str(user_label), user, "2011"] + ordered_2011)
        else:
            if len(events) < args.min_events:
                skipped_short += 1

    if args.fixed_width_pad > 0:
        train_log_rows = maybe_pad_fixed_width(train_log_rows, args.fixed_width_pad)
        test_log_rows = maybe_pad_fixed_width(test_log_rows, args.fixed_width_pad)
        test_user_rows = maybe_pad_fixed_width(test_user_rows, args.fixed_width_pad)
        # dual labels include extra meta columns; keep it variable by default.

    train_path = os.path.join(args.out_dir, "THNS24_2010.csv")
    test_log_path = os.path.join(args.out_dir, "THNS24_2011.csv")
    test_user_path = os.path.join(args.out_dir, "THNS24_2011_userlabel.csv")
    test_dual_path = os.path.join(args.out_dir, "THNS24_2011_dual_labels.csv")
    stat_path = os.path.join(args.out_dir, "THNS24_preprocess_stats.json")

    write_rows(train_path, train_log_rows)
    write_rows(test_log_path, test_log_rows)
    write_rows(test_user_path, test_user_rows)
    write_rows(test_dual_path, test_dual_rows)

    def count_label(rows: List[List[str]], idx: int = 0) -> Dict[str, int]:
        out = {"0": 0, "1": 0}
        for r in rows:
            if len(r) > idx and r[idx] in out:
                out[r[idx]] += 1
        return out

    stats = {
        "mode": args.mode,
        "min_events": args.min_events,
        "fixed_width_pad": args.fixed_width_pad,
        "train_rows": len(train_log_rows),
        "test_rows": len(test_log_rows),
        "test_user_rows": len(test_user_rows),
        "test_dual_rows": len(test_dual_rows),
        "skipped_short_sessions": skipped_short,
        "train_log_label_dist": count_label(train_log_rows, 0),
        "test_log_label_dist": count_label(test_log_rows, 0),
        "test_user_label_dist": count_label(test_user_rows, 0),
        "output_files": {
            "train_log": train_path,
            "test_log": test_log_path,
            "test_user": test_user_path,
            "test_dual": test_dual_path,
        },
    }
    with open(stat_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("[done] wrote:")
    for k, v in stats["output_files"].items():
        print(f"  - {k}: {v}")
    print(f"[done] stats: {stat_path}")
    print(json.dumps({k: stats[k] for k in ('train_log_label_dist', 'test_log_label_dist', 'test_user_label_dist')}, ensure_ascii=False))


if __name__ == "__main__":
    main()
