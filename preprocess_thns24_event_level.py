import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

ACT_DICT = {
    "logon": 1, "logoff": 2, "http": 3, "email": 4,
    "file": 5, "connect": 6, "disconnect": 7,
}

class SemanticFeatureExtractor:
    def __init__(self):
        self.semantic_map = {
            "None": 0, "Http_Normal": 1, "Http_Job": 2, "Http_Leak": 3, "Http_Cloud": 4, "Http_Hack": 5, "Http_Social": 6,
            "File_Other": 11, "File_Doc": 12, "File_Zip": 13, "File_Exe": 14, "File_Img": 15,
            "Email_Int": 21, "Email_Ext": 22,
        }
        self.http_domains = {
            "Http_Job": ["indeed.com", "monster.com", "careerbuilder.com", "simplyhired.com", "aol.com", "job"],
            "Http_Leak": ["wikileaks.org", "freedom.press", "theintercept.com"],
            "Http_Cloud": ["dropbox.com", "drive.google.com", "mega.co.nz", "live.com", "box.com"],
            "Http_Hack": ["keylog", "best-spy-soft", "webwatchernow", "spectorsoft"],
            "Http_Social": ["facebook.com", "twitter.com", "linkedin.com", "reddit.com", "instagram.com"],
        }
        self.file_exts = {
            "File_Zip": ["zip", "rar", "7z", "tar", "gz"],
            "File_Img": ["jpg", "png", "bmp", "jpeg", "gif"],
            "File_Doc": ["doc", "docx", "pdf", "ppt", "pptx", "xls", "xlsx", "txt", "rtf"],
            "File_Exe": ["exe", "sh", "bat", "cmd", "dll", "msi"],
        }

    def semantic_id(self, act_type: str, detail: str) -> int:
        detail = detail.lower() if detail else ""
        if act_type == "http":
            for tag, kws in self.http_domains.items():
                if any(kw in detail for kw in kws): return self.semantic_map[tag]
            return self.semantic_map["Http_Normal"]
        if act_type == "file":
            if "." in detail:
                ext = detail.rsplit(".", 1)[-1]
                for tag, exts in self.file_exts.items():
                    if ext in exts: return self.semantic_map[tag]
            return self.semantic_map["File_Other"]
        if act_type == "email":
            return self.semantic_map["Email_Ext"] if "dtaa.com" not in detail else self.semantic_map["Email_Int"]
        return self.semantic_map["None"]


def parse_dt(date_str: str) -> Optional[datetime]:
    try:
        return datetime.strptime(date_str, "%m/%d/%Y %H:%M:%S")
    except Exception:
        return None

def collect_insider_labels(insider_dir: str) -> Tuple[set, set]:
    anomaly_user_day = set()
    insider_users = set()
    for fp in glob.glob(os.path.join(insider_dir, "r4.2-*.csv")):
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 4: continue
                user = row[3].strip()
                dt = parse_dt(row[2].strip())
                if not user or not dt: continue
                anomaly_user_day.add((user, dt.strftime("%m/%d/%Y")))
                insider_users.add(user)
    return anomaly_user_day, insider_users

def load_sessions(log_dir: str, extractor: SemanticFeatureExtractor) -> Dict[str, List]:
    files = ["logon.csv", "http.csv", "email.csv", "file.csv", "device.csv"]
    data_map = defaultdict(list)
    for name in files:
        path = os.path.join(log_dir, name)
        if not os.path.exists(path): continue
        print(f"[load] {name}")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) < 3: continue
                dt = parse_dt(row[1].strip())
                user = row[2].strip()
                if not dt or not user: continue
                
                base = name.lower()
                act = None
                if base == "logon.csv":
                    act = "logon" if "logon" in (row[4].lower() if len(row) > 4 else "") else "logoff"
                elif base == "device.csv":
                    act = "connect" if "connect" in (row[4].lower() if len(row) > 4 else "") else "disconnect"
                elif base in ["http.csv", "email.csv", "file.csv"]:
                    act = base.split('.')[0]
                
                if not act: continue
                act_id = ACT_DICT.get(act, 0)
                
                detail = ""
                if base in ["http.csv", "file.csv"]:
                    detail = row[4] if len(row) > 4 else ""
                elif base == "email.csv":
                    detail = ";".join([row[i] for i in (4,5,6,7) if len(row) > i and row[i]])
                
                sem_id = extractor.semantic_id(act, detail)
                data_map[user].append((dt, f"{act_id}_{dt.hour}_{sem_id}"))
    return data_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="data/r4.2", type=str)
    parser.add_argument("--insider_dir", default="data/insider", type=str)
    parser.add_argument("--out_dir", default="data/itdbert_event", type=str)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    extractor = SemanticFeatureExtractor()
    anomaly_user_day, insider_users = collect_insider_labels(args.insider_dir)
    print(f"[insider] anomaly_user_day={len(anomaly_user_day)}, insider_users={len(insider_users)}")

    data_map = load_sessions(args.log_dir, extractor)
    print(f"[sessions] total users={len(data_map)}")

    train_rows = []
    test_events_rows = []
    test_labels_rows = []

    for user, events in data_map.items():
        events_2010 = [e for e in events if e[0].year == 2010]
        events_2011 = [e for e in events if e[0].year == 2011]
        
        # Train (2010): Only include pure users
        if events_2010:
            is_anom = any((user, e[0].strftime("%m/%d/%Y")) in anomaly_user_day for e in events_2010)
            if not is_anom:
                ordered = [x[1] for x in sorted(events_2010, key=lambda t: t[0])]
                # Format: [user, event1, event2, ...]
                train_rows.append([user] + ordered)
                
        # Test (2011): Include all users, generate event-level labels
        if events_2011:
            sorted_2011 = sorted(events_2011, key=lambda t: t[0])
            ordered_events = [x[1] for x in sorted_2011]
            ordered_labels = ["1" if (user, x[0].strftime("%m/%d/%Y")) in anomaly_user_day else "0" for x in sorted_2011]
            
            test_events_rows.append([user] + ordered_events)
            test_labels_rows.append([user] + ordered_labels)

    with open(os.path.join(args.out_dir, "THNS24_2010.csv"), "w", newline="") as f:
        csv.writer(f).writerows(train_rows)
    with open(os.path.join(args.out_dir, "THNS24_2011.csv"), "w", newline="") as f:
        csv.writer(f).writerows(test_events_rows)
    with open(os.path.join(args.out_dir, "THNS24_2011_event_labels.csv"), "w", newline="") as f:
        csv.writer(f).writerows(test_labels_rows)
        
    print(f"[done] Extracted event-level datasets to {args.out_dir}")

if __name__ == "__main__":
    main()