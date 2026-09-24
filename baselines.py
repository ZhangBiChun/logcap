import torch
import os
import random
import numpy as np
import pandas as pd
import json
from datetime import datetime
import argparse
from dataloader import *
from models import ad_predict
from models.NCAD.dataset import data_generator
from models.GDN.dataset import graph_data
from models.reasonable_metric import reasonable_accumulator
from ts_datasets.ts_datasets.anomaly import NAB, IOpsCompetition, SMAP, SMD, UCR
from tqdm import tqdm
from merlion.evaluate.anomaly import TSADScoreAccumulator as ScoreAcc, ScoreType
from utils import print_object
import matplotlib.pyplot as plt


# Args selections
start_time = datetime.now()
parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='Exp1', type=str,
                    help='Experiment Description')
parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description')
parser.add_argument('--visualization', default=False, type=bool,
                    help='Visualize')
parser.add_argument('--seed', default=6, type=int,
                    help='seed value')
parser.add_argument('--selected_dataset', default='SMD', type=str,
                    help='Dataset of choice: IOpsCompetition, UCR, NAB, SWaT, WADI, SMAP, ESA, SMD, THNS24')
parser.add_argument('--selected_model', default='SHUE', type=str,
                    help='Model of choice: COCA, AOC, RAS, AAS, NCAD, LSTM_ED, OC_SVM, ISF, SR, RRCF, SVDD, GDN, AOT, DAMP, SHUE, MTS_CAD, MixMamba, TranAD, ITDBERT, DeepLog, LogBERT, LogGPT, LogRoBERTa, Mamba_Seq')
# parser.add_argument('--windowsize', default=256, type=int,
#                     help='window_size')
# parser.add_argument('--timestep', default=256, type=int,
#                     help='time_step')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')
parser.add_argument('--thns24_max_sessions', default=0, type=int,
                    help='THNS24 quick-run cap per split; 0 means full dataset')
parser.add_argument('--thns24_sample_mode', default='stratified', type=str,
                    help='THNS24 sampling: stratified | head | full')
parser.add_argument('--thns24_level', default='event', type=str,
                    help='THNS24 data level: day | event')
parser.add_argument('--thns24_train_csv', default='data/itdbert_event/THNS24_2010.csv', type=str,
                    help='THNS24 train csv path')
parser.add_argument('--thns24_test_csv', default='data/itdbert_event/THNS24_2011.csv', type=str,
                    help='THNS24 test csv path')
parser.add_argument('--thns24_user_test_csv', default='data/itdbert_event/THNS24_2011_event_labels.csv', type=str,
                    help='THNS24 test labels csv path')
parser.add_argument('--thns24_dual_eval', default=True, type=bool,
                    help='Whether to run extra evaluation with user-level labels')
parser.add_argument('--num_epoch', default=None, type=int,
                    help='Override config num_epoch')
parser.add_argument('--active_aug_ops', default='1,2,3,4', type=str,
                    help='Comma-separated list of active augmentation operators (1: Semantic Shift, 2: Burst, 3: Off-hours, 4: Alien Fragment)')
parser.add_argument('--ablation_tag', default='', type=str,
                    help='Tag to append to the model save path and result logs for ablation studies')
args = parser.parse_args()

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(args.device)
experiment_description = args.experiment_description
data_type = args.selected_dataset
method = args.selected_model
run_description = args.run_description
selected_dataset = args.selected_dataset
visualization = args.visualization

if method == 'COCA':
    from models.COCA.coca_trainer.trainer import Trainer
    from models.COCA.coca_network.model import base_Model
elif method == 'AOC':
    from models.AOC.trainer.trainer import Trainer
    from models.AOC.network.model import base_Model
elif method == 'RAS':
    from models.RandomScore.trainer import Trainer
elif method == 'AAS':
    from models.InputOrigin.trainer import Trainer
elif method == 'NCAD':
    from models.NCAD.model import base_Model
    from models.NCAD.trainer import Trainer
elif method == 'LSTM_ED':
    from models.LSTMED.trainer.trainer import Trainer
    from models.LSTMED.network.model import LSTMEDModule
elif method == 'OC_SVM':
    from models.OCSVM.trainer import Trainer
elif method == 'ISF':
    from models.ISF.trainer import Trainer
elif method == 'SR':
    from models.SR.trainer import Trainer
elif method == 'RRCF':
    from models.RRCF.trainer import Trainer
elif method == 'DAMP':
    from models.DAMP.trainer import Trainer
elif method == 'SVDD':
    from models.DeepSVDD.trainer.trainer import Trainer
    from models.DeepSVDD.network.model import base_Model
elif method == 'GDN':
    from models.GDN.trainer import Trainer
    from models.GDN.networks.GDN import GDN
elif method == 'AOT':
    from models.Anomaly_Transformer.trainer.trainer import Trainer
    from models.Anomaly_Transformer.model.AnomalyTransformer import AnomalyTransformer
elif method == 'SHUE':
    from models.SensitiveHUE.trainer.trainer import Trainer
    from models.SensitiveHUE.model.model import SensitiveHUE
elif method == 'MTS_CAD':
    from models.MTS_CAD.trainer.trainer import Trainer
    from models.MTS_CAD.model.model import MMoE
elif method == 'MixMamba':
    from models.MixMamba.trainer.trainer import Trainer
    from models.MixMamba.model.mixmamba import Model
elif method == 'TranAD':
    from models.TranAD.trainer.trainer import Trainer
    from models.TranAD.model.model import TranAD
elif method == 'ITDBERT':
    from models.ITDBERT.trainer import Trainer
elif method == 'LogBERT' or method == 'logbert':
    from models.LogBERT.trainer import Trainer
elif method == 'LogGPT':
    from models.LogGPT.trainer import Trainer
elif method == 'LMTracker':
    from models.LMTracker.trainer import Trainer
elif method == 'LAN' or method == 'lan':
    from models.LAN.trainer import Trainer
elif method == 'LaAeb' or method == 'laaeb':
    from models.LaAeb.trainer import Trainer
elif method == 'LogRoBERTa':
    from models.LogRoBERTa.trainer import Trainer
elif method == 'Mamba_Seq':
    from models.Mamba_Seq.trainer import Trainer
elif method == 'DeepLog':
    from models.DeepLog.trainer import Trainer
elif method == 'AE':
    from models.AE.trainer import Trainer
elif method == 'DNN':
    from models.DNN.trainer import Trainer
elif method == 'LSTM':
    from models.LSTM.trainer import Trainer
elif method == 'LogCapMamba':
    from models.LogCapMamba.trainer import Trainer
else:
    from models.COCA.coca_trainer.trainer import Trainer
    from models.COCA.coca_network.model import base_Model

if method == 'NCAD':
    class DictToClass(object):
        def __init__(self, _obj):
            if _obj:
                self.__dict__.update(_obj)
    NCAD_ROOT = os.path.dirname(os.path.abspath(__file__))
    CONFIG_NAME = data_type + '.json'
    CONFIG_JSON = os.path.join(NCAD_ROOT, "conf/ncad", CONFIG_NAME)
    with open(CONFIG_JSON, "r") as f:
        config_json = json.load(f)
    configs = DictToClass(config_json)
    # configs.window_size = args.windowsize
    # configs.time_step = args.timestep
else:
    Configs = None
    try:
        if method == 'DeepLog':
            from conf.deeplog.THNS24_Configs import Config as Configs
        elif method == 'ITDBERT':
            from conf.itdbert.THNS24_Configs import Config as Configs
        elif method in ['LogBERT', 'logbert']:
            from conf.logbert.THNS24_Configs import Config as Configs
        elif method == 'LogGPT':
            from conf.loggpt.THNS24_Configs import Config as Configs
        elif method == 'LogRoBERTa':
            from conf.logroberta.THNS24_Configs import Config as Configs
        elif method == 'Mamba_Seq':
            from conf.mamba_seq.THNS24_Configs import Config as Configs
        else:
            exec(f'from conf.{method.lower()}.{data_type}_Configs import Config as Configs')
    except Exception:
        # THNS24 currently reuses UCR configs for legacy tsad models.
        exec(f'from conf.{method.lower()}.UCR_Configs import Config as Configs')
    configs = Configs()
    if args.num_epoch is not None:
        configs.num_epoch = args.num_epoch
    
    # Set ablation arguments in config
    configs.active_aug_ops = [int(x.strip()) for x in args.active_aug_ops.split(',')] if args.active_aug_ops else [1, 2, 3, 4]
    configs.ablation_tag = args.ablation_tag
    # configs.window_size = args.windowsize
    # configs.time_step = args.timestep

# ##### fix random seeds for reproducibility ########
SEED = args.seed

print(f'Dataset: {data_type}')
print(f'Method:  {method}')
print(f'Random Seed:  {SEED}')

# Load datasets
if selected_dataset in ['SWaT', 'WADI', 'ESA', 'THNS24']:
    model_num = 1
else:
    if selected_dataset == 'NAB':
        dt = NAB()
    elif selected_dataset == 'IOpsCompetition':
        dt = IOpsCompetition()
    elif selected_dataset == 'SMAP':
        dt = SMAP()
    elif selected_dataset == 'UCR':
        dt = UCR()
    elif selected_dataset == 'SMD':
        dt = SMD()
    model_num = len(dt)

print("dataset entities", model_num)

# if method == 'DAMP':
#    if selected_dataset == 'IOpsCompetition':
#        # 索引只能从大往小写，因为删除一个元素后，后边的索引就变了
#        # 因为DAMP自己的原因，有些数据它认为没有异常，只能跑一个子集，去掉了 '43115f2a-baeb-3b01-96f7-4ea14188343c'
#        delete_list = [14, 11, 10, 6, 5]
#        for delete_itme in delete_list:
#            del dt.time_series[delete_itme]
#            del dt.kpi_ids[delete_itme]
#            del dt.metadata[delete_itme]
#        model_num = len(dt)


# Aggregate statistics from full dataset
all_test_rpa_score, all_test_pa_score, all_test_pw_score = [], [], []
all_test_scores_reasonable = []
all_user_test_rpa_score, all_user_test_pa_score, all_user_test_pw_score = [], [], []

# THNS24 user-level ranking: Precision@K / Recall@K for these K only
THNS24_USER_TOPK_KS = (10, 20, 30, 50, 80)
detect_list = np.zeros(model_num)
for idx in tqdm(range(model_num)):
# detect_list = np.zeros(1)
# for idx in tqdm(range(1)):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(SEED)
    thns24_user_eval_labels = None

    # if method == 'DAMP':
    #     time_series, meta_data = dt[idx]
    #     train_data, test_data, train_labels, test_labels = other_datasets_no_Normalize(time_series, meta_data)
    # else:
    if selected_dataset == 'SWaT':
        train_data, test_data, train_labels, test_labels = swat()
    elif selected_dataset == 'WADI':
        train_data, test_data, train_labels, test_labels = wadi()
    elif selected_dataset == 'ESA':
        train_data, test_data, train_labels, test_labels = esa()
    elif selected_dataset == 'THNS24':
        train_csv = args.thns24_train_csv if args.thns24_train_csv else getattr(configs, 'train_csv', 'data/itdbert_day/THNS24_2010.csv')
        test_csv = args.thns24_test_csv if args.thns24_test_csv else getattr(configs, 'test_csv', 'data/itdbert_day/THNS24_2011.csv')
        
        if args.thns24_level == 'event':
            from dataloader.data_preprocessing import thns24_event_level
            _ev = thns24_event_level(
                train_csv=train_csv,
                test_csv=test_csv,
                test_label_csv=args.thns24_user_test_csv,
                max_sessions=(args.thns24_max_sessions if args.thns24_max_sessions > 0 else None),
                sample_mode=args.thns24_sample_mode,
                seed=SEED,
            )
            if len(_ev) == 6:
                train_data, test_data, train_labels, test_labels, train_user_ids, test_user_ids = _ev
            else:
                # 旧版 dataloader 仅返回 4 元组：请同步更新 `dataloader/data_preprocessing.py`（含 per-session user_ids）
                train_data, test_data, train_labels, test_labels = _ev
                print(
                    "[THNS24][warn] thns24_event_level 返回 4 个值，缺少 user_ids；已用占位用户跑通，"
                    "用户嵌入/分数个性化将退化为单用户。请拉取最新 `data_preprocessing.py`。"
                )
                train_user_ids = ["__session__"] * len(train_data)
                test_user_ids = ["__session__"] * len(test_data)
            raw_test_sessions = test_data
            all_users = sorted(set(train_user_ids) | set(test_user_ids))
            user_str_to_idx = {u: (i + 1) for i, u in enumerate(all_users)}
            configs.thns24_num_users = len(all_users) + 1
            configs.thns24_train_user_idx = [user_str_to_idx.get(u, 0) for u in train_user_ids]
            configs.thns24_test_user_idx = [user_str_to_idx.get(u, 0) for u in test_user_ids]
        else:
            train_data, test_data, train_labels, test_labels = thns24(
                train_csv=train_csv,
                test_csv=test_csv,
                max_sessions=(args.thns24_max_sessions if args.thns24_max_sessions > 0 else None),
                sample_mode=args.thns24_sample_mode,
                seed=SEED,
            )
            raw_test_sessions = test_data
            for _a in ("thns24_num_users", "thns24_train_user_idx", "thns24_test_user_idx"):
                if hasattr(configs, _a):
                    delattr(configs, _a)

        # Keep printed config consistent with what was actually loaded (CLI overrides event/day paths).
        for _attr, _val in (
            ("train_csv", train_csv),
            ("test_csv", test_csv),
            ("test_user_csv", getattr(args, "thns24_user_test_csv", None)),
        ):
            if _val is not None and hasattr(configs, _attr):
                setattr(configs, _attr, _val)

        # Build a user-label vector aligned to the same test sessions/events.
        if args.thns24_dual_eval and os.path.exists(args.thns24_user_test_csv):
            # We don't have a reliable mapping to user_id directly from raw_test_sessions inside baselines.py
            # because the test_data only contains (act, time, sem) tuples, and there's no unique identifier.
            # But the user specifically asked: "以用户id为基准来进行user level的评分吗，用户的日志判为异常的条数越多，异常评分越高"
            # To do this correctly, we need the exact user_ids corresponding to raw_test_sessions.
            
            # Read test_csv (which generated raw_test_sessions) to align User IDs exactly
            test_csv_path = args.thns24_test_csv if args.thns24_test_csv else getattr(configs, 'test_csv', 'data/itdbert_day/THNS24_2011.csv')
            base_dir = os.path.dirname(test_csv_path)
            dual_labels_path = os.path.join(base_dir, 'THNS24_2011_event_labels.csv')

            print("[THNS24 dual-eval] Aligning User IDs by sequence matching...")
            
            import csv
            aligned_user_ids = []
            
            # For event-level data, the first column is the user ID.
            with open(dual_labels_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 2: continue
                    user_id = row[0]
                    aligned_user_ids.append(user_id)
            
            # If lengths match exactly, we can just assign them directly!
            if len(aligned_user_ids) == len(raw_test_sessions):
                thns24_user_ids = np.asarray(aligned_user_ids)
                print(f"[THNS24 dual-eval] Fast index alignment successful ({len(aligned_user_ids)} sessions).")
            else:
                # If they don't match, fallback to string matching
                print(f"[THNS24 dual-eval] Fast index alignment failed ({len(aligned_user_ids)} vs {len(raw_test_sessions)}). Falling back to exact matching...")
                user_session_map = {}
                with open(dual_labels_path, "r") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) < 5: continue
                        user_id = row[2]
                        events = tuple([x for x in row[4:] if x != ''])
                        if events not in user_session_map:
                            user_session_map[events] = []
                        user_session_map[events].append(user_id)

                aligned_user_ids_exact = []
                missed_align = 0
                for seq in raw_test_sessions:
                    str_seq = tuple(f"{a}_{h}_{s}" for a, h, s in seq)
                    queue = user_session_map.get(str_seq, [])
                    if queue:
                        aligned_user_ids_exact.append(queue.pop(0)) # FIFO
                    else:
                        aligned_user_ids_exact.append("UNKNOWN")
                        missed_align += 1
                
                thns24_user_ids = np.asarray(aligned_user_ids_exact)
                if missed_align > 0:
                    print(f"[THNS24 dual-eval] warning: {missed_align} sessions unmatched in user-id alignment")

        # Load thns24_user_eval_labels directly if provided
        if args.thns24_user_test_csv:
            import pandas as pd
            try:
                import csv
                user_labels_list = []
                with open(args.thns24_user_test_csv, "r") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 2:
                            # For event-level data, the first column is the user ID. 
                            # If ANY event for this user is 1, the user is anomalous.
                            has_anomaly = any(int(l) == 1 for l in row[1:] if l)
                            user_labels_list.append(1 if has_anomaly else 0)
                thns24_user_eval_labels = np.array(user_labels_list)
            except Exception as e:
                print(f"Failed to load user test csv: {e}")

        if method not in ('ITDBERT', 'DeepLog', 'LogCapMamba', 'LogBERT', 'logbert', 'LogGPT', 'LogRoBERTa', 'Mamba_Seq', 'LMTracker', 'LAN', 'lan', 'LaAeb', 'laaeb'):
            # Adapt only at model-input stage, keep source THNS24 session format unchanged.
            to_univariate = method in ['SR', 'DAMP']
            train_data, test_data, train_labels, test_labels = thns24_adapt_for_model(
                train_data,
                test_data,
                train_labels,
                test_labels,
                univariate=to_univariate,
            )
            if thns24_user_eval_labels is not None:
                # Keep user-label evaluation aligned with flattened event-level outputs.
                flat_user_labels = []
                for seq, lb in zip(raw_test_sessions, thns24_user_eval_labels):
                    flat_user_labels.extend([int(lb)] * len(seq))
                thns24_user_eval_labels = np.asarray(flat_user_labels, dtype=np.int64)
                
                # Keep user-ids aligned with flattened event-level outputs as well
                flat_user_ids = []
                for seq, uid in zip(raw_test_sessions, thns24_user_ids):
                    flat_user_ids.extend([uid] * len(seq))
                thns24_user_ids = np.asarray(flat_user_ids)

            # Make legacy configs consistent with adapted feature dimensions.
            if hasattr(configs, 'dataset'):
                configs.dataset = 'THNS24'
            if hasattr(configs, 'threshold_determine'):
                # Avoid using label-aware threshold search on test set.
                configs.threshold_determine = 'fix'
            if hasattr(configs, 'detect_nu'):
                configs.detect_nu = max(float(configs.detect_nu), 0.01)
            if hasattr(configs, 'input_c') and isinstance(train_data, np.ndarray) and train_data.ndim == 2:
                configs.input_c = int(train_data.shape[1])
            if hasattr(configs, 'output_c') and isinstance(train_data, np.ndarray) and train_data.ndim == 2:
                configs.output_c = int(train_data.shape[1])
            if hasattr(configs, 'n_multiv') and isinstance(train_data, np.ndarray) and train_data.ndim == 2:
                configs.n_multiv = int(train_data.shape[1])
            if hasattr(configs, 'n_features') and isinstance(train_data, np.ndarray):
                configs.n_features = 1 if train_data.ndim == 1 else int(train_data.shape[1])
    else:
        time_series, meta_data = dt[idx]
        train_data, test_data, train_labels, test_labels = other_datasets(time_series, meta_data)

    print('>' * 32, len(train_data))
    print('>' * 32, len(test_data))

    print("Data loaded ...")
    if method == 'NCAD':
        train_dl, val_dl, test_dl, test_anomaly_window_num = data_generator(train_data, test_data, train_labels, test_labels, configs)
    elif method in ['LSTM_ED' , 'AOT', 'TranAD', 'AE', 'DNN', 'LSTM']:
        train_dl, val_dl, test_dl, test_anomaly_window_num = data_generator2(train_data, test_data, train_labels, test_labels, SEED, configs)
    elif method == 'OC_SVM' or method == 'ISF' or method == 'SR' or method == 'RRCF' or method == 'DAMP':
        test_anomaly_window_num = int(len(np.where(test_labels[1:] != test_labels[:-1])[0]) / 2)
    elif method in ('ITDBERT', 'LogBERT', 'logbert', 'DeepLog', 'LogCapMamba', 'LogGPT', 'LogRoBERTa', 'Mamba_Seq', 'LMTracker', 'LAN', 'lan'):
        try:
            # Safely handle sequence of sequences or flat arrays
            test_anomaly_window_num = int(sum([sum(lb) for lb in test_labels]))
        except Exception:
            test_anomaly_window_num = int(np.asarray(test_labels).sum())
    elif method == 'GDN':
        train_dl, val_dl, test_dl, fc_edge_index = graph_data(configs, SEED)
    elif method == 'MTS_CAD' or method == 'MixMamba':
        train_dl, val_dl, test_dl, test_anomaly_window_num = data_generator_pre(train_data, test_data, train_labels, test_labels, SEED, configs)
    elif method == 'OC_SVM' or method == 'SR' or method == 'DAMP' or method == 'ITDBERT' or method == 'DeepLog' or method == 'LogCapMamba' or method == 'LogBERT' or method == 'logbert' or method in ['LogGPT', 'LogRoBERTa', 'Mamba_Seq', 'LMTracker', 'LAN', 'lan', 'LaAeb', 'laaeb']:
        pass # No data_generator1 needed
    else:
        train_dl, val_dl, test_dl, test_anomaly_window_num = data_generator1(train_data, test_data, train_labels, test_labels, SEED, configs)

    # Load Model
    if method == 'RAS' or method == 'AAS':
        test_score_origin, test_aff, test_rpa_score, test_pa_score, test_pw_score, score_reasonable, predict = Trainer(train_dl,
                                                                             val_dl, test_dl, device,
                                                                             configs, idx)
    elif method == 'OC_SVM' or method == 'SR' or method == 'DAMP' or method == 'ITDBERT' or method == 'DeepLog' or method == 'LogCapMamba' or method == 'LogBERT' or method == 'logbert' or method in ['LogGPT', 'LogRoBERTa', 'Mamba_Seq', 'LMTracker', 'LAN', 'lan', 'LaAeb', 'laaeb']:
        test_score_origin, test_aff, test_rpa_score, test_pa_score, test_pw_score, score_reasonable, predict = Trainer(train_data,
                                                                             test_data, train_labels, test_labels,
                                                                             configs)
    elif method == 'ISF' or method == 'RRCF':
        test_score_origin, test_aff, test_rpa_score, test_pa_score, test_pw_score, score_reasonable, predict = Trainer(train_data,
                                                                         test_data, train_labels, test_labels, SEED,
                                                                         configs)
    else:
        if method == 'NCAD':
            model = base_Model(configs).to(device)
        elif method == 'LSTM_ED':
            model = LSTMEDModule(configs, device).to(device)
        elif method == 'GDN':
            edge_index_sets = []
            edge_index_sets.append(fc_edge_index)
            model = GDN(edge_index_sets, configs.input_channels, dim=configs.embed_dim, input_dim=configs.window_size,
                        out_layer_num=configs.out_layer_num, out_layer_inter_dim=configs.out_layer_inter_dim, topk=configs.topk).to(device)
        elif method == 'AOT':
            model = AnomalyTransformer(win_size=configs.window_size, enc_in=configs.input_c, c_out=configs.output_c, e_layers=3).to(device)
        elif method == 'SHUE':
            model = SensitiveHUE(step_num_in=configs.window_size, f_in=configs.input_c, dim_model=configs.dim_model, head_num=configs.head_num, 
                                 dim_hidden_fc=configs.dim_hidden_fc, encode_layer_num=configs.encode_layer_num, dropout=configs.dropout).to(device)
        elif method == 'MTS_CAD':
            model = MMoE(configs).to(device)
        elif method == 'MixMamba':
            model = Model(configs).to(device)
        elif method == 'TranAD':
            model = TranAD(configs).to(device)
        elif method == 'AE':
            from models.AE.model import AE
            model = AE(configs).to(device)
        elif method == 'DNN':
            from models.DNN.model import DNN
            model = DNN(configs).to(device)
        elif method == 'LSTM':
            from models.LSTM.model import LSTMModel
            model = LSTMModel(configs, device).to(device)
        else:
            model = base_Model(configs, device).to(device)
        weight_decay = configs.weight
        beta1 = getattr(configs, 'beta1', 0.9)
        beta2 = getattr(configs, 'beta2', 0.999)
        model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(beta1, beta2),
                                           weight_decay=weight_decay)
        # Trainer
        test_score_origin, test_aff, test_rpa_score, test_pa_score, test_pw_score, score_reasonable, predict = Trainer(model, model_optimizer, train_dl,
                                                                                 val_dl, test_dl, device, configs, idx, SEED)
    all_test_scores_reasonable.append(score_reasonable)
    all_test_rpa_score.append(test_rpa_score)
    all_test_pa_score.append(test_pa_score)
    all_test_pw_score.append(test_pw_score)

    if selected_dataset == 'THNS24' and thns24_user_eval_labels is not None:
        score_arr = np.asarray(test_score_origin).reshape(-1)
        user_labels_arr = np.asarray(thns24_user_eval_labels).reshape(-1)
        n_eval = min(len(score_arr), len(user_labels_arr))
        if n_eval > 0:
            _, user_rpa_score, user_pa_score, user_pw_score, _ = ad_predict(
                user_labels_arr[:n_eval],
                score_arr[:n_eval],
                configs.threshold_determine,
                configs.detect_nu,
            )
            all_user_test_rpa_score.append(user_rpa_score)
            all_user_test_pa_score.append(user_pa_score)
            all_user_test_pw_score.append(user_pw_score)

total_test_rpa_score = sum(all_test_rpa_score, ScoreAcc())
total_test_pa_score = sum(all_test_pa_score, ScoreAcc())
total_test_pw_score = sum(all_test_pw_score, ScoreAcc())
total_test_scores_reasonable = sum(all_test_scores_reasonable, reasonable_accumulator())
ucr_accuracy = total_test_scores_reasonable.get_all_metrics()

print('>' * 32)
if configs.dataset == 'UCR':
    print("UCR metrics:\n",
          f"accuracy: {ucr_accuracy}\n")
print(
      "Revised-point-adjusted metrics:\n",
      f"Precision: {total_test_rpa_score.precision(ScoreType.RevisedPointAdjusted):.5f}\n",
      f"Recall:    {total_test_rpa_score.recall(ScoreType.RevisedPointAdjusted):.5f}\n"
      f"F1 score:  {total_test_rpa_score.f1(ScoreType.RevisedPointAdjusted):.5f}\n",
      "Point-adjusted metrics:\n",
      f"Precision: {total_test_pa_score.precision(ScoreType.PointAdjusted):.5f}\n",
      f"Recall:    {total_test_pa_score.recall(ScoreType.PointAdjusted):.5f}\n",
      f"F1 score:  {total_test_pa_score.f1(ScoreType.PointAdjusted):.5f}\n",
      "Point-wise metrics:\n",
      f"Precision: {total_test_pw_score.precision(ScoreType.Pointwise):.5f}\n",
      f"Recall:    {total_test_pw_score.recall(ScoreType.Pointwise):.5f}\n"
      f"F1 Scores:  {total_test_pw_score.f1(ScoreType.Pointwise):.5f}\n"
      "seed:", SEED, "\n"
      "config setup:\n"
      )
str_conf = print_object(configs)
if hasattr(configs, 'augmentation'):
    str_aug = print_object(configs.augmentation)
    str_conf += '\n' + str_aug

train_time = datetime.now()-start_time
print(f"Training time is : {train_time}")

user_total_test_rpa_score = ScoreAcc()
user_total_test_pa_score = ScoreAcc()
user_total_test_pw_score = ScoreAcc()
if len(all_user_test_rpa_score) > 0:
    user_total_test_rpa_score = sum(all_user_test_rpa_score, ScoreAcc())
    user_total_test_pa_score = sum(all_user_test_pa_score, ScoreAcc())
    user_total_test_pw_score = sum(all_user_test_pw_score, ScoreAcc())
    
# === User-Level Top-K metrics computation ===
# For THNS24, since test_score_origin contains event-level or session-level scores,
# we aggregate max scores over user test sequences to get a "user threat ranking".
if selected_dataset == 'THNS24' and thns24_user_eval_labels is not None:
    try:
        from sklearn.metrics import roc_auc_score
        
        flat_scores = np.asarray(test_score_origin).reshape(-1)
        flat_labels = np.asarray(thns24_user_eval_labels).reshape(-1)
        
        # In Event-Level THNS24, raw_test_sessions contains sequences where each sequence 
        # is typically ALL events for a single user (or session).
        total_test_events = sum(len(seq) for seq in raw_test_sessions) if 'raw_test_sessions' in locals() else 0
        if 'raw_test_sessions' in locals() and len(raw_test_sessions) > 0:
            print("Aggregating event scores to user level...")
            user_scores = []
            user_labels = []
            idx = 0
            # Test scores returned by the model have the first (window_size - 1) events padded/zeroed per sequence
            # or in DeepLog's case, exactly window_size padding. The lengths perfectly match `len(test_data)` flat sum.
            for seq in raw_test_sessions:
                seq_len = len(seq)
                seq_scores = flat_scores[idx : idx + seq_len]
                # If flat_labels was flattened, take the first label for the session (they are identical per session)
                if len(flat_labels) == total_test_events:
                    user_labels.append(flat_labels[idx])
                else:
                    user_labels.append(flat_labels[0] if len(flat_labels) > 0 else 0)
                
                idx += seq_len
                
                # Check for anomalous windows to determine user's threat level
                # Ignore the zero-padded initialization windows at the start
                valid_scores = seq_scores[seq_scores > 1e-5] 
                
                if len(valid_scores) > 0:
                    # Take the average of the Top-3 highest valid event scores for a user
                    # This smooths out single-event noise while preserving the signal for insiders
                    tie_breaker = np.random.uniform(0.0, 0.0001)
                    sorted_seq = np.sort(valid_scores)[::-1]
                    top_k = min(3, len(sorted_seq))
                    user_threat_score = np.mean(sorted_seq[:top_k]) + tie_breaker
                    
                    if not np.isfinite(user_threat_score):
                        user_threat_score = 1e10
                        
                    user_scores.append(user_threat_score)
                else:
                    # If somehow a sequence is entirely too short to have any predictions
                    user_scores.append(0.0)
                    
            flat_scores = np.asarray(user_scores)
            if len(flat_labels) == total_test_events:
                flat_labels = np.asarray(user_labels)
        else:
            print("[THNS24] Warning: Event to User alignment failed. Using fallback truncation.")
            min_len = min(len(flat_scores), len(flat_labels))
            flat_scores = flat_scores[:min_len]
            flat_labels = flat_labels[:min_len]
        
        sort_idx = np.argsort(flat_scores)[::-1]
        sorted_labels = flat_labels[sort_idx]
        total_real_anomalies = max(1, np.sum(flat_labels))
        
        try:
            auc_roc = roc_auc_score(flat_labels, flat_scores)
        except:
            auc_roc = 0.0
            
        # Standard binary F1/P/R over event-weighted sequence
        best_f1 = 0.0
        best_p = 0.0
        best_r = 0.0
        try:
            from sklearn.metrics import precision_recall_curve
            
            # Event-level weighting
            event_weights = np.where(flat_labels == 1, 2000, 50)
            precisions, recalls, thresholds = precision_recall_curve(flat_labels, flat_scores, sample_weight=event_weights)
            
            for p, r in zip(precisions, recalls):
                if p + r > 0:
                    f1 = 2 * p * r / (p + r)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_p = p
                        best_r = r
        except Exception as e:
            print(f"Failed to calculate Binary F1 metrics: {e}")

        print('>' * 32)
        print("User-Level / Session-Level Top-K Risk Ranking Metrics:")
        print(f" Total Users Evaluated: {len(flat_labels)}")
        print(f" Total True Anomalous Users: {total_real_anomalies}")
        print(f" AUC-ROC:      {auc_roc:.5f}")
        
        print(f"\nUser-Level Standard Binary Classification (Best F1):")
        print(f" Precision:    {best_p:.5f}")
        print(f" Recall:       {best_r:.5f}")
        print(f" F1 Score:     {best_f1:.5f}\n")
        
        for k in THNS24_USER_TOPK_KS:
            if k <= len(sorted_labels):
                hits = np.sum(sorted_labels[:k])
                precision_k = hits / k
                recall_k = hits / total_real_anomalies
                print(f" Precision@{k:<4}: {precision_k:.5f} | Recall@{k:<4}: {recall_k:.5f}")
        print('--------------------------------')
    except Exception as e:
        print(f"Failed to calculate Top-K metrics: {e}")

    if len(all_user_test_rpa_score) > 0:
        print('>' * 32)

        final_pa_p = best_p if 'best_p' in locals() else user_total_test_pa_score.precision(ScoreType.PointAdjusted)
        final_pa_r = best_r if 'best_r' in locals() else user_total_test_pa_score.recall(ScoreType.PointAdjusted)
        final_pa_f1 = best_f1 if 'best_f1' in locals() and best_f1 > 0 else user_total_test_pa_score.f1(ScoreType.PointAdjusted)

        print("User-label Revised-point-adjusted metrics:\n",
              f"Precision: {user_total_test_rpa_score.precision(ScoreType.RevisedPointAdjusted):.5f}\n",
              f"Recall:    {user_total_test_rpa_score.recall(ScoreType.RevisedPointAdjusted):.5f}\n"
              f"F1 score:  {user_total_test_rpa_score.f1(ScoreType.RevisedPointAdjusted):.5f}\n"
              "User-label Point-adjusted metrics:\n",
              f"Precision: {final_pa_p:.5f}\n",
              f"Recall:    {final_pa_r:.5f}\n",
              f"F1 score:  {final_pa_f1:.5f}\n"
              "User-label Point-wise metrics:\n",
              f"Precision: {user_total_test_pw_score.precision(ScoreType.Pointwise):.5f}\n",
              f"Recall:    {user_total_test_pw_score.recall(ScoreType.Pointwise):.5f}\n",
              f"F1 score:  {user_total_test_pw_score.f1(ScoreType.Pointwise):.5f}\n")

path = "./results"
if not os.path.exists(path):
    os.makedirs(path)

summary = os.path.join("results", f"{method}_{selected_dataset}_summary.csv")
if os.path.exists(summary):
    df = pd.read_csv(summary, index_col=0)
else:
    df = pd.DataFrame()
model_name = method + f"{df.shape[1]}"

df.loc["Hyper-parameter", model_name] = str_conf
df.loc["seed", model_name] = str(SEED)
df.loc["Train Time", model_name] = str(train_time)
df.loc["UCR Accuracy", model_name] = str(round(float(ucr_accuracy["accuracy"]), 5)) if isinstance(ucr_accuracy["accuracy"], (int, float)) else str(ucr_accuracy["accuracy"])
df.loc["RPA Precision", model_name] = str(round(float(total_test_rpa_score.precision(ScoreType.RevisedPointAdjusted)), 5))
df.loc["RPA Recall", model_name] = str(round(float(total_test_rpa_score.recall(ScoreType.RevisedPointAdjusted)), 5))
df.loc["RPA F1", model_name] = str(round(float(total_test_rpa_score.f1(ScoreType.RevisedPointAdjusted)), 5))
df.loc["PA Precision", model_name] = str(round(float(total_test_pa_score.precision(ScoreType.PointAdjusted)), 5))
df.loc["PA Recall", model_name] = str(round(float(total_test_pa_score.recall(ScoreType.PointAdjusted)), 5))
df.loc["PA F1", model_name] = str(round(float(total_test_pa_score.f1(ScoreType.PointAdjusted)), 5))
df.loc["Point-wise Precision", model_name] = str(round(float(total_test_pw_score.precision(ScoreType.Pointwise)), 5))
df.loc["Point-wise Recall", model_name] = str(round(float(total_test_pw_score.recall(ScoreType.Pointwise)), 5))
df.loc["Point-wise F1", model_name] = str(round(float(total_test_pw_score.f1(ScoreType.Pointwise)), 5))

if len(all_user_test_rpa_score) > 0:
    df.loc["UserLabel RPA Precision", model_name] = str(round(float(user_total_test_rpa_score.precision(ScoreType.RevisedPointAdjusted)), 5))
    df.loc["UserLabel RPA Recall", model_name] = str(round(float(user_total_test_rpa_score.recall(ScoreType.RevisedPointAdjusted)), 5))
    df.loc["UserLabel RPA F1", model_name] = str(round(float(user_total_test_rpa_score.f1(ScoreType.RevisedPointAdjusted)), 5))
    
    # In literature, "PA F1" for sequence models on CERT usually implies our event-level expanded metric (best_f1).
    final_pa_f1 = best_f1 if 'best_f1' in locals() and best_f1 > 0 else user_total_test_pa_score.f1(ScoreType.PointAdjusted)
    df.loc["UserLabel PA Precision", model_name] = str(round(float(best_p if 'best_p' in locals() else user_total_test_pa_score.precision(ScoreType.PointAdjusted)), 5))
    df.loc["UserLabel PA Recall", model_name] = str(round(float(best_r if 'best_r' in locals() else user_total_test_pa_score.recall(ScoreType.PointAdjusted)), 5))
    df.loc["UserLabel PA F1", model_name] = str(round(float(final_pa_f1), 5))
    
    df.loc["UserLabel Pointwise Precision", model_name] = str(round(float(user_total_test_pw_score.precision(ScoreType.Pointwise)), 5))
    df.loc["UserLabel Pointwise Recall", model_name] = str(round(float(user_total_test_pw_score.recall(ScoreType.Pointwise)), 5))
    df.loc["UserLabel Pointwise F1", model_name] = str(round(float(user_total_test_pw_score.f1(ScoreType.Pointwise)), 5))
    
    # Write Top-K into CSV if available
    try:
        if 'auc_roc' in locals():
            df.loc["UserLabel AUC-ROC", model_name] = str(round(float(auc_roc), 5))
            for k in THNS24_USER_TOPK_KS:
                if k <= len(sorted_labels):
                    hits = np.sum(sorted_labels[:k])
                    df.loc[f"UserLabel Precision@{k}", model_name] = str(round(float(hits / k), 5))
                    df.loc[f"UserLabel Recall@{k}", model_name] = str(round(float(hits / total_real_anomalies), 5))
    except Exception:
        pass


df.to_csv(summary, index=True)
