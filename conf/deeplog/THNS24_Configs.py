class Config(object):
    def __init__(self):
        # dataset
        self.dataset = "THNS24"
        # Defaults match `baselines.py --thns24_level event`; override via CLI if needed.
        self.train_csv = "data/itdbert_event/THNS24_2010.csv"
        self.test_csv = "data/itdbert_event/THNS24_2011.csv"
        self.test_user_csv = "data/itdbert_event/THNS24_2011_event_labels.csv"

        # model/train
        self.device = "cuda"
        self.window_size = 64
        self.num_epoch = 30
        self.batch_size = 1024  # Increased for Event-Level speed
        self.hidden_size = 128
        self.num_layers = 2
        self.dropout = 0.1
        self.user_emb_dim = 16
        # 分数校准：相对「该用户在训练集正常事件上的分数分布」做 z，再混入少量全局 z
        self.thns24_score_personalize = True
        # 提高全局权重，避免完全被用户局部方差压扁
        self.thns24_score_global_weight = 0.5
        self.thns24_zscore_shrink_n = 200.0
        self.learning_rate = 3e-3  # Further Increased to match larger batch size
        self.patience = 5
        self.model_path = "best_network/THNS24/deeplog_lstm.pt"

        # DeepLog detect
        self.num_candidates = 2
        self.tolerance = 1

        # keep compatibility fields for common pipeline output
        self.threshold_determine = "best_f1"
        self.detect_nu = 0.05
        self.time_step = 1

