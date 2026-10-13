class Config(object):
    def __init__(self):
        # dataset
        self.dataset = "THNS24"
        # Match DeepLog / ITDBERT / LogBERT THNS24 event baseline (`--thns24_level event`).
        self.train_csv = "data/itdbert_event/THNS24_2010.csv"
        self.test_csv = "data/itdbert_event/THNS24_2011.csv"
        self.test_user_csv = "data/itdbert_event/THNS24_2011_event_labels.csv"

        # model/train
        self.device = "cuda"
        self.window_size = 64
        self.num_epoch = 30
        self.batch_size = 1024
        self.hidden_size = 128
        self.num_layers = 2
        self.dropout = 0.1
        self.learning_rate = 3e-3
        self.patience = 5
        self.model_path = "best_network/THNS24/logcapmamba.pt"

        # --- Ablation switches (default = main result: BCE, no inference smoothing) ---
        # Set ablation_tag to a short name so each run saves a separate checkpoint, e.g.
        # "bce", "focal", "smooth3", "focal_smooth3". Leave "" for logcapmamba.pt.
        self.ablation_tag = ""
        self.use_focal_loss = False
        self.focal_alpha = 0.9
        self.focal_gamma = 2.0
        self.inference_score_smooth_window = 0  # 0=off; 3=length-3 moving average on event scores

        # keep compatibility fields for common pipeline output
        self.threshold_determine = "best_f1"
        self.detect_nu = 0.05
        self.time_step = 1
