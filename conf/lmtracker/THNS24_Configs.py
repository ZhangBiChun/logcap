class Config(object):
    def __init__(self):
        # dataset
        self.dataset = 'THNS24'
        self.train_csv = 'data/itdbert_event/THNS24_2010.csv'
        self.test_csv = 'data/itdbert_event/THNS24_2011.csv'
        self.test_user_csv = 'data/itdbert_event/THNS24_2011_event_labels.csv'

        # model / train
        self.device = 'cuda'
        self.window_size = 64
        self.num_epoch = 30
        self.batch_size = 1024
        self.hidden_size = 128
        self.learning_rate = 3e-3
        self.patience = 5
        self.model_path = 'best_network/THNS24/lmtracker.pt'

        self.thns24_score_personalize = True
        self.thns24_score_global_weight = 0.5
        self.thns24_zscore_shrink_n = 200.0

        # detect
        self.num_candidates = 2
        self.threshold_determine = 'best_f1'
        self.detect_nu = 0.05
        self.time_step = 1
        self.drop_last = False