class Config(object):
    def __init__(self):
        self.dataset = 'THNS24'
        self.n_features = 1
        self.hidden_size = 64
        self.window_size = 10
        self.time_step = 1
        self.num_epoch = 10
        self.batch_size = 512
        self.lr = 1e-3
        self.weight = 1e-4
        self.nu = 0.05
        self.warmup = 0
        self.train_method = 'blind'
        self.threshold_determine = 'floating'
        self.detect_nu = 0.05
        self.drop_last = False
