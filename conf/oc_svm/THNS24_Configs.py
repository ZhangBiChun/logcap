class Config(object):
    def __init__(self):
        # datasets
        self.dataset = 'THNS24'
        # model configs
        self.kernel = 'rbf'
        self.nu = 0.05
        self.degree = 3
        self.gamma = 'scale'

        self.time_step = 1
        # Anomaly quantile of fixed threshold
        self.detect_nu = 0.05
        # Methods for determining thresholds ("direct","fix","floating","one-anomaly")
        self.threshold_determine = 'floating'
