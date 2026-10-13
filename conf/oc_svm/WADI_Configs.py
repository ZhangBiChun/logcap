class Config(object):
    def __init__(self):
        # datasets
        self.dataset = 'WADI'
        # model configs
        self.kernel = 'rbf'
        self.nu = 0.0001
        self.degree = 3
        self.gamma = 'scale'
        # Anomaly quantile of fixed threshold
        self.detect_nu = 0.0015
        # Methods for determining thresholds ("direct","fix","floating","one-anomaly")
        self.threshold_determine = 'floating'

