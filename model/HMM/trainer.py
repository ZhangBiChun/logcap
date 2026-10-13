import sys
import os
import numpy as np
from hmmlearn import hmm
from merlion.evaluate.anomaly import ScoreType
from models import ad_predict
from models.reasonable_metric import tsad_reasonable
from models.reasonable_metric import reasonable_accumulator
from scipy.special import logsumexp

def Trainer(train_data, test_data, train_labels, test_labels, seed, config):
    print("HMM Training started ....")
    
    # Gaussian HMM for continuous data
    model = hmm.GaussianHMM(n_components=config.n_components, covariance_type="diag", n_iter=20, random_state=seed)
    
    model.fit(train_data)
    print("HMM Training ended ....")

    print("predicting started ....")
    # For anomaly detection point-wise, we compute the negative log likelihood of each point
    log_likelihoods = model._compute_log_likelihood(test_data)
    test_score_origin = -logsumexp(log_likelihoods, axis=1)

    if config.dataset == 'UCR':
        test_affiliation, test_rpa_score, test_pa_score, test_pw_score, predict = ad_predict(test_labels, test_score_origin, config.threshold_determine, config.detect_nu)
        score_reasonable = tsad_reasonable(test_labels, predict, config.time_step)
    else:
        test_affiliation, test_rpa_score, test_pa_score, test_pw_score, predict = ad_predict(test_labels, test_score_origin, config.threshold_determine, config.detect_nu)
        score_reasonable = reasonable_accumulator(1, 0)

    print("Test affiliation-metrics")
    print(f'Test precision: {test_affiliation["precision"]:2.4f}  | \tTest recall: {test_affiliation["recall"]:2.4f}\n')
    
    test_pa_f1 = test_pa_score.f1(ScoreType.PointAdjusted)
    print(f'Test PA F1: {test_pa_f1:2.4f}')

    return test_score_origin, test_affiliation, test_rpa_score, test_pa_score, test_pw_score, score_reasonable, predict
