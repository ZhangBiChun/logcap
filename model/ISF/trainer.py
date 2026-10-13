import sys
import os
import numpy as np
from sklearn.ensemble import IsolationForest
from merlion.evaluate.anomaly import ScoreType
from models import ad_predict
from models.reasonable_metric import tsad_reasonable
from models.reasonable_metric import reasonable_accumulator


def Trainer(train_data, test_data, train_labels, test_labels, seed, config):
    # Start training
    print("Training started ....")
    skl_isolation_forest = IsolationForest(
        max_samples=config.max_n_samples,
        n_estimators=config.n_estimators,
        random_state=seed
    )

    skl_isolation_forest.fit(train_data)
    print("Training ended ....")
    # test_score_origin = skl_isolation_forest.decision_function(test_data)

    # Return negative of model's score, since model scores are in (0, -1],
    # where more negative = more anomalous
    test_score_origin = -skl_isolation_forest.score_samples(test_data)

    print("predicting started ....")
    if config.dataset == 'UCR':
        test_affiliation, test_rpa_score, test_pa_score, test_pw_score, predict = ad_predict(test_labels, test_score_origin, config.threshold_determine,
                                                   config.detect_nu)
        score_reasonable = tsad_reasonable(test_labels, predict, config.time_step)

    else:
        test_affiliation, test_rpa_score, test_pa_score, test_pw_score, predict = ad_predict(test_labels,
                                                                                             test_score_origin,
                                                                                             config.threshold_determine,
                                                                                             config.detect_nu)
        score_reasonable = reasonable_accumulator(1, 0)


    print("Test affiliation-metrics")
    print(
        f'Test precision: {test_affiliation["precision"]:2.4f}  | \tTest recall: {test_affiliation["recall"]:2.4f}\n')
    test_rpa_f1 = test_rpa_score.f1(ScoreType.RevisedPointAdjusted)
    test_rpa_precision = test_rpa_score.precision(ScoreType.RevisedPointAdjusted)
    test_rpa_recall = test_rpa_score.recall(ScoreType.RevisedPointAdjusted)
    print("Test RAP F1")
    print(f'Test F1: {test_rpa_f1:2.4f}  | \tTest precision: {test_rpa_precision:2.4f}  | \tTest recall: {test_rpa_recall:2.4f}\n')

    test_pa_f1 = test_pa_score.f1(ScoreType.PointAdjusted)
    test_pa_precision = test_pa_score.precision(ScoreType.PointAdjusted)
    test_pa_recall = test_pa_score.recall(ScoreType.PointAdjusted)
    print("Test PA F1")
    print(
        f'Test F1: {test_pa_f1:2.4f}  | \tTest precision: {test_pa_precision:2.4f}  | \tTest recall: {test_pa_recall:2.4f}\n')

    test_pw_f1 = test_pw_score.f1(ScoreType.PointAdjusted)
    test_pw_precision = test_pw_score.precision(ScoreType.PointAdjusted)
    test_pw_recall = test_pw_score.recall(ScoreType.PointAdjusted)
    print("Test PW F1")
    print(
        f'Test F1: {test_pw_f1:2.4f}  | \tTest precision: {test_pw_precision:2.4f}  | \tTest recall: {test_pw_recall:2.4f}\n')

    return test_score_origin, test_affiliation, test_rpa_score, test_pa_score, test_pw_score, score_reasonable, predict

