import sys
import numpy as np
import torch
from merlion.evaluate.anomaly import ScoreType
from models import ad_predict
from models.reasonable_metric import tsad_reasonable
from models.reasonable_metric import reasonable_accumulator
from .early_stopping import EarlyStopping


def Trainer(model, model_optimizer, train_dl, val_dl, test_dl, device, config, idx, seed=0):
    # Start training
    print("Training started ....")
    save_path = "./best_network/" + config.dataset
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')
    early_stopping = EarlyStopping(save_path, idx, patience=100)

    all_epoch_train_loss, all_epoch_val_loss, all_epoch_test_loss = [], [], []
    length = torch.tensor(0, device=device)  # radius R initialized with 0 by default.

    checkpoint_file = os.path.join(save_path, str(idx) + '_checkpoint.pth')
    if os.path.exists(checkpoint_file):
        print(f"\n======================================")
        print(f"Found existing model at {checkpoint_file}, skipping training...")
        print(f"======================================\n")
        model.load_state_dict(torch.load(checkpoint_file, map_location=device))
        val_target, val_score_origin, val_loss = model_evaluate(model, val_dl, config, device)
        test_target, test_score_origin, test_loss = model_evaluate(model, test_dl, config, device)
    else:
        for epoch in range(1, config.num_epoch + 1):
            # Train and validate
            train_loss = model_train(model, model_optimizer, train_dl, length, config, device, epoch, seed)
            val_target, val_score_origin, val_loss = model_evaluate(model, val_dl, config, device)
            test_target, test_score_origin, test_loss = model_evaluate(model, test_dl, config, device)
    
            scheduler.step(train_loss)
            print(f'\nEpoch : {epoch}\n'
                         f'Train Loss     : {train_loss:.4f}\t | \n'
                         f'Valid Loss     : {val_loss:.4f}\t  | \n'
                         f'Test Loss     : {test_loss:.4f}\t  | \n'
                         )
            all_epoch_train_loss.append(train_loss.item())
            all_epoch_val_loss.append(val_loss.item())
            all_epoch_test_loss.append(test_loss.item())
            if config.dataset == 'UCR':
                val_affiliation, val_score, _, _, _ = ad_predict(val_target, val_score_origin, config.threshold_determine,
                                                           config.detect_nu)
                test_affiliation, test_score, _, _, predict = ad_predict(test_target, test_score_origin, config.threshold_determine,
                                                           config.detect_nu)
                score_reasonable = tsad_reasonable(test_target, predict, config.time_step)
                indicator = test_score.f1(ScoreType.RevisedPointAdjusted)
                early_stopping(score_reasonable, test_affiliation, test_score, indicator, val_score_origin,
                               test_score_origin, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
            elif config.dataset == 'SWaT' or config.dataset == 'WADI' or config.dataset == 'THNS24':
                early_stopping(0, 0, 0, -val_loss.item(), val_score_origin, test_score_origin, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

        if config.dataset == 'UCR':
            score_reasonable = early_stopping.best_score_reasonable
            test_score_origin = early_stopping.best_predict2
        elif config.dataset in ['SWaT', 'WADI', 'THNS24']:
            val_score_origin = early_stopping.best_predict1
            test_score_origin = early_stopping.best_predict2
            print('best loss: {:.4f}'.format(early_stopping.best_indicator))
        val_affiliation, val_score, _, _, _ = ad_predict(val_target, val_score_origin, config.threshold_determine,
                                                         config.detect_nu)
        test_affiliation, test_rpa_score, test_pa_score, test_pw_score, predict = ad_predict(test_target,
                                                                                             test_score_origin,
                                                                                             config.threshold_determine,
                                                                                             config.detect_nu)
        score_reasonable = reasonable_accumulator(1, 0)
        val_f1 = val_score.f1(ScoreType.RevisedPointAdjusted)
        val_precision = val_score.precision(ScoreType.RevisedPointAdjusted)
        val_recall = val_score.recall(ScoreType.RevisedPointAdjusted)
        print("Valid affiliation-metrics")
        print(
            f'Test precision: {val_affiliation["precision"]:2.4f}  | \tTest recall: {val_affiliation["recall"]:2.4f}\n')
        print("Valid RAP F1")
        print(
            f'Valid F1: {val_f1:2.4f}  | \tValid precision: {val_precision:2.4f}  | \tValid recall: {val_recall:2.4f}\n')


    else:
        val_affiliation, val_score, _, _, _ = ad_predict(val_target, val_score_origin, config.threshold_determine,
                                                         config.detect_nu)
        test_affiliation, test_rpa_score, test_pa_score, test_pw_score, predict = ad_predict(test_target,
                                                                                             test_score_origin,
                                                                                             config.threshold_determine,
                                                                                             config.detect_nu)
        score_reasonable = reasonable_accumulator(1, 0)
        val_f1 = val_score.f1(ScoreType.RevisedPointAdjusted)
        val_precision = val_score.precision(ScoreType.RevisedPointAdjusted)
        val_recall = val_score.recall(ScoreType.RevisedPointAdjusted)
        print("Valid affiliation-metrics")
        print(
            f'Test precision: {val_affiliation["precision"]:2.4f}  | \tTest recall: {val_affiliation["recall"]:2.4f}\n')
        print("Valid RAP F1")
        print(
            f'Valid F1: {val_f1:2.4f}  | \tValid precision: {val_precision:2.4f}  | \tValid recall: {val_recall:2.4f}\n')


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


def model_train(model, model_optimizer, train_loader, length, config, device, epoch, seed):

    total_loss, total_f1, total_precision, total_recall = [], [], [], []

    model.train()
    # torch.autograd.set_detect_anomaly(True)
    for batch_idx, (data, target) in enumerate(train_loader):
        # send to device
        data, target = data.float().to(device), target.long().to(device)
        data = data.permute(0, 2, 1)
        # optimizer
        model_optimizer.zero_grad()
        output = model(data)
        loss_func = torch.nn.MSELoss(reduction="none")
        loss_n = loss_func(output, data)
        loss_n = loss_n.view(-1, config.n_features).mean(1)
        loss_n[loss_n < 0.001] = 0.001
        loss_a = 1 / loss_n

        score = loss_n - loss_a
        # if epoch <= config.warmup:
        if epoch < config.warmup:
            loss = torch.mean(loss_n)
        else:
            if config.train_method == 'soft_boundary':
                diff1 = loss_n - length
                loss = length + (1 / config.nu) * torch.mean(torch.max(torch.zeros_like(diff1), diff1))
            elif config.train_method == 'loe_ts':
                _, idx_n = torch.topk(score, int(score.shape[0] * (1 - config.nu)), largest=False,
                                      sorted=False)
                _, idx_a = torch.topk(score, int(score.shape[0] * config.nu), largest=True,
                                      sorted=False)
                loss = torch.cat([loss_n[idx_n], config.mu * loss_a[idx_a]], 0)
                loss = loss.mean()
            elif config.train_method == 'loe_soft':
                _, idx_n = torch.topk(score, int(score.shape[0] * (1 - config.nu)), largest=False, sorted=False)
                _, idx_a = torch.topk(score, int(score.shape[0] * config.nu), largest=True, sorted=False)
                loss = torch.cat([loss_n[idx_n], 0.5 * loss_n[idx_a] + 0.5 * loss_a[idx_a]], 0)
                loss = loss.mean()
            elif config.train_method == 'refine':
                _, idx_n = torch.topk(loss_n, int(loss_n.shape[0] * (1 - config.nu)), largest=False,
                                      sorted=False)
                loss = loss_n[idx_n]
                loss = loss.mean()
            elif config.train_method == 'blind':
                loss = torch.mean(loss_n)

        # Update hypersphere radius R on mini-batch distances
        if (config.train_method == 'soft_boundary') and (epoch >= config.freeze_length_epoch):
            length = torch.tensor(get_radius(loss_n, config.nu), device=device)

        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()

    total_loss = torch.tensor(total_loss).mean()

    return total_loss


def model_evaluate(model, test_dl, config, device):
    model.eval()
    total_loss = []
    all_target, all_predict = [], []
    with torch.no_grad():
        sample_num = 0
        for data, target in test_dl:
            data, target = data.float().to(device), target.long().to(device)
            data = data.permute(0, 2, 1)
            output = model(data)
            error = torch.nn.L1Loss(reduction="none")(output, data)
            loss_func = torch.nn.MSELoss(reduction="mean")
            loss = loss_func(output, data)
            sample_num += error.shape[0]
            score = np.mean(error.view(-1, config.n_features).data.cpu().numpy(), axis=1)
            total_loss.append(loss.item())
            target = target.reshape(-1)
            all_target.extend(target.detach().cpu().numpy())
            all_predict.extend(score.reshape(-1))
    mean = np.mean(all_predict)
    std = np.std(all_predict)
    all_predict = (all_predict - mean)/std
    all_target = np.array(all_target)
    total_loss = torch.tensor(total_loss).mean()

    return all_target, all_predict, total_loss


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    # return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
    dist = dist.reshape(-1)
    return np.quantile(dist.clone().data.cpu().numpy(), 1 - nu)
