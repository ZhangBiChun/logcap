import sys
import os
import numpy as np
import torch
from merlion.evaluate.anomaly import ScoreType
from models import ad_predict
from models.reasonable_metric import tsad_reasonable
from models.reasonable_metric import reasonable_accumulator
from models.LSTMED.trainer.early_stopping import EarlyStopping

def Trainer(model, model_optimizer, train_dl, val_dl, test_dl, device, config, idx, seed=0):
    print(f"Training started for {config.dataset} ....")
    save_path = "./best_network/" + config.dataset
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')
    early_stopping = EarlyStopping(save_path, idx, patience=10)

    checkpoint_file = os.path.join(save_path, str(idx) + f'_checkpoint.pth')
    
    for epoch in range(1, config.num_epoch + 1):
        train_loss = model_train(model, model_optimizer, train_dl, config, device, epoch, seed)
        val_target, val_score_origin, val_loss = model_evaluate(model, val_dl, config, device)
        test_target, test_score_origin, test_loss = model_evaluate(model, test_dl, config, device)

        scheduler.step(train_loss)
        print(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:.4f}\t | \n'
                     f'Valid Loss     : {val_loss:.4f}\t  | \n'
                     f'Test Loss     : {test_loss:.4f}\t  | \n'
                     )
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
        elif config.dataset in ['SWaT', 'WADI', 'THNS24']:
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

    val_affiliation, val_score, _, _, _ = ad_predict(val_target, val_score_origin, config.threshold_determine, config.detect_nu)
    test_affiliation, test_rpa_score, test_pa_score, test_pw_score, predict = ad_predict(test_target, test_score_origin, config.threshold_determine, config.detect_nu)
    score_reasonable = reasonable_accumulator(1, 0)
    
    return test_score_origin, test_affiliation, test_rpa_score, test_pa_score, test_pw_score, score_reasonable, predict

def model_train(model, model_optimizer, train_loader, config, device, epoch, seed):
    total_loss = []
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.float().to(device), target.long().to(device)
        data = data.permute(0, 2, 1)  # (batch, window_size, n_features)
        
        model_optimizer.zero_grad()
        output = model(data)
        
        loss_func = torch.nn.MSELoss(reduction="none")
        loss_n = loss_func(output, data)
        loss_n = loss_n.view(-1, config.n_features).mean(1)
        loss = torch.mean(loss_n)

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
        for data, target in test_dl:
            data, target = data.float().to(device), target.long().to(device)
            data = data.permute(0, 2, 1)
            
            output = model(data)
            loss_func = torch.nn.MSELoss(reduction="none")
            loss = loss_func(output, data)
            loss = loss.view(-1, config.n_features).mean(1)
            
            all_predict.extend(loss.tolist())
            if target.dim() == 2:
                all_target.extend(target[:, 0].tolist())
            elif target.dim() == 3:
                all_target.extend(target[:, 0, 0].tolist())
            else:
                all_target.extend(target.view(-1).tolist())
            total_loss.append(torch.mean(loss).item())

    total_loss = torch.tensor(total_loss).mean()
    
    # Pad to match original sequence length (window_size - 1 elements dropped by subsequences)
    if config.dataset == 'THNS24' and config.time_step == 1:
        pad_len = config.window_size - 1
        all_predict = [0.0] * pad_len + all_predict
        all_target = [0] * pad_len + all_target
        
    return all_target, all_predict, total_loss
