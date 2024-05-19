import os
import pandas as pd
import numpy as np
import torch
import GPUtil
import pickle as pkl
from tqdm import tqdm
from torch import nn
from torch import optim
from model import Classifier
from sklearn.metrics import average_precision_score
import torch.utils.data as Data
from sklearn.model_selection import KFold
import argparse
import os

parser = argparse.ArgumentParser(description="""Main script of prottrans classifier.""")
parser.add_argument('--feat', type=int, default=1, help='path to the prottrans embedding.')
parser.add_argument('--label', type=int, default=6, help='path to the label.')
parser.add_argument('--outpth', type=int, default=6, help='path to the output folder.')
inputs = parser.parse_args()
 
if not os.path.isdir(inputs.outpth):
    os.makedirs(inputs.outpth)

embed = pkl.load(open(f'{inputs.feat}', 'rb'))
label = pkl.load(open(f'{inputs.label}', 'rb'))

embed_size = embed.shape[1]
batch_size = 64

def return_batch(embed, label, batch_size=64, flag=False):
    X = torch.from_numpy(embed)
    y = torch.from_numpy(label).float()
    train_dataset = Data.TensorDataset(X, y)
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=flag,
        num_workers=0,
    )
    return train_loader

    
    
     
def cal_Fmax(test_loader):
    y_true = []
    probs = []
    _ = model.eval()
    for step, (batch_x, batch_y) in enumerate(test_loader):
        scores = model(batch_x.to(device))
    
        for i in torch.arange(0, batch_y.shape[0]):
            y_true.append(batch_y[i].detach().cpu().numpy())
            probs.append(scores[i].detach().cpu().numpy())
    probs = np.array(probs)
    y_true = np.array(y_true)
    
    # delete zero-columns, if the test set does not have this label
    deleted_idx = []
    for i in range(y_true.shape[1]):
        true_labels_i = y_true[:, i]
        if(np.all(true_labels_i == 0)):
            deleted_idx.append(i)
    tmp = sorted(deleted_idx, reverse=True)

    print(f'deleted labels: {len(deleted_idx)}', deleted_idx)
    probs = np.delete(probs, deleted_idx, axis=1)
    y_true = np.delete(y_true, deleted_idx, axis=1)
    print(probs.shape, y_true.shape)
    
    f1s = []
    for threshold in np.arange(0, 1.01, 0.01):
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= threshold)] = 1

        prs, rcs = [], []
        for pred, label in zip(y_pred, y_true):
            pred_num = np.count_nonzero(pred == 1)
            ans_num = np.count_nonzero(label == 1)
            if(ans_num==0):
                continue
            if(pred_num==0):
                rc = 0
                rcs.append(rc)
                continue
            else:
                identical = np.count_nonzero((pred==label)&(pred!=0))
                pr = identical/pred_num
                rc = identical/ans_num
            prs.append(pr)
            rcs.append(rc)
        avg_precision = np.mean(prs)
        avg_recall = np.mean(rcs)
        avg_f1 = (2*avg_precision*avg_recall)/(avg_precision+avg_recall)
        f1s.append(avg_f1)
    
    # calculate AUPR
    num_labels = y_true.shape[1]
    aupr_scores = []
    idx = -1
    for i in range(num_labels):
        idx+=1
        true_labels_i = y_true[:, i]
        predicted_probabilities_i = probs[:, i]
        aupr = average_precision_score(true_labels_i, predicted_probabilities_i)
        aupr_scores.append(aupr)
    mean_aupr = np.mean(aupr_scores)

    x, y = round(max(f1s), 4), round(mean_aupr, 4)
    print('Fmax/AUPR:', f'{x}/{y}')
    return x, y



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean_model():
    model = Classifier(
        embed_size=embed_size,
        out = 5585
    ).to(device)
    print(model)
    loss_func = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    _=model.to(device)
    return model, loss_func, optimizer




kf = KFold(n_splits=10, shuffle=True, random_state=42)
eval_max_Fmax = []
eval_max_AUC = []
for i, (train_index, test_index) in enumerate(kf.split(embed)):
    train_data = embed[train_index]
    test_data = embed[test_index]
    train_label = label[train_index]
    test_label = label[test_index]
    train_loader = return_batch(train_data, train_label, batch_size, True)
    test_loader = return_batch(test_data, test_label, batch_size, False)

    model, loss_func, optimizer = clean_model()

    pbar = tqdm(range(20))
    max_Fmax = 0
    maxAUC = 0
    for epoch in pbar:
        _ = model.train()
        losses = []
        count = 0
        y_true = []
        y_pred = []
        for step, (batch_x, batch_y) in enumerate(train_loader):
            #GPUtil.showUtilization()
            scores = model(batch_x.to(device))
            scores = scores.squeeze()
            loss = loss_func(scores, batch_y.to(device))
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mean_loss = sum(losses) / len(losses)
        pbar.set_description(f'Loss at epoch {epoch} was {mean_loss:.5f}')
        Fmax, AUC = cal_Fmax(test_loader)
        if Fmax > max_Fmax:
            max_Fmax = Fmax
            maxAUC = AUC
    eval_max_Fmax.append(max_Fmax)
    eval_max_AUC.append(maxAUC)

pkl.dump(eval_max_Fmax, open(f'{inputs.outpth}/prottrans_Fmax.pkl', 'wb'))
pkl.dump(eval_max_AUC, open(f'{inputs.outpth}/prottrans_AUC.pkl', 'wb'))




