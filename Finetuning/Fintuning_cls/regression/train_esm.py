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
import torch.utils.data as Data
from sklearn.model_selection import KFold
import argparse
import os

parser = argparse.ArgumentParser(description="""Main script of EMS classifier.""")
parser.add_argument('--feat', type=str, help='path to the ems embedding.')
parser.add_argument('--label', type=str, help='path to the label.')
parser.add_argument('--outpth', type=str, help='path to the output folder.')
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

    
    
     
def return_metric(test_loader):
    y_true = []
    y_pred = []
    
    _ = model.eval()
    for step, (batch_x, batch_y) in enumerate(test_loader):
        scores = model(batch_x.to(device))
        scores = scores.squeeze()
    
        for i in torch.arange(0, batch_y.shape[0]):
            y_true.append(batch_y[i].detach().cpu().numpy())
            y_pred.append(scores[i].detach().cpu().numpy())
    
    return np.corrcoef(y_pred, y_true)[0, 1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean_model():
    model = Classifier(
        embed_size=embed_size,
        out = 1
    ).to(device)
    print(model)
    loss_func = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    _=model.to(device)
    return model, loss_func, optimizer




kf = KFold(n_splits=10, shuffle=True, random_state=42)
eval_metric = []
for i, (train_index, test_index) in enumerate(kf.split(embed)):
    train_data = embed[train_index]
    test_data = embed[test_index]
    train_label = label[train_index]
    test_label = label[test_index]
    train_loader = return_batch(train_data, train_label, batch_size, True)
    test_loader = return_batch(test_data, test_label, batch_size, False)

    model, loss_func, optimizer = clean_model()

    pbar = tqdm(range(20))
    max_person_r = 0
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

            for i in torch.arange(0, batch_y.shape[0]):
                y_true.append(batch_y[i].detach().cpu().numpy())
                y_pred.append(scores[i].detach().cpu().numpy())

        mean_loss = sum(losses) / len(losses)
        pbar.set_description(f'Loss at epoch {epoch} was {mean_loss:.5f}. Pearson R: {np.corrcoef(y_pred, y_true)[0, 1]}')
        person_r = return_metric(test_loader)
        if person_r > max_person_r:
            max_person_r = person_r
    eval_metric.append(max_person_r)
pkl.dump(eval_metric, open(f'{inputs.outpth}/esm_results.pkl', 'wb'))


