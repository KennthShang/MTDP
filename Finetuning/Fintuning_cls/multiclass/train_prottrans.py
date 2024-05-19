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
from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score
import torch.utils.data as Data
from sklearn.model_selection import KFold
import argparse


parser = argparse.ArgumentParser(description="""Main script of prottrans classifier.""")
parser.add_argument('--feat', type=int, default=1, help='path to the prottrans embedding.')
parser.add_argument('--label', type=int, default=6, help='path to the label.')
parser.add_argument('--outpth', type=int, default=6, help='path to the output folder.')
inputs = parser.parse_args()
 


embed = pkl.load(open(f'{inputs.feat}', 'rb'))
label = pkl.load(open(f'{inputs.label}', 'rb'))

embed_size = embed.shape[1]
batch_size = 64

def return_batch(embed, label, batch_size=64, flag=False):
    X = torch.from_numpy(embed)
    y = torch.from_numpy(label).long()
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
        predictions = np.argmax(scores.detach().cpu().numpy(), axis=1)
    
        for i in torch.arange(0, batch_y.shape[0]):
            y_true.append(batch_y[i].detach().cpu().numpy())
            y_pred.append(predictions[i])
    
    #print(classification_report(y_true, y_pred))
    return balanced_accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='weighted')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean_model():
    model = Classifier(
        embed_size=embed_size,
        out = 10
    ).to(device)
    print(model)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    _=model.to(device)
    return model, loss_func, optimizer




kf = KFold(n_splits=10, shuffle=True, random_state=42)
eval_F1 = []
eval_acc = []
for i, (train_index, test_index) in enumerate(kf.split(embed)):
    train_data = embed[train_index]
    test_data = embed[test_index]
    train_label = label[train_index]
    test_label = label[test_index]
    train_loader = return_batch(train_data, train_label, batch_size, True)
    test_loader = return_batch(test_data, test_label, batch_size, False)

    model, loss_func, optimizer = clean_model()

    pbar = tqdm(range(20))
    max_f1 = 0
    max_acc = 0
    for epoch in pbar:
        _ = model.train()
        losses = []
        count = 0
        y_true = []
        y_pred = []
        for step, (batch_x, batch_y) in enumerate(train_loader):
            #GPUtil.showUtilization()
            scores = model(batch_x.to(device))
            loss = loss_func(scores, batch_y.to(device))
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prediction = np.argmax(scores.detach().cpu().numpy(), axis=1)
            for i in torch.arange(0, batch_y.shape[0]):
                y_true.append(batch_y[i].detach().cpu().numpy())
                y_pred.append(prediction[i])

        mean_loss = sum(losses) / len(losses)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        pbar.set_description(f'Loss at epoch {epoch} was {mean_loss:.5f}. accuracy: {balanced_accuracy_score(y_true, y_pred)}')
        acc, f1 = return_metric(test_loader)
        if f1 > max_f1:
            max_f1 = f1
            max_acc = acc
    eval_F1.append(max_f1)
    eval_acc.append(max_acc)
pkl.dump(eval_F1, open(f'{inputs.outpth}/prottrans_F1.pkl', 'wb'))
pkl.dump(eval_acc, open(f'{inputs.outpth}/prottrans_acc.pkl', 'wb'))


