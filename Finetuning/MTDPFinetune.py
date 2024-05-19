from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
from tqdm import tqdm
import torch.nn as nn
from embedder import MTDP
import pickle as pkl
import numpy as np
from tqdm import tqdm, trange
import torch.utils.data as Data
from torch import optim
from sklearn.metrics import confusion_matrix
from cls import Classifier
from ulity import return_batch, return_ids
import argparse
import os

parser = argparse.ArgumentParser(description="""Main script of MTDP embedder.""")
parser.add_argument('--train_FA', type=int, default=1, help='input path to the training protein sequences.')
parser.add_argument('--test_FA', type=int, default=1, help='input path to the testing protein sequences.')
parser.add_argument('--train_label', type=int, default=1, help='input path to the training label.')
parser.add_argument('--test_label', type=int, default=1, help='input path to the testing label.')
parser.add_argument('--db', type=int, default=6, help='path to the database.')
parser.add_argument('--outpth', type=int, default=6, help='path to the output folder.')
inputs = parser.parse_args()


if not os.path.isdir(inputs.outpth):
    os.makedirs(inputs.outpth)


train_FA = inputs.train_FA
train_label = inputs.train_label
test_FA = inputs.test_FA
test_label = inputs.test_label
MTDP_path = inputs.db
outpth = inputs.outpth
    

train_label = pkl.load(open(f'{train_label}', 'rb'))
train_data  = return_ids(MTDP_path, train_FA, outpth, train_label)
test_label  = pkl.load(open(f'{test_label}', 'rb'))
test_data   = return_ids(MTDP_path, test_FA, outpth, test_label)



# set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MTDP()
model.to(device)


# load model
try:
    model.load_state_dict(
        torch.load('test/checkpoint-245300/pytorch_model.bin', map_location='cpu'))
except:
    print('No model found, initialize a new model')


clf = Classifier(
    embed_size=1280,
    out_dim=2
)


# initialize the model with the LoRA framework
#lora_config = LoraConfig()
# T5Encoder = get_peft_model(T5Encoder, lora_config)
model.to(device)
clf.to(device)

# define the loss function and optimizer
loss_func = nn.CrossEntropyLoss()
params = list(model.parameters()) + list(clf.parameters())
optimizer = optim.AdamW(params, lr=1e-4)


test_loader = return_batch(test_data, batch_size=16, flag=False)

def return_metric():
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for step, batch_x in enumerate(test_loader):
            input_ids = np.array([item.numpy() for item in batch_x['input_ids']]).T
            input_ids = torch.Tensor(input_ids).long()
            mask = np.array([item.numpy() for item in batch_x['attention_mask']]).T
            mask = torch.Tensor(mask).long()
            embedding_repr = model(input_ids=input_ids.to(device), attention_mask=mask.to(device))
            logits = clf(embedding_repr['logits'])
            _, predictions = logits.max(dim=1)
            batch_y = batch_x['label']

            #print(predictions)
        
            for i in torch.arange(0, batch_y.shape[0]):
                y_true.append(batch_y[i].cpu().numpy())
                y_pred.append(predictions[i].cpu().numpy())
    
    print(classification_report(y_true, y_pred, target_names=['membrane', 'other']))
    print(f'precision_score: {precision_score(y_true, y_pred)}')
    print(f'recall_score: {recall_score(y_true, y_pred)}')
    print(f'f1_score: {f1_score(y_true, y_pred)}')
    



for epoch in range(10):
    y_true = []
    y_pred = []
    train_loader = return_batch(train_data, batch_size=16, flag=False)
    pbar = tqdm(train_loader)
    for step, batch_x in enumerate(pbar):
        # generate embeddings
        input_ids = np.array([item.numpy() for item in batch_x['input_ids']]).T
        input_ids = torch.Tensor(input_ids).long()
        mask = np.array([item.numpy() for item in batch_x['attention_mask']]).T
        mask = torch.Tensor(mask).long()
        embedding_repr = model(input_ids=input_ids.to(device), attention_mask=mask.to(device))
        logits = clf(embedding_repr['logits'])
        _, predictions = logits.max(dim=1)
        batch_y = batch_x['label']
        loss = loss_func(logits, batch_y.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for i in torch.arange(0, batch_y.shape[0]):
            y_true.append(batch_y[i].cpu().numpy())
            y_pred.append(predictions[i].cpu().numpy())
        
        pbar.set_description(f'Processing {step} Epoch:{epoch} | Step:{step} | loss:{loss.item()}')
    
    print(f'Results at epoch {epoch} was Precision: {precision_score(y_true, y_pred, average="binary")}, Recall: {recall_score(y_true, y_pred, average="binary")}, F1: {f1_score(y_true, y_pred, average="binary")}')
    return_metric()
