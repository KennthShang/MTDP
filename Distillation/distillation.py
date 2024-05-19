import os
import re
import torch
import GPUtil
import numpy as np
import pickle as pkl
import torch.nn as nn
from MTDP import MTDP
from tqdm import tqdm
from tqdm import tqdm, trange
from torch import optim
import torch.nn.functional as F
from transformers import TrainingArguments, Trainer
from transformers import T5Tokenizer
from ulity import create_dataset
import argparse

parser = argparse.ArgumentParser(description="""Main script of MTDP embedder.""")
parser.add_argument('--teacher1', type=int, default=1, help='input pth to the teacher1 embedding.')
parser.add_argument('--teacher2', type=int, default=1, help='input pth to the teacher2 embedding.')
parser.add_argument('--inputFA', type=int, default=1, help='input path to the protein sequences.')
parser.add_argument('--db', type=int, default=6, help='path to the database.')
parser.add_argument('--outpth', type=int, default=6, help='path to the output folder.')
inputs = parser.parse_args()


teacher1 = pkl.load(open(f'{inputs.teacher1}.pkl', 'rb'))
teacher2 = pkl.load(open(f'{inputs.teacher2}.pkl', 'rb'))
train_set = create_dataset(inputs.db, inputs.inputFA, inputs.outpth, list(teacher1), list(teacher2))



# set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MTDP()
# if multi-card is available, use it
#if torch.cuda.device_count() > 1:
#    print(f'Use {torch.cuda.device_count()} GPUs!\n')
#    model = nn.DataParallel(model)
model.to(device)

# define the loss function and optimizer
# loss_func = nn.MSELoss()
# kl_loss = nn.KLDivLoss(reduction="batchmean")
# optimizer = optim.AdamW(model.parameters(), lr=1e-5)


model.load_state_dict(torch.load('model_with_uniprotKB/checkpoint-454920/pytorch_model.bin', map_location='cpu'))

# load pre-trained model if nessary
#try:
#    model.load_state_dict(
#        torch.load('test/checkpoint-57080/pytorch_model.bin', map_location='cpu'))
#except:
#    print('No model found, initialize a new model')
    #exit()


args = TrainingArguments(
        f"./{inputs.outpth}",
        #evaluation_strategy = "steps",
        #eval_steps = 528,
        logging_strategy = "steps",
        remove_unused_columns=False,
        save_strategy = "epoch",
        learning_rate=3e-4,
        #weight_decay=1e-4,
        warmup_ratio=0.1,
        per_device_train_batch_size=16,
        #per_device_eval_batch_size=val_batch,
        #per_device_eval_batch_size=batch,
        gradient_accumulation_steps=2,
        num_train_epochs=150,
        #deepspeed= 'ds_config.json',
        fp16=False,
    )


def compute_metrics(eval_pred):
    #metric = load("spearmanr")
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    predictions, labels = eval_pred
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    with torch.no_grad():
        output = F.log_softmax(out, dim=1)
        loss   = kl_loss(output, F.softmax(labels, dim=1))

    return loss



trainer = Trainer(
        model,
        args,
        train_dataset=train_set,
    )
trainer.train()
