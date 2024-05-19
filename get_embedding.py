import torch
from tqdm import tqdm
import torch.nn as nn
from embedder import MTDP
import pickle as pkl
import numpy as np
from tqdm import tqdm, trange
import torch.utils.data as Data
from datasets import Dataset
from transformers import T5Tokenizer
from Bio import SeqIO
from ulity import *
import argparse



parser = argparse.ArgumentParser(description="""Main script of MTDP embedder.""")
parser.add_argument('--inputs', type=int, default=1, help='input path to the protein sequences.')
parser.add_argument('--db', type=int, default=6, help='path to the database.')
parser.add_argument('--outpth', type=int, default=6, help='path to the output folder.')
inputs = parser.parse_args()

FASTA_file = inputs.inputs
MTDP_path = inputs.db
outpth = inputs.outpth


# set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'running with {device}')

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained(f'{MTDP_path}/MTDP_tokenizer/', do_lower_case=False, model_max_length = 1001)

# prepare the protein sequences as a list
sequence_examples = []
acc = []
for record in SeqIO.parse(f'{FASTA_file}', 'fasta'):
    sequence_examples.append(str(record.seq)[:1000])
    acc.append(record.id)
# replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

# tokenize sequences and pad up to the longest sequence in the batch
ids = tokenizer(sequence_examples, add_special_tokens=True, padding="max_length", truncation=True, max_length=1001)
pkl.dump(ids, open(f'{outpth}/T5_idx.pkl', 'wb'))

# load the dataset
train_data = Dataset.from_dict(ids)


model = MTDP()
try:
    model.load_state_dict(
        torch.load(f'{MTDP_path}/UniProtKB/uniprotKB.bin', map_location='cpu'))
except:
    print('No model found, initialize a new model')
    exit()

model.to(device)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




with torch.no_grad():
    embed = []
    #label = []
    train_loader = return_batch(train_data, batch_size=10, flag=False)
    pbar = tqdm(train_loader)
    for step, batch_x in enumerate(pbar):
        # generate embeddings
        input_ids = np.array([item.numpy() for item in batch_x['input_ids']]).T
        input_ids = torch.Tensor(input_ids).long()
        mask = np.array([item.numpy() for item in batch_x['attention_mask']]).T
        mask = torch.Tensor(mask).long()
        embedding_repr = model(input_ids=input_ids.cuda(), attention_mask=mask.cuda())
        embed.append(embedding_repr['logits'].detach().cpu().numpy())
        pbar.set_description(f'Processing {step}')

embed = np.concatenate(embed)
embed_dict = dict(zip(acc, embed))    
pkl.dump(embed_dict, open(f'{outpth}/MTDP_embed.dict', 'wb'))

