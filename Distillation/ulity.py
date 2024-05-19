import torch.utils.data as Data
from transformers import T5Tokenizer
from Bio import SeqIO
import pickle as pkl
from datasets import Dataset
import re


def create_dataset(MTDP_path, FASTA_file, outpth, teacher1, teacher2):
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
    train_data = train_data.add_column("t1", teacher1)
    train_data = train_data.add_column("t2", teacher2)
    return train_data