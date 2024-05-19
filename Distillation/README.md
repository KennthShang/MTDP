# Distillation

The scripts provided in this folder can be use to distill the knowledge from any provided teachers.

## Example
There are several steps you need to follow before distillation.

1. You need to select teachers by yourself. Cuerrent, the most commonly used large models for proteins are: [ESM](https://github.com/facebookresearch/esm/tree/main) and [ProtTrans](https://github.com/agemagician/ProtTrans)
2. Prepare a FASTA file for distillation. In our manuscript, we use the UniProtKB database
3. Generating embedding using your teacher model and store them on hard device. This will help you to save GPU memory when distillation. Take ESM2-33 as an example, you need to run the extract.py script provided in [ESM](https://github.com/facebookresearch/esm/tree/main)
4. You still need to resort the embeddings according to your FASTA accessions since both ESM and ProtTrans models will shuffle the original order in the FASTA file.
5. Then you can run the following command to distill your model. Please make sure that your "*.feat" file is matrix with shape (number of sequences, dimension of embedding)

```
python -u distillation.py --teacher1 esm.feat --teacher2 prottrans.feat --inputFA protein.fa --db MTDP_tokenizer/ --outpth out/
```

We also provide a sbatch script for you reference if you are using HPC for training

## Data available
|          Dataset              |                                    Pth                                    |  
| ----------------------------- | :---------------------------------------------------------------------------: |
|	UniProtKB |      [Download](https://www.uniprot.org/help/downloads)    |
|	UniProtRef50 |      [Download](https://www.uniprot.org/help/downloads)    |
|	UniProtRef100 |      [Download](https://www.uniprot.org/help/downloads)    |
