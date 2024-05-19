# Finetuning MTDP
There are two ways to finetining MTDP results:

```
Finetuning_all: Finetuing all the parameters in MTDP with a classifier

Finetuning_cls: Using the embeddings output by MTDP and train a classifier
```

In our manucsript, we use the "Finetuning_cls" mode. Thus, in the "Finetuning_cls" folder, we provide how we trained classifiers for each task. 
We also provide the scripts for ESM and ProtTrans finetuning. The model and hyperparameters used in different scripts are the same to maintain a fair comparision.

In the "Finetuning_all" folder, we provide a way if anyone want to finetuning all the parameters used in MTDP. Because MTDP is significant small in size compared to the large
protein model. It can be easily run on a GPU card with minimum 10 Gb memory.


## Data used in the manuscript
|          Dataset              |                                    Pth                                    |  
| ----------------------------- | :---------------------------------------------------------------------------: |
|	Binary classification |      [Download](https://github.com/KennthShang/MTDP/tree/main/Datasets/deeploc)    |
|	Multiclass classification | [Download](https://github.com/KennthShang/MTDP/tree/main/Datasets/deeploc)  |
|	Multilabel classification	| [Download](https://github.com/KennthShang/MTDP/tree/main/Datasets/goterm) |
|	Regression (meltome)			| [Download](https://github.com/KennthShang/MTDP/tree/main/Datasets/meltome) |
|	Regression (solubility)				| [Download](https://github.com/KennthShang/MTDP/tree/main/Datasets/solubility) |

