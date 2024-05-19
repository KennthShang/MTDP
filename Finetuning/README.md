# Finetuning MTDP
Usually there are two ways to finetining MTDP results:

```
Finetuning_all: Finetuing all the parameters in MTDP with a classifier

Finetuning_cls: Using the output of the embedding by MTDP and train a classifier
```

In our manuscript, we use the "Finetuning_cls" mode. However, Because MTDP is significantly smaller in size than the large
protein model, sometimes you might want to finetune all the parameters used in MTDP. Currently, it can be easily run on a GPU card with a minimum of 10 Gb memory.


## Data used in the manuscript
|          Dataset              |                                    Pth                                    |  
| ----------------------------- | :---------------------------------------------------------------------------: |
|	Binary classification |      [Download](https://github.com/KennthShang/MTDP/tree/main/Datasets/deeploc)    |
|	Multiclass classification | [Download](https://github.com/KennthShang/MTDP/tree/main/Datasets/deeploc)  |
|	Multilabel classification	| [Download](https://github.com/KennthShang/MTDP/tree/main/Datasets/goterm) |
|	Regression (meltome)			| [Download](https://github.com/KennthShang/MTDP/tree/main/Datasets/meltome) |
|	Regression (solubility)				| [Download](https://github.com/KennthShang/MTDP/tree/main/Datasets/solubility) |

