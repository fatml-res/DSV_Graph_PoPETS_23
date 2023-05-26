# Disparate Vulnerability in Link Inference Attacks against Graph Neural Networks
This is the repo for Paper "Disparate Vulnerability in Link Inference Attacks against Graph Neural Networks'

## File dictionary
### full_experient_v2.py
This file is used to run target model experiment and attack model experiment. In particular, the parameters are listed as:

| Parameter   | valid value      | Instruction                                                                     |
|-------------|------------------|---------------------------------------------------------------------------------|
| model_type  | GAT\|GCN         | The type of target model.                                                       |
| dataset     | pokec\| facebook | The name of dataset for target model training                                   |
| datapath    | dataset/         | The path to the dataset.                                                        |
| epoch       | [integer]        | Number of epoch for target model training                                       |
| Min         | N/A              | `--Min` will triger target model with FairDefense                               |
| gamma       | [float]          | The parameter for $\gamma$                                                      |
| run_dense   | N/A              | `--run_dense` will triger Dense model training, which is needed before Attack B |
| run_Target  | N/A              | `--run_Target` will triger target model training                                |
| run_partial | N/A              | `--run_partial` will triger MIA training/testing preparation                    |
|  run_Attack | N/A              | `--run_attack` will triger attack model training and testing                    |

Running `full_experient_v2.py` will launch experiment with completed or selected steps.

###


### How to use scripts
### Package from other paper
1. gcn  
This package is from https://github.com/tkipf/gcn, which is the implement of paper [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)
2. pyGAT  This package is from https://github.com/Diego999/pyGAT, which is the python implement of paper [Graph Attention Networks](https://arxiv.org/abs/1710.10903)
3. stealing_link  This package is a implement of paper [Stealing Links from Graph Neural Networks](https://arxiv.org/abs/2005.02131)

