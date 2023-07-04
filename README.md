# Disparate Vulnerability in Link Inference Attacks against Graph Neural Networks
This is the repo for Paper "Disparate Vulnerability in Link Inference Attacks against Graph Neural Networks'.
In the paper, we introduce a new defense mechanism called FairDefense to address disparities in subgroup vulnerability (DSV) while providing protection against Link-level Membership Inference Attacks (LMIA). The repository's code includes implementations of the Graph Attention Network (GAT) and Graph Convolutional Network (GCN) models integrated with FairDefense.

## Development Environment
This repo uses Python 3.7. The packages used in this repo are listed in `requirements.txt`. To install the packages, run `pip3 install -r requirements.txt`.

The following system requirements must be met in order to run the code:
- Linux operating system or windows operating system
- x86_64 architecture

## Repo Instruction
### full_experient_v2.py
This file is used to run target model experiment and attack model experiment. In particular, the parameters are listed as:

| Parameter   | valid value      | Instruction                                                                     |
|-------------|------------------|---------------------------------------------------------------------------------|
| model_type  | GAT\|GCN         | The type of target model.                                                       |
| dataset     | pokec\| facebook | The name of dataset for target model training                                   |
| datapath    | dataset/         | The path to the dataset.                                                        |
| epoch       | [integer]        | Number of epoch for target model training                                       |
| FD          | N/A              | `--FD` will triger target model with FairDefense                                |
| gamma       | [float]          | The parameter for $\gamma$                                                      |
| run_dense   | N/A              | `--run_dense` will triger Dense model training, which is needed before Attack B |
| run_Target  | N/A              | `--run_Target` will triger target model training                                |
| run_partial | N/A              | `--run_partial` will triger MIA training/testing preparation                    |
| run_Attack  | N/A              | `--run_attack` will triger attack model training and testing                    |
| use_cpu     | N/A              | `--use_cpu` will triger the code to run on cpu                                  |
Running `full_experient_v2.py` will launch experiment with completed or selected steps.

### Use case for experiment
#### CPU usage
If gpu is not available, please add `--use_cpu` to the command line. so that the code will run on cpu.

#### Use case 1: run completed experiment without defense:
In this case, experiment with no defense mechanism will be finished.\
`python3 full_experiment_v2.py --model_type GCN --dataset facebook --epoch 100 --run_dense --run_Target --run_partial --run_attack`

#### Use case 2: run attack experiment when target model experiment has been finished:
In this case, target model experiment will be skipped.\
`python3 full_experiment_v2.py --model_type GCN --dataset facebook --run_partial --run_attack`

#### Use case 3: run experiment with FairDefense:
In this case, target model experiment will run with FairDefense. $\gamma$ is set as 0.1\
`python3 full_experiment_v2.py --model_type GCN --dataset facebook --epoch 100 --run_Target --run_partial --run_attack --FD --gamma 0.1`

### Expected output
#### Overall Attack Performance
The output of `full_experiment_v2.py` contains a csv file that includes the overall performance of attack model. The result file will be saved as `[model]/[dataset]/[gamma=?]/attack_res.csv`.

#### Subgroup Attack Performance and DSV
The output of `full_experiment_v2.py` contains a csv file that includes the subgroup performance of attack model and the averaged DSV. The result file will be saved as `[model]/[dataset]/[gamma=?]/attack_subgroups.csv`.

#### Performance of FairDefense
With FairDefense applied, the Attack performance should be smaller than the result withouut FairDefense.

With FairDefense applied, the DSV should be smaller than the result withouut FairDefense. This information will be print out when the experiment is run with FairDefense:

```DSV of GCN is changed from 0.064595367625992 to 0.03606620882052991```

## Contributions and Reference
### Contribution
The scripts in this repo is implemented based on previousworks. The major contributions of this repo are:
1. Added features to GAT/GCN with defense mechanisms
2. Added functions that measures the performance of subgroups.
### Packages Reference

#### GCN model:
GCN model locates in `pyGCN` directory is from [this repo](https://github.com/tkipf/pygcn), which is the official implement of [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)
#### GAT model:
GAT model locates in `pyGAT` directory is from [this repo](https://github.com/Diego999/pyGAT), which is the official implement of paper [Graph Attention Networks](https://arxiv.org/abs/1710.10903)
#### GAT model:
The code for attack in `stealing_link` directory is from [this repo](https://github.com/xinleihe/link_stealing_attack), which is the official implement of paper [Stealing Links from Graph Neural Networks](https://arxiv.org/abs/2005.02131)

