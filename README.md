# MIA_Graph
This is a project about privacy leakage disparity on graph data.
## Data prepration
### Data Requirements
A good data for this project should satisfy following requirements:
1. Data should have reasonable target label for classification, or use some features as target label.
2. Data should have enough node number, better 1000~4000.
3. Data should have sensitive feature such as gender. This info is acceptable if it's represented by digits.
4. Data should have feature matrix.

### Data Files Format
Dataloader: a dataloader is necsssary to return following items:
1. Adjacency matrix: Torch.LongTensor
2. Feature Matrix: Torch.FloatTensor
3. Gender vector: Torch.FloatTensor
4. label: Torch.LongTensor, not non-hot.


### Current Data Status
1. Facebook \#107: Ready to go, Target is finished, all attack finished, Cora is used as shadow dataset.
2. Cora: Target is finished, attack is not finished

## Target experiment
### how to run:
In full_experiment.py, there are 5 flags: 
1. run_dense: Boolean variable, True if dense experiment is required,
2. run_target Boolean variable, True if target experiment is required,
3. run_partial: Boolean variable, True if we need to generate new partial file for attack
4. fair_sample: Boolean variable, True if the MIA training set is picked balanced in three gender groups (inner 1, inner 2, intra gender)
5. run_attack Boolean variable, True if attack experiment is required
6. prepare_new: Boolean variable, True if we need to generate a new version of attack input. If False, the code will read the prepared MIA input.

Please manually set run_target as true while the others as False.

Please set \<dataset\> and \<datapath\>. You can just pull dataset from this GitHub so the default setting will be good.

Please set \<model_type\> as "GAT" or "gcn".

### What to Tune
In GAT.py, line 63-71, "nhid", "nheads", "dropout" in GAT model initialization and "lr", "weight_decay" in optimizer are tunable parameters.


## File Introduction
### How to use scripts
### Package from other paper
1. gcn  
This package is from https://github.com/tkipf/gcn, which is the implement of paper [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)
2. pyGAT  This package is from https://github.com/Diego999/pyGAT, which is the python implement of paper [Graph Attention Networks](https://arxiv.org/abs/1710.10903)
3. stealing_link  This package is a implement of paper [Stealing Links from Graph Neural Networks](https://arxiv.org/abs/2005.02131)

### Dataset
1. Facebook ego network 107
Containing 1034 users'   

   file name | file type | item in file | Explaination 
   ------------ | ------------- | ------------- | ------------- 
   ego-adj-feat.pkl | pickle | adj | adjancy matrix, 1034\*1034, csr_matrix 
   | |   | ft | feature matrix, 1034\*576, ndarray
   107.featnames| line by line | - | 576 feature names
   
