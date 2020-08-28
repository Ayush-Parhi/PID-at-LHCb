# Particle Identification at LHCb
Particle identification has always been a crucial machine learning task in any of the CERN's (European Council for Nuclear Research) experiments (ATLAS, CMS, ALICE, LHCb, LHCf, TOTEM, MoEDAL and DUNE). Before the rise of deep learning algorithms, this task was done using the boosted machine learning algorithms, like boosted Decision trees.  
In this project, I have trained two deep learning models, MLP and 1D-CNN, to successfully classify 6 different types of particles from the Monte Carlo's simulated dataset of LHCb.  
The metric used for evaluating the perfoemance of the models is ROC-AUC or Receiver operating characteristic- area under curve. These models achieved great results from this dataset, but still new contributions are welcome!  

# Brief description of the codebase
- The `networks` directory contains the network architecture of the two models mentioned earlier.
- `models` directory compile those networks to get ready for training.
- The preprocessing of the data required for the training is present in the `preprocessing` directory.
- Finally the `training` folder contains scripts to train both the models.
- To make everythin simple, I have added a `scripts` folder that run shell scripts commands for training and other tasks.
- The `tests` directory contains the scripts to test the individual predictors present in the root directory.
- The `notebooks` directory contains the notebooks used to write the experiments of the software.  

# Installation and Usage  
The `scripts` makes it easy for users to run the training and testing commands directly from the command line.  
For successful installation and training the models, clone the repository, move into the root dirctory of the project, and run the following commands in your terminal or command prompt:
``` 
1. $ scripts/setup.sh 
```
``` 
2. $ scripts/train_mlp.sh
```
``` 
3. $ scripts/test_mlp.sh
```
``` 
4. $ scripts/train_cnn.sh
```
``` 
5. $ scripts/test_cnn.sh
```
