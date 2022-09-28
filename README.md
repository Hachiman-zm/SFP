# SFP
Implementation of "A Compressive Prior Guided Mask Predictive Coding Approach for Video Analysis" in ACCV2022

The code will be updated as soon as possible! 

Sorry for the possible delay because of the coming doctoral thesis proposal presentation and thesis defense.😭😭 

## Environments
You will have to choose cudatoolkit version to match your compute environment. 
The code is tested on PyTorch 1.11.0 but other versions might also work. 

The version of python library is shown in sfp.yaml and requirements.txt. Many libraries may be unneccessary beacause SFP is tested on various VOS models.

## Demo
```Shell
sh demo.sh
```
## Train
```Shell
sh train.sh
```
## Evaluate
```Shell
sh evaluate.sh


## Acknowledgement
The overall code framework is adapted from [RAFT](https://github.com/princeton-vl/RAFT). We
thank the authors for the contribution. We also thank [Phil Wang](https://github.com/lucidrains)
for open-sourcing transformer implementations. 
