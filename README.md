


# Multimodal Emotion Recognition in Conversations via Graph Structure Learning

> The official implementation for paper: Multimodal Emotion Recognition in Conversations via Graph Structure Learning, ICME 2025 (Oral).


<img src="https://img.shields.io/badge/Venue-ICME--25-blue" alt="venue"/> <img src="https://img.shields.io/badge/Status-Accepted-success" alt="status"/> <img src="https://img.shields.io/badge/Issues-Welcome-red">


## Requirements
```
Python 3.10.13
PyTorch 1.13.0
torch_geometric 2.5.2
torch-cluster 1.6.1
torch-scatter 2.1.1
torch-sparse 0.6.17
torch-spline-conv 1.2.2
sparsemax 0.1.9
CUDA 11.7
```




## Preparation
1. Download  [**multimodal-features**](https://www.dropbox.com/scl/fo/veblbniqjrp3iv3fs3z6p/AEzkNgWqPHHzldBZ0zEzr2Y?rlkey=yhlr653c0vnvaf1krpdkla36u&e=1&dl=0) 
2. Save data/iemocap/iemocap_features_roberta.pkl, data/iemocap/IEMOCAP_features.pkl in `data/`; Save meld_features_roberta.pkl, data/meld/MELD_features_raw1.pkl in `data/`. 


## Training
### Training on IEMOCAP
1. Train RIGSL using the IEMOCAP dataset.
```shell
python train.py --Dataset=IEMOCAP --av_using_lstm=False --backbone=M3Net --base_model=GRU --batch_size=8 --class_weight=True --denoise_dropout=0.4 --dropout=0.3 --epochs=80 --gamma=-0.95 --gumbel_k=64 --hidden_dim=512 --l2=3e-05 --link_loss_coff=0.1 --link_loss_epoch=15 --lr=0.0001 --modals=avl --nb_features_dim=32 --nodeformer_dropout=0.4 --nodeformer_heads=6 --norm=None --num_K=2 --num_L=4 --penalty_weight_coff=0.0005 --seed=1475 --tau=0.2 --temperature=1.5 --testing=False --use_gumbel=True --use_jk_nodeformer=False --use_residue=False --use_residue_denoise=True --use_residue_nodeformer=True --use_speaker=True --use_wandb=False --windowf=10 --windowp=10 --zeta=1.05
```

### Training on MELD
2. Train RIGSL using the MELD dataset.
```shell
python train.py --Dataset=MELD --av_using_lstm=False --backbone=M3Net --base_model=GRU --batch_size=16 --class_weight=True --denoise_dropout=0 --dropout=0.4 --epochs=40 --gamma=-0.95 --gumbel_k=4 --hidden_dim=512 --l2=3e-05 --link_loss_coff=0.1 --link_loss_epoch=10 --lr=0.0001 --modals=avl --nb_features_dim=64 --nodeformer_dropout=0.3 --nodeformer_heads=8 --norm=None --num_K=3 --num_L=2 --penalty_weight_coff=0.0005 --seed=44751 --tau=0.4 --temperature=1 --testing=False --use_gumbel=True --use_jk_nodeformer=False --use_residue=True --use_residue_denoise=True --use_residue_nodeformer=True --use_speaker=True --use_wandb=False --windowf=10 --windowp=10 --zeta=1.05
```





### Evaluation

Downloading Checkpoint  [**IEMOCAP**](https://drive.google.com/drive/folders/1bEOC5jnIYE1lmpWur4UvB5MEmDF_56Jl?usp=share_link) 

```python
# Evaluation for IEMOCAP
python test.py --Dataset=IEMOCAP --av_using_lstm=False --backbone=M3Net --base_model=GRU --batch_size=8 --class_weight=True --denoise_dropout=0.4 --dropout=0.3 --gamma=-0.95 --gumbel_k=64 --hidden_dim=512 --l2=3e-05 --link_loss_coff=0.1 --link_loss_epoch=15 --lr=0.0001 --modals=avl --nb_features_dim=32 --nodeformer_dropout=0.4 --nodeformer_heads=6 --norm=None --num_K=2 --num_L=4 --penalty_weight_coff=0.0005 --tau=0.2 --temperature=1.5 --testing=True --use_gumbel=True --use_jk_nodeformer=False --use_residue=False --use_residue_denoise=True --use_residue_nodeformer=True --use_speaker=True --use_wandb=False --windowf=10 --windowp=10 --zeta=1.05 --model_path=checkpoints/IEMOCAP.pkl
```








## Citation
If you find our work useful for your research, please kindly cite our paper as follows:
```
@inproceedings{xiong2025graph,
title = {Multimodal Emotion Recognition in Conversations via Graph Structure Learning},
author = {Feng Xiong, Geng Tu, Yice Zhang, Jun Wang, Shiwei Chen, Bin Liang, Yue Yu, Min Yang, Ruifeng Xu},
booktitle={IEEE International Conference on Multimedia and Expo (ICME)},
year = {2025}
}
```

## Acknowledgements
Special thanks to the following authors for their contributions through open-source implementations.
* [NodeFormer: A Graph Transformer with Linear Complexity](https://github.com/qitianwu/NodeFormer)
* [Multivariate, Multi-frequency and Multimodal: Rethinking Graph Neural Networks for Emotion Recognition in Conversation](https://github.com/feiyuchen7/M3NET)




