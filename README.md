## Fine-Grained Predicates Learning for Scene Graph Generation(CVPR 2022)

![LICENSE](https://img.shields.io/badge/license-MIT-green)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.2.0-%237732a8)

This repository contains code for the paper "Fine-Grained Predicates Learning for Scene Graph Generation(CVPR 2022)". This code is based on [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch). 

## Abstract
The performance of current Scene Graph Generation models is severely hampered by some hard-to-distinguish predicates, eg, ''woman-on/standing on/walking on-beach'' or ''woman-near/looking at/in front of-child''. While general SGG models are prone to predict head predicates and existing re-balancing strategies prefer tail categories, none of them can appropriately handle these hard-to-distinguish predicates. To tackle this issue, inspired by fine-grained image classification, which focuses on differentiating among hard-to-distinguish object classes, we propose a method named Fine-Grained Predicates Learning (FGPL) which aims at differentiating among hard-to-distinguish predicates for Scene Graph Generation task. Specifically, we first introduce a Predicate Lattice that helps SGG models to figure out fine-grained predicate pairs. Then, utilizing the Predicate Lattice, we propose a Category Discriminating Loss and an Entity Discriminating Loss, which both contribute to distinguishing fine-grained predicates while maintaining learned discriminatory power over recognizable ones. The proposed model-agnostic strategy significantly boosts the performances of three benchmark models (Transformer, VCTree, and Motif) by 22.8\%, 24.1\% and 21.7\% of Mean Recall (mR@100) on the Predicate Classification sub-task, respectively. Our model also outperforms state-of-the-art methods by a large margin (i.e., 6.1\%, 4.6\%, and 3.2\% of Mean Recall (mR@100)) on the Visual Genome dataset.
<div align=center><img height="400" width="600" src=abstract.png/></div>

## Framework
Within our Fine-Grained Predicates Learning (FGPL) framework, shown below, we first construct a Predicate Lattice concerning context information to understand ubiquitous correlations among predicates. Then, utilizing the Predicate Lattice, we develop a Category Discriminating Loss and an Entity Discriminating Loss which help SGG models differentiate hard-to-distinguish predicates.
<div align=center><img src=framework.png/></div>

## Visualization
<div align=center><img  height="600" width="800" src=visual_sp-1.png/></div>

## Device
All our experiments are trained using one NVIDIA GeForce RTX 3090, if you wanna run it on your own device, make sure to follow distributed training instructions in [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch). 

## Dataset
Follow [DATASET.md](DATASET.md) for instructions of dataset preprocessing.

## Train
Follow the [instructions](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) to install and use the code. Also, we provide scripts for training models with FGPL our model (in `scripts/885train_[motif/trans/vctree].sh`(https://github.com/XinyuLyu/FGPL/tree/master/scripts)), and 
key commands for training script should be set up as follows：\

    python ./tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer_FGPL.yaml" ("configs/e2e_relation_X_101_32_8_FPN_1x_motif_FGPL.yaml", ("configs/e2e_relation_X_101_32_8_FPN_1x_vctree_FGPL.yaml")) \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True\
    MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor/VCTreePredictor/MotifPredictor \
    . 
    . 
    .
    (This is for FGPL) MODEL.ROI_RELATION_HEAD.USE_EXTRA_LOSS True \
    (This is for CDL) MODEL.ROI_RELATION_HEAD.USE_LOGITS_REWEIGHT  True \  
    (These are paramters for CDL) MODEL.ROI_RELATION_HEAD.MITIGATION_FACTOR_HYPER  1.5 \
                      MODEL.ROI_RELATION_HEAD.COMPENSATION_FACTOR_HYPRT  2.0 \
    (This is for EDL) MODEL.ROI_RELATION_HEAD.USE_CONTRA_LOSS  True \ 
    (This is for EDL) MODEL.ROI_RELATION_HEAD.USE_CONTRA_BCE  True  \   
    (These are parameters for EDL) MODEL.ROI_RELATION_HEAD.CONTRA_DISTANCE_LOSS_VALUE  0.6 \
                      MODEL.ROI_RELATION_HEAD.CONTRA_DISTANCE_LOSS_COF  0.1 \
                      MODEL.ROI_RELATION_HEAD.CANDIDATE_NUMBER  5 \    
    OUTPUT_DIR ./checkpoints/${MODEL_NAME};
    
## Test
The trained models(Transformer-FGPL, Motif-FGPL, VCTree-FPGL) on Predcls\SGCLs\SGDet are released as below. We provide `test.sh` for directly reproduce the results in our paper. Remember to set `MODEL.WEIGHT` as checkopints we provided and choose the corresponding dataset split in `DATASETS.TEST`.


| Predcls                | SGCLs                 | SGDet                 |
|--------------------|--------------------|--------------------|
| [Motif-FGPL-Predcls](https://stduestceducn-my.sharepoint.com/:f:/g/personal/202011081621_std_uestc_edu_cn/EvKzNYpi0lRBrw9GpK8GGBMB3s8vMQ0t0N1KqKlvfQPseg?e=eFNrY6) | [Motif-FGPL-SGCLS](https://stduestceducn-my.sharepoint.com/:f:/g/personal/202011081621_std_uestc_edu_cn/EgYUqzzmHGRPtcTcGh7hqOQBupOrijaCcb00jFtLiDAAfg?e=q4Mgrb) | [Motif-FGPL-SGDet](https://stduestceducn-my.sharepoint.com/:f:/g/personal/202011081621_std_uestc_edu_cn/EhAXOXoRZaJBtXO_IALmxJ0BmppYCxOhhT7CCNQhWKYeiw?e=wZ5EIP) |
| [Transformer-FGPL-Predcls](https://stduestceducn-my.sharepoint.com/:f:/g/personal/202011081621_std_uestc_edu_cn/EuJnkW0h8DtOqo7SlGYYM-kB_wVNTX4AlR60iYGGnrg1sw?e=yPIHgX)   | [Transformer-FGPL-SGCLS](https://stduestceducn-my.sharepoint.com/:f:/g/personal/202011081621_std_uestc_edu_cn/EsgsaHtiL8ROgSYGaeQxt7EBjqH5p0PBc1kNbSL9oSXkPA?e=ReKVcU)   | [Transformer-FGPL-SGDet](https://stduestceducn-my.sharepoint.com/:f:/g/personal/202011081621_std_uestc_edu_cn/Ei88EjODSRlIqivvpv_ptHwBVRQIcZ-gKIuXJH0FUr_S-w?e=kwKowx) |
| [VCTree-FGPL-Predcls](https://stduestceducn-my.sharepoint.com/:f:/g/personal/202011081621_std_uestc_edu_cn/Eok6ZaK5qXpNoSlbVFUUN1wBg0M9gCFIBOvegjImkIo8gA?e=4e0mou)   | [VCTree-FPGL-SGCLS](https://stduestceducn-my.sharepoint.com/:f:/g/personal/202011081621_std_uestc_edu_cn/Enp8HTh7eV9Njtpehl0ARF4BNYRBpUXaDk2Fk3XoSLwkFw?e=okOTdu)   | [VCTree-FPGL-SGDet](https://stduestceducn-my.sharepoint.com/:f:/g/personal/202011081621_std_uestc_edu_cn/EhAXOXoRZaJBtXO_IALmxJ0BmppYCxOhhT7CCNQhWKYeiw?e=56vOzk) |

## Help
Be free to contact me (xinyulyu68@gmail.com) if you have any questions!

## Acknowledgement
The code is implemented based on [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch), and [SGG-G2S](https://github.com/ZhuGeKongKong/SGG-G2S). Thanks for their great works! 

## Bibtex

```
@inproceedings{sgg:FPGL,
  author    = {Xinyu Lyu and
               Lianli Gao and
               Yuyu Guo and
               Zhou Zhao and
               Hao Huang and
               Heng Tao Shen and
               Jingkuan Song},
  title     = {Fine-Grained Predicates Learning for Scene Graph Generation},
  booktitle = {CVPR},
  year      = {2022}
}
```
