## Fine-Grained Predicates Learning for Scene Graph Generation(CVPR 2022)

![LICENSE](https://img.shields.io/badge/license-MIT-green)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.2.0-%237732a8)

This repository contains code for the paper "Fine-Grained Predicates Learning for Scene Graph Generation(CVPR 2022)". This code is based on [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch). 

## Abstract
The performance of current Scene Graph Generation models is severely hampered by some hard-to-distinguish predicates, eg, ''woman-on/standing on/walking on-beach'' or ''woman-near/looking at/in front of-child''. While general SGG models are prone to predict head predicates and existing re-balancing strategies prefer tail categories, none of them can appropriately handle these hard-to-distinguish predicates. To tackle this issue, inspired by fine-grained image classification, which focuses on differentiating among hard-to-distinguish object classes, we propose a method named Fine-Grained Predicates Learning (FGPL) which aims at differentiating among hard-to-distinguish predicates for Scene Graph Generation task. Specifically, we first introduce a Predicate Lattice that helps SGG models to figure out fine-grained predicate pairs. Then, utilizing the Predicate Lattice, we propose a Category Discriminating Loss and an Entity Discriminating Loss, which both contribute to distinguishing fine-grained predicates while maintaining learned discriminatory power over recognizable ones. The proposed model-agnostic strategy significantly boosts the performances of three benchmark models (Transformer, VCTree, and Motif) by 22.8\%, 24.1\% and 21.7\% of Mean Recall (mR@100) on the Predicate Classification sub-task, respectively. Our model also outperforms state-of-the-art methods by a large margin (i.e., 6.1\%, 4.6\%, and 3.2\% of Mean Recall (mR@100)) on the Visual Genome dataset.
<div align=center><img height="400" width="600" src=abstract.png/></div>

## Framework
<div align=center><img src=framework.png/></div>

## Visualization
<div align=center><img  height="600" width="800" src=visual_sp-1.png/></div>

## Device
All our experiments are trained using one NVIDIA GeForce RTX 3090, if you wanna run it on your own device, make sure to follow distributed training instructions in [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch). 

## Dataset
Follow [DATASET.md](DATASET.md) for instructions of dataset preprocessing.

## Train
Follow the [instructions](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) to install and use the code. 
### Scene Graph Generation Model
We provide scripts for training models with FGPL our model( in `scripts/885train_[motif/trans/vctree].sh`), and 
key commands for training script should be set up as follows：\

    python ./tools/relation_train_net.py \

    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer_FGPL.yaml" ("configs/e2e_relation_X_101_32_8_FPN_1x_motif_FGPL.yaml", ("configs/e2e_relation_X_101_32_8_FPN_1x_vctree_FGPL.yaml")) \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True\
    
    MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor/VCTreePredictor/MotifPredictor \
    
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
    
    SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH $NUM_GUP \
    
    SOLVER.MAX_ITER 16000 SOLVER.BASE_LR 1e-3 \
    
    SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
    
    SOLVER.STEPS "(10000, 16000)" SOLVER.VAL_PERIOD 2000 \
    
    SOLVER.CHECKPOINT_PERIOD 4000 GLOVE_DIR ./datasets/vg/ \
    
    MODEL.PRETRAINED_DETECTOR_CKPT path_to_pretrained_fastercnn/model_final.pth \
    . 
    . 
    .
    (This is for FGPL) MODEL.ROI_RELATION_HEAD.USE_EXTRA_LOSS True \
    (This is for CDL) MODEL.ROI_RELATION_HEAD.USE_LOGITS_REWEIGHT  True \  
    (This are paramter for CDL) MODEL.ROI_RELATION_HEAD.MITIGATION_FACTOR_HYPER  1.5 \
                      MODEL.ROI_RELATION_HEAD.COMPENSATION_FACTOR_HYPRT  2.0 \
    (This is for EDL) MODEL.ROI_RELATION_HEAD.USE_CONTRA_LOSS  True \ 
    (This is for EDL) MODEL.ROI_RELATION_HEAD.USE_CONTRA_BCE  True  \   
    (This are parameter for EDL) MODEL.ROI_RELATION_HEAD.CONTRA_DISTANCE_LOSS_VALUE  0.6 \
                      MODEL.ROI_RELATION_HEAD.CONTRA_DISTANCE_LOSS_COF  0.1 \
                      MODEL.ROI_RELATION_HEAD.CANDIDATE_NUMBER  5 \    
    OUTPUT_DIR ./checkpoints/${MODEL_NAME};
    


### Model
The trained models(Transformer-FGPL, Motif-FGPL, VCTree-FPGL) can be download from [BaiDuYun](https://pan.baidu.com/s/1vbsFDIHI57o9HxIv5BJiZA) (Password: 5u2o)\

We realsed the weights for the pretained VCTree model on the Visual Genome dataset trained using both cross-entropy based and energy-based training.

| Predcls                | SGCls                 | SGDet                 |
|--------------------|--------------------|
| [VCTree-Predcls](https://tinyurl.com/vctree-ebm-predcls) | [VCTree-SGCls](https://tinyurl.com/yxpt4n7w) || [VCTree-SGDets](https://tinyurl.com/yxpt4n7w) |
| [Transformer-SGCLS](https://tinyurl.com/vctree-ebm-sgcls)   | [Transformer-SGCLS](https://tinyurl.com/vctree-ce-sgcls)   | [Transformer-SGCls](https://tinyurl.com/yxpt4n7w) |
| [Motif-SGDET](https://tinyurl.com/vctree-ebm-sgdet)   | [Motif-SGDET](https://tinyurl.com/vctree-ce-sgdet)   | [Motif-SGCls](https://tinyurl.com/yxpt4n7w) |

## Test
We provide `test.sh` for directly reproduce the results in our paper. Remember to set `MODEL.WEIGHT` as checkopints we provided and choose the corresponding dataset split in `DATASETS.TEST`.

## Help
Be free to contact me (xinyulyu68@gmail.com) if you have any questions!

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
