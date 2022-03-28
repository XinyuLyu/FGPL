## Fine-Grained Predicates Learning for Scene Graph Generation(CVPR 2022)

![LICENSE](https://img.shields.io/badge/license-MIT-green)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.2.0-%237732a8)

This repository contains code for the paper "Fine-Grained Predicates Learning for Scene Graph Generation(CVPR 2022)". This code is based on [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch). 
<div align=center><img width="672" height="440" src=demo/framework_generalSGG.png/></div>

## Abstract
The performance of current Scene Graph Generation models is severely hampered by some hard-to-distinguish predicates, e.g., ``woman-on/standing on/walking on-beach'' or ``woman-near/looking at/in front of-child''. While general SGG models are prone to predict head predicates and existing re-balancing strategies prefer tail categories, none of them can appropriately handle these hard-to-distinguish predicates. To tackle this issue, inspired by fine-grained image classification, which focuses on differentiating among hard-to-distinguish object classes, we propose a method named Fine-Grained Predicates Learning (FGPL) which aims at differentiating among hard-to-distinguish predicates for Scene Graph Generation task. Specifically, we first introduce a Predicate Lattice that helps SGG models to figure out fine-grained predicate pairs. Then, utilizing the Predicate Lattice, we propose a Category Discriminating Loss and an Entity Discriminating Loss, which both contribute to distinguishing fine-grained predicates while maintaining learned discriminatory power over recognizable ones. The proposed model-agnostic strategy significantly boosts the performances of three benchmark models (Transformer, VCTree, and Motif) by 22.8\%, 24.1\% and 21.7\% of Mean Recall (mR@100) on the Predicate Classification sub-task, respectively. Our model also outperforms state-of-the-art methods by a large margin (i.e., 6.1\%, 4.6\%, and 3.2\% of Mean Recall (mR@100)) on the Visual Genome dataset.

## Framework
<div align=center><img width="672" height="508" src=demo/framework_v6_1.pdf/></div>


## Visualization
<div align=center><img width="994" height="774" src=/visual_1.pdf/></div>

## Train
### Scene Graph Generation Model
We provide scripts for training models with FGPL our model( in `scripts/885train_[motif/trans/vctree].sh`).

### Model
The trained transformer model can be download from [BaiDuYun](https://pan.baidu.com/s/1vbsFDIHI57o9HxIv5BJiZA) (Password: 5u2o)

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
