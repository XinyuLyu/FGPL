#!/usr/bin/env bash
export PYTHONPATH=/home/lvxinyu/lib/apex:/home/lvxinyu/lib/cocoapi:/home/lvxinyu/lib/scene-graph-benchmark:$PYTHONPATH
if [ $1 == "1" ]; then
    export CUDA_VISIBLE_DEVICES=2
    export NUM_GUP=1
    echo "TRAINING Predcls"
    mode="Predcls_"
    MODEL="motif_FGPL"
    MODEL_NAME=${mode}${MODEL}
    mkdir ./checkpoints/${MODEL_NAME}/
    python ./tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_motif_FPGL.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True\
    MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
    DTYPE "float32" \
    SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH $NUM_GUP \
    SOLVER.MAX_ITER 16000 SOLVER.BASE_LR 1e-3 \
    SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
    SOLVER.STEPS "(10000, 16000)" SOLVER.VAL_PERIOD 2000 \
    SOLVER.CHECKPOINT_PERIOD 4000 GLOVE_DIR ./datasets/vg/ \
    MODEL.PRETRAINED_DETECTOR_CKPT /mnt/hdd1/lvxinyu/datasets/visual_genome/model/checkpoints/pretrained_faster_rcnn/model_final.pth \
    MODEL.ROI_RELATION_HEAD.USE_EXTRA_LOSS True \
    MODEL.ROI_RELATION_HEAD.USE_LOGITS_REWEIGHT  True \
    MODEL.ROI_RELATION_HEAD.MITIGATION_FACTOR_HYPER  1.5 \
    MODEL.ROI_RELATION_HEAD.COMPENSATION_FACTOR_HYPRT  2.0 \
    MODEL.ROI_RELATION_HEAD.USE_CONTRA_LOSS  True \
    MODEL.ROI_RELATION_HEAD.USE_CONTRA_BCE  True \
    MODEL.ROI_RELATION_HEAD.CONTRA_DISTANCE_LOSS_VALUE  0.6 \
    MODEL.ROI_RELATION_HEAD.CONTRA_DISTANCE_LOSS_COF  0.1 \
    MODEL.ROI_RELATION_HEAD.CANDIDATE_NUMBER  5 \
    OUTPUT_DIR ./checkpoints/${MODEL_NAME} ;
fi
