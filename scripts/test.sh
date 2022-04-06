#!/usr/bin/env bash
export PYTHONPATH=/home/lvxinyu/lib/apex:/home/lvxinyu/lib/cocoapi:/home/lvxinyu/lib/scene-graph-benchmark:$PYTHONPATH
if [ $1 == "0" ]; then
    export CUDA_VISIBLE_DEVICES=0 #5,4 #,4 #3,4
    export NUM_GUP=1
    echo "Testing SGDet"
    MODEL_NAME=""
    MODEL_PATH=/mnt/hdd1/lvxinyu/datasets/visual_genome/model/checkpoints/${MODEL_NAME}
    OUTPUT_PATH=${MODEL_PATH}/inference_val/
    mkdir ${OUTPUT_PATH}
    python  ./tools/relation_test_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer_FPGL.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True\
    MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
    DTYPE "float32" \
    SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH $NUM_GUP \
    SOLVER.MAX_ITER 4000 SOLVER.BASE_LR 1e-3 \
    SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
    SOLVER.STEPS "(10000, 16000)" SOLVER.VAL_PERIOD 2000 \
    SOLVER.CHECKPOINT_PERIOD 4000 GLOVE_DIR ./datasets/vg/ \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    MODEL.WEIGHT ./checkpoints/${MODEL_NAME}/model_final.pth \
    MODEL.ROI_RELATION_HEAD.VAL_ALPHA 0 \
    TEST.ALLOW_LOAD_FROM_CACHE False \
    TEST.VAL_FLAG False \
    OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
    MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE 0.3 ;
fi
