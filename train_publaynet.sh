python train_net.py --config-file cascade_layoutlmv3.yaml --num-gpus 1 \
        MODEL.WEIGHTS path/to/layoutlmv3-base/pytorch_model.bin \
        OUTPUT_DIR path/to/output/dir \
        PUBLAYNET_DATA_DIR_TRAIN path/to/data/train \
        PUBLAYNET_DATA_DIR_TEST path/to/data/val