import sys
import os
print(sys.path)
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
import matplotlib.pyplot as plt
import cv2
from detectron2.config import CfgNode as CN
from layoutlmv3_model.ditod import add_vit_config

image = 'examples/test.png'
cfg = get_cfg()
cfg.MODEL.MAX_LENGTH = 510
add_vit_config(cfg)
cfg.merge_from_file("/Users/MinhHieu/Desktop/thesis/final/SC_19127400_19127074_PHÁT HIỆN BỐ CỤC CỦA VĂN BẢN DỰA TRÊN MÔ HÌNH TRANSFORMERS/DATASET/PUBLAYNET/model/config.yaml")
cfg.MODEL.MASK_ON = False
cfg.MODEL.WEIGHTS = "/Users/MinhHieu/Desktop/thesis/final/SC_19127400_19127074_PHÁT HIỆN BỐ CỤC CỦA VĂN BẢN DỰA TRÊN MÔ HÌNH TRANSFORMERS/DATASET/PUBLAYNET/model/model_final.pth" # path for final model
predictor = DefaultPredictor(cfg)
im = cv2.imread(os.path.join("/Users/MinhHieu/Desktop/thesis/final/SC_19127400_19127074_PHÁT HIỆN BỐ CỤC CỦA VĂN BẢN DỰA TRÊN MÔ HÌNH TRANSFORMERS/SOURCE/PUBLAYNET/layoutlmv3_thesis/examples/test.png", image))
outputs = predictor(im)
v = Visualizer(im[:, :, ::-1],
                metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                scale=0.5,
                instance_mode=ColorMode.IMAGE_BW)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
img = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_RGBA2RGB)
plt.imshow(img)
plt.show()