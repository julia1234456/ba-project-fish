from detectron2.engine import DefaultTrainer
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common libraries
import numpy as np
import os, json,random
import cv2
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode

register_coco_instances("fish_train", {}, "datasets/5fish_4k/annotations/5gtr_fish_coco_train_4k.json", "datasets/5fish_4k/train")
register_coco_instances("fish_val", {}, "datasets/5fish_4k/annotations/5gtr_fish_coco_val_4k.json", "datasets/5fish_4k/val")
   

cfg = get_cfg()
cfg.NUM_GPUS = 2
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("fish_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml") # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2 # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00090 # pick a good LR
cfg.SOLVER.MAX_ITER = 10000 # 300 iterations seems good enough for this toy dataset
cfg.SOLVER.STEPS = [] # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5 # 5 different zebra fishes. (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # set the testing threshold for this model
cfg.DATASETS.TEST = ("fish_val", )
predictor = DefaultPredictor(cfg)

metadata = MetadataCatalog.get("fish_val")
image_ids = DatasetCatalog.get("fish_val")
dataset_dicts = [DatasetCatalog.get("fish_val")[i] for i in range(len(image_ids))]

output_dir = "./output"

i=0
for d in random.sample(dataset_dicts, 10):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=0.9)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    output_path = os.path.join(output_dir, f"image_{i}.png")
    success =cv2.imwrite(output_path, v.get_image()[:, :, ::-1])
    i+=1
    print(success)
    


from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("fish_val", output_dir="./output")
val_loader = build_detection_test_loader(cfg, "fish_val")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
# another equivalent way to evaluate the model is to use `trainer.test`