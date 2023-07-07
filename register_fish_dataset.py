from detectron2.data.datasets import register_coco_instances
import random
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog
import cv2

register_coco_instances("fish_train", {}, "../datasets/fish/annotations/gtr_fish_coco_train.json", "datasets/fish/train")
register_coco_instances("fish_val", {}, "datasets/fish/gtr_fish_coco_val.json", "datasets/fish/val")

nuts_metadata = MetadataCatalog.get('fish_train')
dataset_dicts = DatasetCatalog.get("fish_train")
