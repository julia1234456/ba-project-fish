import os
import json
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

def get_dataset_dicts(dataset_path):
    with open(os.path.join(dataset_path, "datasets/fish/annotations/annotations.json")) as f:
        annotations = json.load(f)
    
    dataset_dicts = []
    for ann in annotations:
        record = {}
        record["file_name"] = os.path.join(dataset_path, "images", ann["image_id"])
        record["image_id"] = ann["image_id"]
        record["height"] = ann["height"]
        record["width"] = ann["width"]
        
        objs = []
        for obj in ann["objects"]:
            obj_dict = {}
            obj_dict["bbox"] = obj["bbox"]
            obj_dict["bbox_mode"] = BoxMode.XYXY_ABS
            obj_dict["category_id"] = obj["category_id"]
            obj_dict["segmentation"] = obj["segmentation"]
            obj_dict["area"] = obj["area"]
            obj_dict["iscrowd"] = obj["iscrowd"]
            obj_dict["attributes"] = obj["attributes"]
            objs.append(obj_dict)
        
        record["annotations"] = objs
        dataset_dicts.append(record)
    
    return dataset_dicts

def register_dataset(dataset_name, dataset_path):
    DatasetCatalog.register(dataset_name, lambda: get_dataset_dicts(dataset_path))
    MetadataCatalog.get(dataset_name).set(
        thing_classes=["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15"],
        evaluator_type="coco",
        attribute_names=["occluded", "rotation", "track_id", "keyframe"]
    )


register_dataset("fish_train", "/datasets/fish/train")
register_dataset("fish_val", "/datasets/fish/val")