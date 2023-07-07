
from detectron2.engine import DefaultTrainer
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common libraries
import sys
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
from detectron2.data import MetadataCatalog

# GTR libraries
sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from gtr.config import add_gtr_config
from gtr.predictor import GTRPredictor, TrackingVisualizer

import imageio

#register_coco_instances("fish_train", {}, "datasets/5fish/annotations/5gtr_fish_coco_train.json", "datasets/5fish/train")
register_coco_instances("fish_val", {}, "datasets/5fish/annotations/5gtr_fish_coco_val.json", "datasets/5fish/val")


# Build the detector and download our pretrained weights
cfg = get_cfg()
cfg.NUM_GPUS = 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cfg.DATASETS.TRAIN = ("fish_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
add_centernet_config(cfg)
add_gtr_config(cfg)
cfg.merge_from_file("configs/GTR_FISH_NEW_CONFIG.yaml")
cfg.MODEL.WEIGHTS = 'output/FISH/5fish_2000fr_5000it/model_final.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
metadata = MetadataCatalog.get(
    cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused")
predictor = GTRPredictor(cfg)
tracker_visualizer = TrackingVisualizer(metadata)


# Functions to load and same videos
import imageio
from IPython.core.display import Video
from IPython.display import display
def show_video(filename, frames, fps=5):
    imageio.mimwrite(
        filename, [x[..., ::-1] for x in frames], fps=fps)
    display(Video(filename, embed=True))

def _frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break
        

# Load images from video
video_path = 'docs/5fish_video_CROPPED.mp4'
video = cv2.VideoCapture(video_path)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames_per_second = video.get(cv2.CAP_PROP_FPS)
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
basename = os.path.basename(video_path)
codec, file_ext = "mp4v", ".mp4"
# codec = 'H264'
frames = [x for x in _frame_from_video(video)]
video.release()

show_video('5fish_input.mp4', frames)




# Create a video writer with FFMPEG
writer = imageio.get_writer('5fish_input.mp4', macro_block_size=1)
# Iterate over frames and add them to the writer
for frame in frames:
    writer.append_data(frame)
# Close the writer to finalize the video
writer.close()

# Run model
outputs = predictor(frames)

# Post processing and save output video
def _process_predictions(tracker_visualizer, frame, predictions):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predictions = predictions["instances"].to('cpu')
    vis_frame = tracker_visualizer.draw_instance_predictions(
        frame, predictions)
    vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
    return vis_frame

out_frames = []
for frame, instances in zip(frames, outputs):
    out_frame = _process_predictions(tracker_visualizer, frame, instances)
    out_frames.append(out_frame)

show_video('./output/FISH/5fish_output.mp4', out_frames)