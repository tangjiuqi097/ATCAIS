#!/usr/bin/python

import sys
import time
import importlib
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import vot
from vot import Rectangle,Polygon,Point


del os.environ['MKL_NUM_THREADS']    #note:todo when called from matlab, it should be added


import torch
import cv2
import PIL.Image as Image

from pytracking.evaluation import Tracker

tracker = Tracker('ATCAIS_cpu', 'default')
tracker_name=tracker.name
parameters = tracker.get_parameters()
tracker_module = importlib.import_module('pytracking.tracker.{}'.format(tracker.name))
tracker_class = tracker_module.get_tracker_class()
tracker=tracker_class(parameters)


def overlay_boxes( image, state,image_file,tracker_name):

    state = torch.tensor(state).reshape(-1, 4)
    boxes = state.clone()
    boxes[:, 2:4] = state[:, 0:2] + state[:, 2:4]

    for box in boxes:
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), [0, 0, 255], 2
        )

    save_path=os.path.join("/media/tangjiuqi097/ext",tracker_name)
    os.makedirs(save_path,exist_ok=True)
    split_image_file=image_file.split('/')
    seq_name=split_image_file[-3]
    im_name=split_image_file[-1]
    os.makedirs(os.path.join(save_path,seq_name),exist_ok=True)
    im_path=os.path.join(save_path,seq_name,im_name)
    cv2.imwrite(im_path,image[:,:,::-1])


# handle = vot.VOT("rectangle")
handle = vot.VOT("polygon",'rgbd')
selection = handle.region()


if isinstance(selection, Polygon):
    selection = np.array(selection).reshape(-1)
    cx = np.mean(selection[0::2])
    cy = np.mean(selection[1::2])
    x1 = np.min(selection[0::2])
    x2 = np.max(selection[0::2])
    y1 = np.min(selection[1::2])
    y2 = np.max(selection[1::2])
    A1 = np.linalg.norm(selection[0:2] - selection[2:4]) * np.linalg.norm(selection[2:4] - selection[4:6])
    A2 = (x2 - x1) * (y2 - y1)
    s = np.sqrt(A1 / A2)
    w = s * (x2 - x1) + 1
    h = s * (y2 - y1) + 1
    selection = np.array([cx - (w - 1) / 2, cy - (h - 1) / 2, w, h])


# Process the first frame
imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

# imagefile=''.join(imagefile)
# imagefile=imagefile[20:-2]


# image=tracker._read_image(imagefile)
imagefile_rgb=imagefile[0]
imagefile_d=imagefile[1]
assert imagefile_rgb.find("color")>=0
assert imagefile_d.find("depth")>=0
image=tracker._read_image(imagefile_rgb)
depth_im=np.array(Image.open(imagefile_d))

tracker.initialize(image=image, state=selection,depth_im=depth_im)



while True:
    imagefile = handle.frame()
    if not imagefile:
        break

    # imagefile = ''.join(imagefile)
    # imagefile = imagefile[20:-2]

    # image = tracker._read_image(imagefile)

    imagefile_rgb = imagefile[0]
    imagefile_d = imagefile[1]
    assert imagefile_rgb.find("color") >= 0
    assert imagefile_d.find("depth") >= 0
    image = tracker._read_image(imagefile_rgb)
    depth_im = np.array(Image.open(imagefile_d))


    start_time = time.time()
    state = tracker.track(image=image,depth_im=depth_im)

    #overlay_boxes(image, state, imagefile[0], tracker_name)

    #confidence=1
    selection=Rectangle(state[0],state[1],state[2],state[3])
    handle.report(selection)
    time.sleep(0.01)


