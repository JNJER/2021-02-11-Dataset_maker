
# Importing libraries
import os
from PIL import Image, ImageDraw, ImageFont
from time import strftime, gmtime
import time 
import json
import time 
import numpy as np
import imageio
from utils import *
from numpy import random
import argparse
import cv2 
import os.path as osp
import shutil
from torchvision.datasets import ImageFolder

# to plot &  display
import matplotlib.pyplot as plt
def pprint(message):
    print('-'*len(message))
    print(message)
    print('-'*len(message))

# parse the root to the init module
def arg_parse(): 
    parser = argparse.ArgumentParser(description='DCNN_training_benchmark/init.py set root')
    parser.add_argument("--path_in", dest = 'path_in', help = 
                        "Set the Directory containing images to perform detection upon",
                        default = '/Users/jjn/Desktop/test', type = str)
    parser.add_argument("--path_out", dest = 'path_out', help = 
                        "Set the Directory containing images to perform detection upon",
                        default = '/Users/jjn/Desktop/test_1', type = str)
    parser.add_argument("--HOST", dest = 'HOST', help = 
                    "Set the name of your machine",
                    default = os.uname()[1], type = str)
    parser.add_argument("--datetag", dest = 'datetag', help = 
                    "Set the datetag of the result's file",
                    default = strftime("%Y-%m-%d", gmtime()), type = str)
    parser.add_argument("--image_size", dest = 'image_size', help = 
                    "Set the image_size of the input",
                    default = 256)
    parser.add_argument("--image_size_SSD", dest = 'image_size_SSD', help = 
                    "Set the image_size of the input",
                    default = 300)
    parser.add_argument("--imagenet_label_root", dest = 'imagenet_label_root', help = 
                        "Set the Directory containing imagenet labels",
                        default = 'imagenet_classes.txt', type = str)
    parser.add_argument("--min_score", dest = "min_score", help = 
                        "minimum threshold for a detected box to be considered a match for a certain class",
                        default = 0.6)
    parser.add_argument("--max_overlap", dest = "max_overlap", help = 
                        "maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)",
                        default = 0.3)
    parser.add_argument("--checkpoint", dest = 'checkpoint', help = 
                        "weightsfile",
                        default = "checkpoint_ssd300.pth.tar", type = str)
    parser.add_argument("--top_k", dest = 'top_k', help = 
                        "if there are a lot of resulting detection across all classes, keep only the top 'k'",
                        default = 200 , type = str)
    parser.add_argument("--class_crop", dest = 'class_crop', help = 
                        "Select a class to trigger the crop",
                        default = "person", type = str)
    parser.add_argument("--class_roll", dest = 'class_roll', help = 
                        "Select a class to trigger the roll",
                        default = "person", type = str)
    return parser.parse_args()

args = arg_parse()

# SSD variables 
min_score = float(args.min_score)
max_overlap = float(args.max_overlap)
checkpoint = args.checkpoint
top_k = float(args.top_k)
animated= ( 'bird', 'cat', 'cow', 'dog', 'horse', 'person', 'sheep')
class_crop = args.class_crop
class_roll = args.class_roll

# figure's variables
fig_width = 20
phi = (np.sqrt(5)+1)/2 # golden ratio
phi = phi**2
colors = ['b', 'r', 'k','g']

# host & date's variables 
HOST = args.HOST
datetag = args.datetag
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#dataset configuration
image_size = args.image_size # default image resolution
image_size_SSD = args.image_size_SSD # default image resolution
id_dl = ''

path_in = args.path_in 
path_out = args.path_out 
path_out_a = os.path.join(path_out+'/animated')
path_out_b = os.path.join(path_out+'/non_animated')
path_out_crop = os.path.join(path_out+'/crop')
path_out_roll = os.path.join(path_out+'/roll')

imagenet_label_root = args.imagenet_label_root
with open(imagenet_label_root) as f:
    labels = [line.strip() for line in f.readlines()]
labels[0].split(', ')
labels = [label.split(', ')[1].lower().replace('_', ' ') for label in labels]

def is_animated(x):
    if x <=397:
        #alive.append(True)
        return 1
    else:
        #alive.append(False)
        return
pprint('init.py : Done !')
