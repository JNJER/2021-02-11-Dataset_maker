
# Importing libraries
import os
from PIL import Image, ImageDraw, ImageFont
from time import strftime, gmtime
import time 
import json
from nltk.corpus import wordnet as wn
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
import pandas as pd

# to plot &  display
import matplotlib.pyplot as plt
def pprint(message):
    print('-'*len(message))
    print(message)
    print('-'*len(message))

# parse the root to the init module
def arg_parse(): 
    parser = argparse.ArgumentParser(description='DCNN_training_benchmark/init.py set root')
    parser.add_argument("--root", dest = 'root', help = 
                        "Set the Directory containing images to perform detection upon",
                        default = '/home/jnjer/Bureau/', type = str)
    parser.add_argument("--folder", dest = 'folder', help = 
                        "Set the Directory containing images to perform detection upon",
                        default = 'test', type = str)
    parser.add_argument("--path_out", dest = 'path_out', help = 
                        "Set the Directory containing images to perform detection upon",
                        default = '/home/jnjer/Bureau/det', type = str)
    parser.add_argument("--path_to_ssd", dest = 'path_out_to_ssd', help = 
                        "Set the Directory containing images to perform detection upon",
                        default = '/home/jnjer/Bureau/det/to_ssd', type = str)
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
    parser.add_argument("--min_score", dest = "min_score", help = 
                        "minimum threshold for a detected box to be considered a match for a certain class",
                        default = 0.4)
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
    parser.add_argument("--class_select", dest = 'class_select', help = 
                        "Select a class of interest",
                        default = ('person'), type = tuple)
    parser.add_argument("--class_select_dcnn", dest = 'class_select_dcnn', help = 
                        "Select a class of interest",
                        default = ['animal', 'person'], type = list)
    parser.add_argument("--class_loader", dest = 'class_loader', help = 
                        "Select a class of interest",
                        default = "imagenet_label_to_wordnet_synset.json", type = str)
    parser.add_argument("--N_images_per_class", dest = 'N_images_per_class', help = 
                        "Set the number of images per classe in the train folder",
                        default = 10)
    parser.add_argument("--N_labels", dest = 'N_labels', help = 
                        "Set the number of images per classe in the train folder",
                        default = 30)
    parser.add_argument("--class_roll", dest = 'class_roll', help = 
                        "Select a class to trigger the roll",
                        default = "person", type = str)

    return parser.parse_args()

args = arg_parse()
datetag = args.datetag
json_fname = os.path.join( datetag + '_config_args.json')
load_parse = True # False to custom the config

if load_parse:
    try:
        with open(json_fname, 'rt') as f:
            print(f'file {json_fname} exists: LOADING')
            override = json.load(f)
            args.__dict__.update(override)
    except:
        print(f'Creating file {json_fname}')
        with open(json_fname, 'wt') as f:
            json.dump(vars(args), f, indent=4)
            
# SSD/DCNN variables 
min_score = float(args.min_score)
max_overlap = float(args.max_overlap)
checkpoint = args.checkpoint
top_k = float(args.top_k)
#animated = ( 'bird', 'cat', 'cow', 'dog', 'horse', 'person', 'sheep')
animated = args.class_select
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


N_labels = args.N_labels
i_labels = random.randint(1000, size=N_labels+1)
image_size = args.image_size # default image resolution
image_size_SSD = args.image_size_SSD # default image resolution
id_dl = ''

root = args.root
folder = args.folder
N_images_per_class = args.N_images_per_class

#path configuration
path_in = os.path.join(root, folder)
path_out_to_ssd = args.path_out_to_ssd
path_out = args.path_out
path_out_to_ssd_file = os.path.join(path_out_to_ssd+'/non_trig')
path_out_a = os.path.join(path_out+'/trig')
path_out_b = os.path.join(path_out+'/non_trig')
path_out_crop = os.path.join(path_out+'/crop')
path_out_roll = os.path.join(path_out+'/roll')

class_loader = args.class_loader
with open(class_loader, 'r') as fp: # get all the classes on the data_downloader
    imagenet = json.load(fp)

trig_ = args.class_select_dcnn
match = []
labels = []
for a, img_id in enumerate(imagenet):
    labels.append(imagenet[img_id]['label'])
    syn_ = wn.synset_from_pos_and_offset('n', int(imagenet[img_id]['id'].replace('-n','')))
    if int(img_id) in i_labels:
        id_dl += 'n' + (imagenet[img_id]['id'].replace('-n','')) + ' '
    sem_ = syn_.hypernym_paths()[0]
    for i in np.arange(len(sem_)):
        if sem_[i].lemmas()[0].name() in trig_ :
            match.append(a)

# Store the config variables : 
df_config = pd.DataFrame([], columns=['path_in', 'path_out', 'HOST', 'datetag', 'image_size', 'image_size_SSD', 'imagenet_label_root', 'min_score', 'max_overlap', 'checkpoint', 'top_k', 'class_crop', 'class_roll'])#
                
pprint('init.py : Done !')
