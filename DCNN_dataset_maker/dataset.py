
from init import *

# check if the folder exist
if os.path.isdir(path_out):
    list_dir = os.listdir(path_out)
    if not "animated" in list_dir:
        pprint(f"No existing path match for this folder, creating a folder at {path_out_a}")
        os.makedirs(path_out_a)
    if not "non_animated" in list_dir:
        pprint(f"No existing path match for this folder, creating a folder at {path_out_b}")
        os.makedirs(path_out_b)
    if not "crop" in list_dir:
        pprint(f"No existing path match for this folder, creating a folder at {path_out_b}")
        os.makedirs(path_out_crop)
    if not "roll" in list_dir:
        pprint(f"No existing path match for this folder, creating a folder at {path_out_b}")
        os.makedirs(path_out_roll)
    else:
        pprint(f"The folder already exists, it includes: {list_dir}")

# no folder, creating one 
else :
    print(f"No existing path match for this folder, creating a folder at {path_out}")
    os.makedirs(path_out)
    os.makedirs(path_out_a)
    os.makedirs(path_out_b)
    list_dir = os.listdir(path_out)
    pprint(f"Now the folder contains : {os.listdir(path_out)}")

if os.path.isdir(path_in):
    list_dir = os.listdir(path_in)
    pprint(f"The folder already exists, it includes: {list_dir}")
else :
    pprint(f"No existing path match for this folder at {path_in} check your data !")
