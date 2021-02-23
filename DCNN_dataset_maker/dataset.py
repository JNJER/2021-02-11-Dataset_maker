
from init import *

# check if the folder exist
if os.path.isdir(path_out):
    list_dir = os.listdir(path_out)
    if not "trig" in list_dir:
        pprint(f"No existing path match for this folder, creating a folder at {path_out_a}")
        os.makedirs(path_out_a)
        
    if not "non_trig" in list_dir:
        pprint(f"No existing path match for this folder, creating a folder at {path_out_b}")
        os.makedirs(path_out_b)
        
    if not "crop" in list_dir:
        pprint(f"No existing path match for this folder, creating a folder at {path_out_crop}")
        os.makedirs(path_out_crop)
        
    if not "roll" in list_dir:
        pprint(f"No existing path match for this folder, creating a folder at {path_out_roll}")
        os.makedirs(path_out_roll)
        
    if not "to_ssd" in list_dir:
        pprint(f"No existing path match for this folder, creating a folder at {path_out_to_ssd}")
        os.makedirs(path_out_to_ssd_file)
        
    else:
        pprint(f"The folder already exists, it includes: {list_dir}")

# no folder, creating one 
else :
    pprint(f"No existing path match for this folder, creating a folder at {path_out}")
    os.makedirs(path_out)
    os.makedirs(path_out_a)
    os.makedirs(path_out_b)
    list_dir = os.listdir(path_out)
    print(f"Now the folder contains : {os.listdir(path_out)}")

if os.path.isdir(path_in):
    list_dir = os.listdir(path_in)
    print(f"The folder already exists, it includes: {list_dir}")


    # no folder, creating one 
else :
    print(f"No existing path match for this folder, creating a folder at {path_in}")
    os.makedirs(path_in)

list_dir = os.listdir(path_in)

# if the folder is empty, download the images using the ImageNet-Datasets-Downloader
if len(list_dir) < N_labels : 
    print('This folder do not have anough classes, downloading some more') 
    cmd =f"python3 ImageNet-Datasets-Downloader/downloader.py -data_root {root} -data_folder {folder} -images_per_class {N_images_per_class} -use_class_list True  -class_list {id_dl} -multiprocessing_workers 0"
    print('Command to run : '+ cmd)
    os.system(cmd) # running it
    list_dir = os.listdir(path_in)
    print("Now the folder contains :" , os.listdir(path_in))
    
elif len(os.listdir(path_in)) == N_labels :
    print(f'The folder already contains : {len(list_dir)} classes')

else : # if there are to many folders delete some
    print('The folder have to many classes, deleting some')
    for elem in os.listdir(path_in):
        contenu = os.listdir(f'{path_in}/{elem}')
        if len(os.listdir(path_in)) > N_labels :
            for y in contenu:
                os.remove(f'{path_in}/{elem}/{y}') # delete exces folders
            try:
                os.rmdir(f'{path_in}/{elem}')
            except:
                os.remove(f'{path_in}/{elem}')
    list_dir = os.listdir(path_in)
    print("Now the folder contains :" , os.listdir(path_in))
