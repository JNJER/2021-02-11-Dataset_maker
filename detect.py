from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
import argparse
import os 
import cv2 
import os.path as osp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    parser = argparse.ArgumentParser(description='SSD_detection_crop_module')
   
    parser.add_argument("--path", dest = 'path', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "det", type = str)
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
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "300", type = str)

    return parser.parse_args()

args = arg_parse()
path = args.path
min_score = float(args.min_score)
max_overlap = float(args.max_overlap)
checkpoint = args.checkpoint
top_k = float(args.top_k)
reso = float(args.reso)
det = args.det

# Load model checkpoint
checkpoint = 'checkpoint_ssd300.pth.tar'
try:
    checkpoint = torch.load(checkpoint)
except:
    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()
animated= ( 'bird', 'cat', 'cow', 'dog', 'horse', 'person', 'sheep')
# Transforms
resize = transforms.Resize((int(reso), int(reso)))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """
    
    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image
    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.load_default()
    #font = ImageFont.truetype("./calibril.ttf", 15)

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
        draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
            det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                  font=font)
    del draw
    
    return annotated_image

if __name__ == '__main__':
    for elem in os.listdir(path):
        img_path = f'{path}/{elem}'
        original_image = Image.open(img_path, mode='r')
        original_image = original_image.convert('RGB')
        detect(original_image, min_score, max_overlap, top_k).show()
