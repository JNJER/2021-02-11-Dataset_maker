
from DCNN_dataset_maker.models import *

for i_image, (data, label) in enumerate(image_dataset_SSD):
    # Forward prop.
    predicted_locs, predicted_scores = model(data.unsqueeze(0).to(device)) # Move to default device

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)
    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_image = Image.open(image_dataset.imgs[i_image][0], mode='r')
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    annotated_images = []
    i=0
    for x in det_labels:
        if x == class_crop :
            left =  float(det_boxes[i][0])
            upper = float(det_boxes[i][1])
            right = float(det_boxes[i][2])
            lower = float(det_boxes[i][3])
            annotated_image = original_image.crop((left, upper, right, lower))
            i += 1
            print(f'There is {i} {class_crop}(s) in {image_dataset.imgs[i_image][0]}.')
            annotated_image.save(f'{path_out_crop}/{i_image}_{i}.jpg')
        else:
             print (f"No {class_crop} in {image_dataset.imgs[i_image][0]}")        
