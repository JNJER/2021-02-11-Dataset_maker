
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
    N_X = original_image.width # Get the size of the picture
    N_Y = original_image.height
    original_dims = torch.FloatTensor(
        [N_X, N_Y, N_X, N_Y]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # Choose the region of interest as the highest saliency for a given label
    det_score = det_scores[0].tolist()
    ROI = det_score.index(max(det_score)) 
    if class_roll in det_labels:
        if det_labels[ROI] != class_roll:
            det_score[ROI] = 0
            ROI = det_score.index(max(det_score))    

    for x in det_labels:
        if x == class_roll : 
            left =  float(det_boxes[ROI][0]) #Get the position of the bounding box
            upper = float(det_boxes[ROI][1])
            right = float(det_boxes[ROI][2])
            lower = float(det_boxes[ROI][3])
            c1 = int((N_Y//2)-(((lower-upper)//2)+upper)) # "distance" from the center of the picture to the center of the box on the 0 axis
            c2 = int((N_X//2)-(((right-left)//2)+left)) # "distance" from the center of the picture to the center of the box on the 1 axis
            annotated_image = np.roll(original_image, c2, axis=1) # sliding the gaze to the right by moving the picture to the left
            annotated_image = np.roll(annotated_image, c1, axis=0)# sliding the gaze up by moving the picture to the bottom
            print(f'There is a {class_roll}(s) in {image_dataset.imgs[i_image][0]}.')
            annotated_image = Image.fromarray(np.uint8(annotated_image)).save(f'{path_out_roll}/{i_image}.jpg')
        else:
            print (f"No {class_roll} in {image_dataset.imgs[i_image][0]}")        
