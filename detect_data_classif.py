
from DCNN_dataset_maker.models import *
image_dataset_SSD = ImageFolder(path_in_ssd, transform=transform_SSD) # save the dataset
i = 0 
suppress=None

for i_image, (data, label) in enumerate(image_dataset_SSD):
    
        # Forward prop.
        predicted_locs, predicted_scores = model(data.unsqueeze(0).to(device)) # Move to default device

        # Detect objects in SSD output
        det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                                 max_overlap=max_overlap, top_k=top_k)

        # Decode class integer labels
        det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

        # Sort by the labels
        is_anime = False
        for lab_ in det_labels :
            if lab_ in animated:
                is_anime = True
                
        if is_anime : 
            i+=1
            #print(image_dataset.imgs[i_image][0], is_anime)
            shutil.copy(image_dataset.imgs[i_image][0], path_out_a)
        else:
            #print(image_dataset.imgs[i_image][0], is_anime)
            shutil.copy(image_dataset.imgs[i_image][0], path_out_ssd)
print(i)
