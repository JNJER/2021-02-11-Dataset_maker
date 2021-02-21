

#import model's script and set the output file
from DCNN_dataset_maker.models import * 
df_config.loc[datetag] = {'path_in':path_in, 'path_out':path_out, 'HOST':HOST, 'datetag': datetag,
                               'image_size':image_size, 'image_size_SSD':image_size_SSD, 
                               'imagenet_label_root':imagenet_label_root, 'min_score':min_score, 
                               'max_overlap':max_overlap, 'checkpoint':checkpoint, 'top_k':top_k,
                                'class_crop':class_crop, 'class_roll':class_roll}
for i_image, (data, label) in enumerate(image_dataset):
    i = 0 
    j = 0
    for name in models.keys():
        model = models[name]
        out = model(data.unsqueeze(0).to(device)).squeeze(0)
        percentage = torch.nn.functional.softmax(out, dim=0) * 100
        _, indices = torch.sort(out, descending=True)
        for idx in indices[:2]:
            if idx in match:
                i += 1
                j += percentage[idx].item()
            print(f'The {name} model get {labels[idx]} at {percentage[idx].item():.2f} % confidence {i}')
    if i>=5 or j>= 250: 
        shutil.copy(image_dataset.imgs[i_image][0], path_out_a)
    else:
        shutil.copy(image_dataset.imgs[i_image][0], path_out_b)   
    pprint(image_dataset.imgs[i_image][0] + ' ' + str(j))
df_config
