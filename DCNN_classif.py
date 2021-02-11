

#import model's script and set the output file
from DCNN_dataset_maker.models import * 
i = 0 
alive = []
for i_image, (data, label) in enumerate(image_dataset):
    for name in models.keys():
        model = models[name]
        tic = time.time()
        out = model(data.unsqueeze(0).to(device)).squeeze(0)
        percentage = torch.nn.functional.softmax(out, dim=0) * 100
        _, indices = torch.sort(out, descending=True)
        dt = time.time() - tic  
        for idx in indices[:1]:
            if percentage[idx].item() > 25 and  idx <=397:
                alive.append(1)
                #print(image_dataset.imgs[i_image][0], alive)
    if len(alive)>=2 : 
        i+=1
        print(f'The {name} model get {labels[idx]} at {percentage[idx].item():.2f} % confidence in {dt:.3f} seconds') # Affichage du meilleur pourcentage
        shutil.copy(image_dataset.imgs[i_image][0], path_out_a)
    else:
        shutil.copy(image_dataset.imgs[i_image][0], path_out_b)   
    alive = []
print(i)
