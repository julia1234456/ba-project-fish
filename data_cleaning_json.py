import json 

#remove all the images (and annotations associated to these images) must not be contained in the dataset
def gtr_fish_clean(file_name, new_file_name, total_img, nb_img):
    
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
      
    #remove images of unlabelled frames
    nb_img_to_delete = total_img - nb_img +1
    data['images'] = data['images'][:-nb_img_to_delete]
    
    #remove annotations of unlabelled frames
    ann_to_remove = []
    for ann in data['annotations']:
        if ann['image_id'] > nb_img:
            ann_to_remove.append(ann)
    
    for ann in ann_to_remove: 
        data['annotations'].remove(ann)
        

    json_str = json.dumps(data)
    
    
    with open(new_file_name, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, indent=2))

#create a json file containing annotation for validation set 
def gtr_fish_split_val(file_name, new_file_name, nb_img_split):
    
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    #remove the firsts images
    data['images'] = data['images'][nb_img_split:]
    
    #remove the firsts annotations 
    ann_to_remove = []
    for ann in data['annotations']:
        if ann['image_id'] <= nb_img_split:
            ann_to_remove.append(ann)
    
    for ann in ann_to_remove: 
        data['annotations'].remove(ann)

    json_str = json.dumps(data)
    
    
    with open(new_file_name, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, indent=2))

#create json file containing annotations for training set
def gtr_fish_split_train(file_name, new_file_name, total_img, nb_img_split):
    
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Remove the last images from the 'images' category
    
    data['images'] = data['images'][:-(total_img-nb_img_split)]
    
    
    # Remove the last *15 annotations from the 'annotations' category
    ann_to_remove = []
    for ann in data['annotations']:
        if ann['image_id'] > nb_img_split:
            ann_to_remove.append(ann)
    
    for ann in ann_to_remove: 
        data['annotations'].remove(ann)
    
    # Convert the modified dictionary back to a JSON string
    json_str = json.dumps(data)
    
    
    with open(new_file_name, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, indent=2))