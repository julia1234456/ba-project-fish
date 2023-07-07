import json

def pad_id(id, num_digits):
    return str(id).zfill(num_digits)
=
input_file = '../datasets/fish/annotations/gtr_fish_coco_train.json'
output_file = '../datasets/fish/annotations/detectron_fish_coco_train.json'

num_digits = 6

with open(input_file, 'r') as f:
    data = json.load(f)

for item in data["images"]:
    item['id'] = pad_id(item['id'], num_digits)
    
for item in data["annotations"]:
    item['image_id'] = pad_id(item['image_id'], num_digits)

with open(output_file, 'w') as f:
    json.dump(data, f)