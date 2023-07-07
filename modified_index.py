import json

chemin_fichier = "5gtr_fish_coco_train_4k.json"

with open(chemin_fichier, "r") as fichier:
    data = json.load(fichier)

for element in data['images']:
    if 'id' in element:
        element['id'] -= 1
for element in data['annotations']:
    if 'id' in element: 
        element['id']-=1
        element['image_id'] -=1


with open(chemin_fichier, "w") as fichier:
    json.dump(data, fichier, indent=4)