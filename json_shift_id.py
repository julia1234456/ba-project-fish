import json

def shift_id(json_file):
    # Load the JSON file
    with open(json_file, "r") as f:
        data = json.load(f)

    # Shift all indexes by 1
    for item in data["images"]:
        item['id'] -=1
    for item in data["annotations"]:
        item['id'] -=1
        item['image_id'] -=1
        
    # Save the updated JSON file
    with open("../datasets/fish/annotations/gtr_fish_coco_val.json", "w") as f:
        json.dump(data, f)
        
shift_id("../datasets/fish/annotations/gtr_fish_coco_val.json")