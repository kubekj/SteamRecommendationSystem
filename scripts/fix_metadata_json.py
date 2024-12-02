# file to fix original json file from dataset
# https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam

import json

def fix_json_file(input_f, output_f):
    json_objects = []

    with open(input_f, 'r', encoding='utf-8') as f:
        current_object = ""
        for line in f:
            current_object += line
            try:
                json_obj = json.loads(current_object)
                json_objects.append(json_obj)
                current_object = ""
            except json.JSONDecodeError:
                continue

    with open(output_f, 'w', encoding='utf-8') as f:
        json.dump(json_objects, f, indent=4)

    print(f"Successfully processed {len(json_objects)} objects")
    return len(json_objects)


if __name__ == "__main__":
    data_dir = "../data"
    input_file = f"{data_dir}/games_metadata.json"
    output_file = f"{data_dir}/games_metadata_fixed.json"
    num_objects = fix_json_file(input_file, output_file)
