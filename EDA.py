
import os
import json

json_base_dir = 'E:\ResearchData\Keyphrase Generation\data\json\\'
dataset_name = 'kp20k'
training_json_path = os.path.join(json_base_dir, dataset_name, '%s_test_tokenized.json' % dataset_name)

source_lengths = []

with open(training_json_path, 'r', encoding="utf8", ) as input_json:
    for json_line in input_json:
        json_dict = json.loads(json_line)
        print(type(json_dict['tokenized']['tgt']))