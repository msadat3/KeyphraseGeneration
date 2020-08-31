from EasyDataAugmentation import *

import json

input_json_path = "E:\ResearchData\Keyphrase Generation\data\json\kp20k\\kp20k_train.json"
output_json_path = "E:\ResearchData\Keyphrase Generation\data\json\kp20k\\kp20k_train_augmented.json"

with open(input_json_path, 'r', encoding="utf8", ) as input_json, open(output_json_path, 'w') as output_json:
    count = 0
    for json_line in input_json:
        #count+=1
        #print(count)
        json_dict = json.loads(json_line)

        title = json_dict['title']
        abstract = json_dict['abstract']

        #print('title', title)
        aug_titles = []
        aug_abstracts = []
        try:
            aug_titles = eda(title,num_aug=4)
            #print('aug_titles',aug_titles)
            #print('abstract', abstract)
            aug_abstracts = eda(abstract, num_aug=4)
        except Exception as e:
            print(title)
            print(e)
       # print('aug_abstract', aug_abstracts)


        output_json.write(json.dumps(json_dict) + '\n')
        if len(aug_titles) > 0 and len(aug_abstracts) > 0:
            for t, a in zip(aug_titles, aug_abstracts):
                #print(json_dict['keywords'])
                output_json_dict = json_dict

                output_json_dict['title'] = t
                output_json_dict['abstract'] = a
                #print(output_json_dict)
                output_json.write(json.dumps(output_json_dict) + '\n')

