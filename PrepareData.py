#adopted from: https://github.com/memray/OpenNMT-kpg-release/blob/master/notebook/json_process.ipynb

import os
import sys
import re
import json
import numpy as np
from collections import defaultdict
from Utils import *
import statistics

def get_length_stat(lengths):
    print('Max_len=%d, Mean_len=%d, Median_len=%d, Stdv_len=%d'
          % (max(lengths), statistics.mean(lengths), statistics.median(lengths), statistics.stdev(lengths)))

trainOrTest = 'valid'

dataset_names = []
if trainOrTest == 'test':
    dataset_names = ['inspec', 'krapivin', 'nus', 'semeval', 'kp20k', 'duc', 'stackexchange']
else:
    dataset_names = ['kp20k']
json_base_dir = 'E:\ResearchData\Keyphrase Generation\data\json\\'

for dataset_name in dataset_names:
    print(dataset_name,"=======================================")

    source_lengths = []
    tgt_lengths = []
    present_tgt_lengths = []
    absent_tgt_lengths = []
    if trainOrTest == 'test':
        input_json_path = os.path.join(json_base_dir, dataset_name, '%s_test.json' % dataset_name)
        output_json_path = os.path.join(json_base_dir, dataset_name, '%s_test_tokenized.json' % dataset_name)
    elif trainOrTest == 'valid':
        input_json_path = os.path.join(json_base_dir, dataset_name, '%s_valid.json' % dataset_name)
        output_json_path = os.path.join(json_base_dir, dataset_name, '%s_valid_tokenized.json' % dataset_name)
    else:
        input_json_path = os.path.join(json_base_dir, dataset_name, '%s_train.json' % dataset_name)
        output_json_path = os.path.join(json_base_dir, dataset_name, '%s_train_tokenized.json' % dataset_name)

    doc_count, present_doc_count, absent_doc_count = 0, 0, 0
    tgt_num, present_tgt_num, absent_tgt_num = [], [], []

    unique_titles = []
    with open(input_json_path, 'r', encoding="utf8", ) as input_json, open(output_json_path, 'w') as output_json:
        for json_line in input_json:
            json_dict = json.loads(json_line)

            if dataset_name == 'stackexchange':
                json_dict['abstract'] = json_dict['question']
                json_dict['keywords'] = json_dict['tags']
                del json_dict['question']
                del json_dict['tags']

            title = json_dict['title']
            abstract = json_dict['abstract']
            keywords = json_dict['keywords']


            if isinstance(keywords, str):
                keywords = keywords.split(';')
                json_dict['keywords'] = keywords
            # remove all the abbreviations/acronyms in parentheses in keyphrases
            keywords = [re.sub(r'\(.*?\)|\[.*?\]|\{.*?\}', '', kw) for kw in keywords]

            # tokenize text
            title_token = meng17_tokenize(title)
            abstract_token = meng17_tokenize(abstract)
            keywords_token = [meng17_tokenize(kw) for kw in keywords]

            # replace numbers
            title_token = replace_numbers_to_DIGIT(title_token, k=1)
            abstract_token = replace_numbers_to_DIGIT(abstract_token, k=1)
            keywords_token = [replace_numbers_to_DIGIT(kw, k=1) for kw in keywords_token]
            #print(keywords_token)
            keywords_token = remove_duplicate_keyphrases(keywords_token)
            if len(keywords_token)>0 and title not in unique_titles:
                unique_titles.append(title)
                src_token = title_token + ["."] + abstract_token
                tgts_token = keywords_token

                #             print(json_dict)
                #             print(src_token)
                #             print(tgts_token)

                # split tgts by present/absent



                src_seq = src_token
                tgt_seqs = [x for x in tgts_token if len(x)<10]


                present_tgt_flags, occurance_positions, _ = if_present_duplicate_phrases(src_seq, tgt_seqs)
                present_tgts = [tgt for tgt, present in zip(tgt_seqs, present_tgt_flags) if present]
                absent_tgts = [tgt for tgt, present in zip(tgt_seqs, present_tgt_flags) if ~present]

                #sorting the present keyphrases by their occurance
                present_phrases_occurance_positions = [x for x in occurance_positions if x!=-1]
                assert len(present_tgts) == len(present_phrases_occurance_positions)

                #print(present_phrases_occurance_positions,present_tgts)

                present_tgts = [x for _,x in sorted(zip(present_phrases_occurance_positions,present_tgts))]
                tgts_token = present_tgts + absent_tgts
                #print(present_tgts)

                doc_count += 1
                present_doc_count = present_doc_count + 1 if len(present_tgts) > 0 else present_doc_count
                absent_doc_count = absent_doc_count + 1 if len(absent_tgts) > 0 else absent_doc_count

                tgt_num.append(len(tgt_seqs))
                present_tgt_num.append(len(present_tgts))
                absent_tgt_num.append(len(absent_tgts))

                # write to output json
                source_lengths.append(len(src_token))
                tgt_lengths.append(len(tgts_token))
                present_tgt_lengths.append(len(present_tgts))
                absent_tgt_lengths.append(len(absent_tgts))
                tokenized_dict = {'src': src_token, 'tgt': tgts_token,
                                  'present_tgt': present_tgts, 'absent_tgt': absent_tgts}
                json_dict['tokenized'] = tokenized_dict
                output_json.write(json.dumps(json_dict) + '\n')

    print('#doc=%d, #present_doc=%d, #absent_doc=%d, #tgt=%d, #present=%d, #absent=%d'
          % (doc_count, present_doc_count, absent_doc_count,
             sum(tgt_num), sum(present_tgt_num), sum(absent_tgt_num)))
    print('Source Lengths')
    get_length_stat(source_lengths)
    print('Target Lengths')
    get_length_stat(tgt_lengths)
    print('Present target Lengths')
    get_length_stat(present_tgt_lengths)
    print('Absent target Lengths')
    get_length_stat(absent_tgt_lengths)

#remove_testing_instances_from_training_set(location):
all_test_set_titles = []
test_set_names = ['inspec', 'krapivin', 'nus', 'semeval', 'kp20k', 'duc', 'stackexchange']

for dataset_name in test_set_names:
    test_json_path = os.path.join(json_base_dir, dataset_name, '%s_test_tokenized.json' % dataset_name)
    with open(test_json_path, 'r', encoding="utf8", ) as input_json:
        for json_line in input_json:
            json_dict = json.loads(json_line)
            title = json_dict['title']
            all_test_set_titles.append(title)

input_json_path = os.path.join(json_base_dir, 'kp20k', '%s_train_tokenized.json' % 'kp20k')
output_json_path = os.path.join(json_base_dir, 'kp20k', '%s_train_tokenized_duplicates_from_test_sets_removed.json' % 'kp20k')

count = 0
with open(input_json_path, 'r', encoding="utf8", ) as input_json, open(output_json_path, 'w') as output_json:
    for json_line in input_json:
        json_dict = json.loads(json_line)
        title = json_dict['title']
        print(title)
        if title not in all_test_set_titles:
            output_json.write(json.dumps(json_dict) + '\n')
            count+=1
print(count)