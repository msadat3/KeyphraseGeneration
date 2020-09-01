import json
from Utils import *
import os
from nltk.probability import FreqDist
from transformers import BartTokenizer

'''target_start_token = "<s>"
target_end_token = "</s>"
separator_token = "<sep>"
present_absent_separator_token = "<eofpr>"
pad_token = "<pad>"'''



tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
word_to_idx = tokenizer.get_vocab()
idx_to_word = {v: k for k, v in word_to_idx.items()}

#target_start_token = ""
target_end_token = "</s>"
separator_token = ","
present_absent_separator_token = ";"
pad_token = "<pad>"



def create_combined_target_sequence_BART(json_location, output_location, present_max, absent_max):
    #max_length = -1
    with open(json_location, 'r', encoding="utf8", ) as input_json,  open(output_location, 'w') as output_json:
        for json_line in input_json:
            json_dict = json.loads(json_line)
            #print(json_dict)
            present_tgts = json_dict['tokenized']['present_tgt']
            absent_tgts = json_dict['tokenized']['absent_tgt']
            if len(present_tgts) > present_max:
                present_tgts = present_tgts[0:present_max]
            if len(absent_tgts) > absent_max:
                absent_tgts = absent_tgts[0:absent_max]
            combined = ""
            for tgt in present_tgts:
                combined += ' '.join(tgt)
                combined+= separator_token
            present_tgts_combined = combined
            combined+=present_absent_separator_token
            absent_tgts_combined = ""
            for tgt in absent_tgts:
                combined += ' '.join(tgt)
                absent_tgts_combined += ' '.join(tgt)
                combined += separator_token
                absent_tgts_combined += separator_token
            #combined += [target_end_token]
            #print(combined)
                #print(combined)
                #print(title)
            json_dict['combined_target_sequence_for_BART'] = combined
            json_dict['combined_present_target_sequence_for_BART'] = present_tgts_combined
            json_dict['combined_absent_target_sequence_for_BART'] = absent_tgts_combined
            output_json.write(json.dumps(json_dict) + '\n')
    #return max_length



def create_combined_src_sequence_BART(json_location, output_location, max_len):
    with open(json_location, 'r', encoding="utf8", ) as input_json, open(output_location, 'w') as output_json:
        for json_line in input_json:
            json_dict = json.loads(json_line)
            tokenized_src = json_dict['tokenized']['src']
            if len(tokenized_src) > max_len:
                tokenized_src = tokenized_src[0:max_len]
            src = ' '.join(tokenized_src)
            json_dict['combined_src_sequence_for_BART'] = src
            output_json.write(json.dumps(json_dict) + '\n')



def encode_text_all(input_location, output_location_src, output_location_tgt, output_location_present_tgt, output_location_absent_tgt, tokenizer):
    encoded_src_list = []
    encoded_tgt_list = []
    encoded_present_tgt_list = []
    encoded_absent_tgt_list = []
    src_max_length = -1
    tgt_max_length = -1
    present_tgt_max_length = -1
    absent_tgt_max_length = -1
    with open(input_location, 'r', encoding="utf8", ) as input_json:
        for json_line in input_json:
            json_dict = json.loads(json_line)
            src = json_dict['combined_src_sequence_for_BART']
            tgt = json_dict['combined_target_sequence_for_BART']
           # print(json_dict)
           # quit()
            presnent_tgt = json_dict['combined_present_target_sequence_for_BART']
            absent_tgt = json_dict['combined_absent_target_sequence_for_BART']

            encoded_src = tokenizer.encode(src, add_special_tokens = False)
            encoded_tgt = tokenizer.encode(tgt, add_special_tokens = True)
            encoded_present_tgt = tokenizer.encode(presnent_tgt, add_special_tokens=True)
            encoded_absent_tgt = tokenizer.encode(absent_tgt, add_special_tokens=True)

            if len(encoded_src) > src_max_length:
                src_max_length = len(encoded_src)
            if len(encoded_tgt) > tgt_max_length:
                tgt_max_length = len(encoded_tgt)
            if len(encoded_present_tgt) > present_tgt_max_length:
                present_tgt_max_length = len(encoded_present_tgt)
            if len(encoded_absent_tgt) > absent_tgt_max_length:
                absent_tgt_max_length = len(encoded_absent_tgt)
                #print(json_dict, absent_tgt_max_length)
            #if len(encoded_absent_tgt) >= 44:
             #   print(json_dict)
            encoded_src_list.append(encoded_src)
            encoded_tgt_list.append(encoded_tgt)
            encoded_present_tgt_list.append(encoded_present_tgt)
            encoded_absent_tgt_list.append(encoded_absent_tgt)

    save_data(encoded_src_list, output_location_src)
    save_data(encoded_tgt_list, output_location_tgt)
    save_data(encoded_present_tgt_list, output_location_present_tgt)
    save_data(encoded_absent_tgt_list, output_location_absent_tgt)

    return src_max_length, tgt_max_length, present_tgt_max_length, absent_tgt_max_length

def pad_text(encoded_loc, output_location, max_len):
    encoded_list = load_data(encoded_loc)

    padded_list = []
    for text in encoded_list:
        if len(text) > max_len:
            text = text[0:max_len]
        else:
            while len(text) < max_len:
                text.append(word_to_idx[pad_token])
        padded_list.append(text)
    save_data(padded_list, output_location)
    del padded_list
    del encoded_list

def pad_text_all(src_encoded_location, tgt_encoded_location, present_tgt_encoded_location, absent_tgt_encoded_location, src_output_location, tgt_output_location, present_tgt_output_location, absent_tgt_output_location, src_max_length, tgt_max_length, present_tgt_max_length, absent_tgt_max_length):

    pad_text(src_encoded_location,src_output_location,src_max_length)
    pad_text(tgt_encoded_location, tgt_output_location, tgt_max_length)
    pad_text(present_tgt_encoded_location, present_tgt_output_location, present_tgt_max_length)
    pad_text(absent_tgt_encoded_location, absent_tgt_output_location, absent_tgt_max_length)

    '''src_padded_list = []
    tgt_padded_list = []
    present_tgt_padded_list = []
    absent_tgt_padded_list = []

    src_encoded_list = load_data(src_encoded_location)
    tgt_encoded_list = load_data(tgt_encoded_location)
    present_tgt_encoded_list = load_data(present_tgt_encoded_location)
    absent_tgt_encoded_list = load_data(absent_tgt_encoded_location)

    for src in src_encoded_list:
        if len(src) > src_max_length:
            src = src[0:src_max_length]
        else:
            while len(src) < src_max_length:
                src.append(word_to_idx[pad_token])
        src_padded_list.append(src)

    for tgt in tgt_encoded_list:
        if len(tgt) > present_tgt_max_length:
            tgt = tgt[0:present_tgt_max_length]
        else:
            while len(tgt) < tgt_max_length:
                tgt.append(word_to_idx[pad_token])
        tgt_padded_list.append(tgt)

    for tgt in present_tgt_encoded_list:
        if len(tgt) > present_tgt_max_length:
            tgt = tgt[0:present_tgt_max_length]
        else:
            while len(tgt) < present_tgt_max_length:
                tgt.append(word_to_idx[pad_token])
        present_tgt_padded_list.append(tgt)

    for tgt in absent_tgt_encoded_list:
        if len(tgt) > absent_tgt_max_length:
            tgt = tgt[0:absent_tgt_max_length]
        else:
            while len(tgt) < absent_tgt_max_length:
                tgt.append(word_to_idx[pad_token])
        absent_tgt_padded_list.append(tgt)

    save_data(src_padded_list, src_output_location)
    save_data(tgt_padded_list, tgt_output_location)
    save_data(present_tgt_padded_list, present_tgt_output_location)
    save_data(absent_tgt_padded_list, absent_tgt_output_location)'''

def create_attention_masks(src_padded_location, tgt_padded_location,present_tgt_padded_location, absent_tgt_padded_location, src_attention_mask_location, tgt_attention_mask_location, present_tgt_attention_mask_location, absent_tgt_attention_mask_location):
    src_padded_list = load_data(src_padded_location)
    tgt_padded_list = load_data(tgt_padded_location)
    present_tgt_padded_list = load_data(present_tgt_padded_location)
    absent_tgt_padded_list = load_data(absent_tgt_padded_location)

    src_attention_mask_list = []
    tgt_attention_mask_list = []
    present_tgt_attention_mask_list = []
    absent_tgt_attention_mask_list = []

    for src in src_padded_list:
        src_attention_mask = [0 if x == word_to_idx[pad_token] else 1 for x in src]
        src_attention_mask_list.append(src_attention_mask)

    for tgt in tgt_padded_list:
        tgt_attention_mask = [0 if x == word_to_idx[pad_token] else 1 for x in tgt]
        tgt_attention_mask_list.append(tgt_attention_mask)

    for tgt in present_tgt_padded_list:
        tgt_attention_mask = [0 if x == word_to_idx[pad_token] else 1 for x in tgt]
        present_tgt_attention_mask_list.append(tgt_attention_mask)

    for tgt in absent_tgt_padded_list:
        tgt_attention_mask = [0 if x == word_to_idx[pad_token] else 1 for x in tgt]
        absent_tgt_attention_mask_list.append(tgt_attention_mask)

    save_data(src_attention_mask_list, src_attention_mask_location)
    save_data(tgt_attention_mask_list, tgt_attention_mask_location)
    save_data(present_tgt_attention_mask_list, present_tgt_attention_mask_location)
    save_data(absent_tgt_attention_mask_list, absent_tgt_attention_mask_location)

output_base = "E:\ResearchData\Keyphrase Generation\DataForExperiments_BART\\"
#trainOrTest_all = ['test','valid','train']
trainOrTest_all = ['train_aug']
json_base_dir = 'E:\ResearchData\Keyphrase Generation\data\json\\'
present_max = 8
absent_max = 3
src_max_len = 300
##creating combined sequences
for trainOrTest in trainOrTest_all:
    if trainOrTest == 'test':
        dataset_names = ['inspec', 'krapivin', 'nus', 'semeval', 'kp20k', 'duc']
    else:
        dataset_names = ['kp20k']

    for dataset_name in dataset_names:
        if trainOrTest == 'test':
            input_json_path = os.path.join(json_base_dir, dataset_name, '%s_test_tokenized.json' % dataset_name)
            output_json_path = os.path.join(json_base_dir, dataset_name, '%s_test_tgt_combined_for_BART.json' % dataset_name)
        elif trainOrTest == 'valid':
            input_json_path = os.path.join(json_base_dir, dataset_name, '%s_valid_tokenized.json' % dataset_name)
            output_json_path = os.path.join(json_base_dir, dataset_name, '%s_valid_tgt_combined_for_BART.json' % dataset_name)
        elif trainOrTest == 'train':
            input_json_path = os.path.join(json_base_dir, dataset_name, '%s_train_tokenized.json' % dataset_name)
            output_json_path = os.path.join(json_base_dir, dataset_name, '%s_train_tgt_combined_for_BART.json' % dataset_name)
        else:
            print('aug')
            input_json_path = os.path.join(json_base_dir, dataset_name, '%s_train_augmented_tokenized.json' % dataset_name)
            output_json_path = os.path.join(json_base_dir, dataset_name, '%s_train_augmented_tgt_combined_for_BART.json' % dataset_name)

        create_combined_target_sequence_BART(input_json_path, output_json_path, present_max, absent_max)

for trainOrTest in trainOrTest_all:
    if trainOrTest == 'test':
        dataset_names = ['inspec', 'krapivin', 'nus', 'semeval', 'kp20k', 'duc']
    else:
        dataset_names = ['kp20k']

    for dataset_name in dataset_names:
        if trainOrTest == 'test':
            input_json_path = os.path.join(json_base_dir, dataset_name, '%s_test_tgt_combined_for_BART.json' % dataset_name)
            output_json_path = os.path.join(json_base_dir, dataset_name, '%s_test_src_tgt_combined_for_BART.json' % dataset_name)
        elif trainOrTest == 'valid':
            input_json_path = os.path.join(json_base_dir, dataset_name, '%s_valid_tgt_combined_for_BART.json' % dataset_name)
            output_json_path = os.path.join(json_base_dir, dataset_name, '%s_valid_src_tgt_combined_for_BART.json' % dataset_name)
        elif trainOrTest == 'train':
            input_json_path = os.path.join(json_base_dir, dataset_name, '%s_train_tgt_combined_for_BART.json' % dataset_name)
            output_json_path = os.path.join(json_base_dir, dataset_name, '%s_train_src_tgt_combined_for_BART.json' % dataset_name)
        else:
            input_json_path = os.path.join(json_base_dir, dataset_name, '%s_train_augmented_tgt_combined_for_BART.json' % dataset_name)
            output_json_path = os.path.join(json_base_dir, dataset_name, '%s_train_augmented_src_tgt_combined_for_BART.json' % dataset_name)

        create_combined_src_sequence_BART(input_json_path, output_json_path, src_max_len)

src_max_length = -1
tgt_max_length = -1
present_tgt_max_length = -1
absent_tgt_max_length = -1
##creating encoded sequences
for trainOrTest in trainOrTest_all:
    if trainOrTest == 'test':
        dataset_names = ['inspec', 'krapivin', 'nus', 'semeval', 'kp20k', 'duc']
    else:
        dataset_names = ['kp20k']

    for dataset_name in dataset_names:
        if os.path.isdir(os.path.join(output_base, dataset_name+'\\')) == False:
            os.mkdir(os.path.join(output_base, dataset_name+'\\'))
        if trainOrTest == 'test':
            input_json_path = os.path.join(json_base_dir, dataset_name, '%s_test_src_tgt_combined_for_BART.json' % dataset_name)
            output_location_src = os.path.join(output_base, dataset_name, '%s_test_src_encoded.pkl' % dataset_name)
            output_location_tgt = os.path.join(output_base, dataset_name, '%s_test_tgt_encoded.pkl' % dataset_name)
            output_location_present_tgt = os.path.join(output_base, dataset_name, '%s_test_present_tgt_encoded.pkl' % dataset_name)
            output_location_absent_tgt = os.path.join(output_base, dataset_name, '%s_test_absent_tgt_encoded.pkl' % dataset_name)
        elif trainOrTest == 'valid':
            input_json_path = os.path.join(json_base_dir, dataset_name, '%s_valid_src_tgt_combined_for_BART.json' % dataset_name)
            output_location_src = os.path.join(output_base, dataset_name, '%s_valid_src_encoded.pkl' % dataset_name)
            output_location_tgt = os.path.join(output_base, dataset_name, '%s_valid_tgt_encoded.pkl' % dataset_name)

            output_location_present_tgt = os.path.join(output_base, dataset_name,
                                                       '%s_valid_present_tgt_encoded.pkl' % dataset_name)
            output_location_absent_tgt = os.path.join(output_base, dataset_name,
                                                      '%s_valid_absent_tgt_encoded.pkl' % dataset_name)
        elif trainOrTest == 'train':
            input_json_path = os.path.join(json_base_dir, dataset_name, '%s_train_src_tgt_combined_for_BART.json' % dataset_name)
            output_location_src = os.path.join(output_base, dataset_name, '%s_train_src_encoded.pkl' % dataset_name)
            output_location_tgt = os.path.join(output_base, dataset_name, '%s_train_tgt_encoded.pkl' % dataset_name)

            output_location_present_tgt = os.path.join(output_base, dataset_name,
                                                       '%s_train_present_tgt_encoded.pkl' % dataset_name)
            output_location_absent_tgt = os.path.join(output_base, dataset_name,
                                                      '%s_train_absent_tgt_encoded.pkl' % dataset_name)

        else:
            input_json_path = os.path.join(json_base_dir, dataset_name, '%s_train_augmented_src_tgt_combined_for_BART.json' % dataset_name)
            output_location_src = os.path.join(output_base, dataset_name, '%s_train_augmented_src_encoded.pkl' % dataset_name)
            output_location_tgt = os.path.join(output_base, dataset_name, '%s_train_augmented_tgt_encoded.pkl' % dataset_name)

            output_location_present_tgt = os.path.join(output_base, dataset_name,
                                                       '%s_train_augmented_present_tgt_encoded.pkl' % dataset_name)
            output_location_absent_tgt = os.path.join(output_base, dataset_name,
                                                      '%s_train_augmented_absent_tgt_encoded.pkl' % dataset_name)

        src_max_dataset,tgt_max_dataset, present_tgt_max_dataset, absent_tgt_max_dataset =  encode_text_all(input_json_path, output_location_src, output_location_tgt, output_location_present_tgt, output_location_absent_tgt, tokenizer)
        #encode_text_all(input_location, output_location_src, output_location_tgt, output_location_present_tgt, output_location_absent_tgt, tokenizer)
        print(dataset_name, 'src max', src_max_dataset, 'tgt max', tgt_max_dataset, 'present tgt max', present_tgt_max_dataset, 'absent tgt max', absent_tgt_max_dataset)

        if src_max_dataset > src_max_length:
            src_max_length = src_max_dataset
        if tgt_max_dataset > tgt_max_length:
            tgt_max_length = tgt_max_dataset

        if present_tgt_max_dataset > present_tgt_max_length:
            present_tgt_max_length = present_tgt_max_dataset
        if absent_tgt_max_dataset > absent_tgt_max_length:
            absent_tgt_max_length = absent_tgt_max_dataset
absent_tgt_max_length = 44 # setting manually to avoid noise. only 20 samples among 500000 has greater length than 44
##creating padded sequences
for trainOrTest in trainOrTest_all:
    if trainOrTest == 'test':
        dataset_names = ['inspec', 'krapivin', 'nus', 'semeval', 'kp20k', 'duc']
    else:
        dataset_names = ['kp20k']

    for dataset_name in dataset_names:
        if os.path.isdir(os.path.join(output_base, dataset_name+'\\')) == False:
            os.mkdir(os.path.join(output_base, dataset_name+'\\'))
        if trainOrTest == 'test':
            src_encoded_location = os.path.join(output_base, dataset_name, '%s_test_src_encoded.pkl' % dataset_name)
            tgt_encoded_location = os.path.join(output_base, dataset_name, '%s_test_tgt_encoded.pkl' % dataset_name)
            present_tgt_encoded_location = os.path.join(output_base, dataset_name, '%s_test_present_tgt_encoded.pkl' % dataset_name)
            absent_tgt_encoded_location = os.path.join(output_base, dataset_name, '%s_test_absent_tgt_encoded.pkl' % dataset_name)

            src_encoded_padded_location = os.path.join(output_base, dataset_name, '%s_test_src_encoded_padded.pkl' % dataset_name)
            tgt_encoded_padded_location = os.path.join(output_base, dataset_name, '%s_test_tgt_encoded_padded.pkl' % dataset_name)
            present_tgt_encoded_padded_location = os.path.join(output_base, dataset_name,
                                                       '%s_test_present_tgt_encoded_padded.pkl' % dataset_name)
            absent_tgt_encoded_padded_location = os.path.join(output_base, dataset_name,
                                                       '%s_test_absent_tgt_encoded_padded.pkl' % dataset_name)
        elif trainOrTest == 'valid':
            src_encoded_location = os.path.join(output_base, dataset_name, '%s_valid_src_encoded.pkl' % dataset_name)
            tgt_encoded_location = os.path.join(output_base, dataset_name, '%s_valid_tgt_encoded.pkl' % dataset_name)
            present_tgt_encoded_location = os.path.join(output_base, dataset_name,
                                                        '%s_valid_present_tgt_encoded.pkl' % dataset_name)
            absent_tgt_encoded_location = os.path.join(output_base, dataset_name,
                                                       '%s_valid_absent_tgt_encoded.pkl' % dataset_name)

            src_encoded_padded_location = os.path.join(output_base, dataset_name,
                                                       '%s_valid_src_encoded_padded.pkl' % dataset_name)
            tgt_encoded_padded_location = os.path.join(output_base, dataset_name,
                                                       '%s_valid_tgt_encoded_padded.pkl' % dataset_name)
            present_tgt_encoded_padded_location = os.path.join(output_base, dataset_name,
                                                               '%s_valid_present_tgt_encoded_padded.pkl' % dataset_name)
            absent_tgt_encoded_padded_location = os.path.join(output_base, dataset_name,
                                                              '%s_valid_absent_tgt_encoded_padded.pkl' % dataset_name)
        elif trainOrTest == 'train':
            src_encoded_location = os.path.join(output_base, dataset_name, '%s_train_src_encoded.pkl' % dataset_name)
            tgt_encoded_location = os.path.join(output_base, dataset_name, '%s_train_tgt_encoded.pkl' % dataset_name)
            present_tgt_encoded_location = os.path.join(output_base, dataset_name,
                                                        '%s_train_present_tgt_encoded.pkl' % dataset_name)
            absent_tgt_encoded_location = os.path.join(output_base, dataset_name,
                                                       '%s_train_absent_tgt_encoded.pkl' % dataset_name)

            src_encoded_padded_location = os.path.join(output_base, dataset_name,
                                                       '%s_train_src_encoded_padded.pkl' % dataset_name)
            tgt_encoded_padded_location = os.path.join(output_base, dataset_name,
                                                       '%s_train_tgt_encoded_padded.pkl' % dataset_name)
            present_tgt_encoded_padded_location = os.path.join(output_base, dataset_name,
                                                               '%s_train_present_tgt_encoded_padded.pkl' % dataset_name)
            absent_tgt_encoded_padded_location = os.path.join(output_base, dataset_name,
                                                              '%s_train_absent_tgt_encoded_padded.pkl' % dataset_name)

        else:
            src_encoded_location = os.path.join(output_base, dataset_name, '%s_train_augmented_src_encoded.pkl' % dataset_name)
            tgt_encoded_location = os.path.join(output_base, dataset_name, '%s_train_augmented_tgt_encoded.pkl' % dataset_name)
            present_tgt_encoded_location = os.path.join(output_base, dataset_name,
                                                        '%s_train_augmented_present_tgt_encoded.pkl' % dataset_name)
            absent_tgt_encoded_location = os.path.join(output_base, dataset_name,
                                                       '%s_train_augmented_absent_tgt_encoded.pkl' % dataset_name)

            src_encoded_padded_location = os.path.join(output_base, dataset_name,
                                                       '%s_train_augmented_src_encoded_padded.pkl' % dataset_name)
            tgt_encoded_padded_location = os.path.join(output_base, dataset_name,
                                                       '%s_train_augmented_tgt_encoded_padded.pkl' % dataset_name)
            present_tgt_encoded_padded_location = os.path.join(output_base, dataset_name,
                                                               '%s_train_augmented_present_tgt_encoded_padded.pkl' % dataset_name)
            absent_tgt_encoded_padded_location = os.path.join(output_base, dataset_name,
                                                              '%s_train_augmented_absent_tgt_encoded_padded.pkl' % dataset_name)

        pad_text_all(src_encoded_location, tgt_encoded_location,present_tgt_encoded_location,absent_tgt_encoded_location, src_encoded_padded_location, tgt_encoded_padded_location, present_tgt_encoded_padded_location, absent_tgt_encoded_padded_location,src_max_length, tgt_max_length, present_tgt_max_length, absent_tgt_max_length)

        #pad_text_all(src_encoded_location, tgt_encoded_location, present_tgt_encoded_location, absent_tgt_encoded_location, src_output_location, tgt_output_location, present_tgt_output_location, absent_tgt_output_location, src_max_length, tgt_max_length, present_tgt_max_length, absent_tgt_max_length):


##creating attention masks
for trainOrTest in trainOrTest_all:
    if trainOrTest == 'test':
        dataset_names = ['inspec', 'krapivin', 'nus', 'semeval', 'kp20k', 'duc']
    else:
        dataset_names = ['kp20k']

    for dataset_name in dataset_names:
        if os.path.isdir(os.path.join(output_base, dataset_name+'\\')) == False:
            os.mkdir(os.path.join(output_base, dataset_name+'\\'))
        if trainOrTest == 'test':
            src_encoded_padded_location = os.path.join(output_base, dataset_name, '%s_test_src_encoded_padded.pkl' % dataset_name)
            tgt_encoded_padded_location = os.path.join(output_base, dataset_name, '%s_test_tgt_encoded_padded.pkl' % dataset_name)

            present_tgt_encoded_padded_location = os.path.join(output_base, dataset_name,
                                                               '%s_test_present_tgt_encoded_padded.pkl' % dataset_name)
            absent_tgt_encoded_padded_location = os.path.join(output_base, dataset_name,
                                                              '%s_test_absent_tgt_encoded_padded.pkl' % dataset_name)

            src_attention_mask_location = os.path.join(output_base, dataset_name,
                                                       '%s_test_src_attention_masks.pkl' % dataset_name)
            tgt_attention_mask_location = os.path.join(output_base, dataset_name,
                                                       '%s_test_tgt_attention_masks.pkl' % dataset_name)
            present_tgt_attention_mask_location = os.path.join(output_base, dataset_name,
                                                       '%s_test_present_tgt_attention_masks.pkl' % dataset_name)
            absent_tgt_attention_mask_location = os.path.join(output_base, dataset_name,
                                                       '%s_test_absent_tgt_attention_masks.pkl' % dataset_name)
        elif trainOrTest == 'valid':

            src_encoded_padded_location = os.path.join(output_base, dataset_name,
                                                       '%s_valid_src_encoded_padded.pkl' % dataset_name)
            tgt_encoded_padded_location = os.path.join(output_base, dataset_name,
                                                       '%s_valid_tgt_encoded_padded.pkl' % dataset_name)
            present_tgt_encoded_padded_location = os.path.join(output_base, dataset_name,
                                                               '%s_valid_present_tgt_encoded_padded.pkl' % dataset_name)
            absent_tgt_encoded_padded_location = os.path.join(output_base, dataset_name,
                                                              '%s_valid_absent_tgt_encoded_padded.pkl' % dataset_name)

            src_attention_mask_location = os.path.join(output_base, dataset_name,
                                                       '%s_valid_src_attention_masks.pkl' % dataset_name)
            tgt_attention_mask_location = os.path.join(output_base, dataset_name,
                                                       '%s_valid_tgt_attention_masks.pkl' % dataset_name)
            present_tgt_attention_mask_location = os.path.join(output_base, dataset_name,
                                                               '%s_valid_present_tgt_attention_masks.pkl' % dataset_name)
            absent_tgt_attention_mask_location = os.path.join(output_base, dataset_name,
                                                              '%s_valid_absent_tgt_attention_masks.pkl' % dataset_name)
        elif trainOrTest == 'train':
            src_encoded_padded_location = os.path.join(output_base, dataset_name,
                                                       '%s_train_src_encoded_padded.pkl' % dataset_name)
            tgt_encoded_padded_location = os.path.join(output_base, dataset_name,
                                                       '%s_train_tgt_encoded_padded.pkl' % dataset_name)

            present_tgt_encoded_padded_location = os.path.join(output_base, dataset_name,
                                                               '%s_train_present_tgt_encoded_padded.pkl' % dataset_name)
            absent_tgt_encoded_padded_location = os.path.join(output_base, dataset_name,
                                                              '%s_train_absent_tgt_encoded_padded.pkl' % dataset_name)

            src_attention_mask_location = os.path.join(output_base, dataset_name,
                                                       '%s_train_src_attention_masks.pkl' % dataset_name)
            tgt_attention_mask_location = os.path.join(output_base, dataset_name,
                                                       '%s_train_tgt_attention_masks.pkl' % dataset_name)
            present_tgt_attention_mask_location = os.path.join(output_base, dataset_name,
                                                               '%s_train_present_tgt_attention_masks.pkl' % dataset_name)
            absent_tgt_attention_mask_location = os.path.join(output_base, dataset_name,
                                                              '%s_train_absent_tgt_attention_masks.pkl' % dataset_name)

        else:
            src_encoded_padded_location = os.path.join(output_base, dataset_name,
                                                       '%s_train_augmented_src_encoded_padded.pkl' % dataset_name)
            tgt_encoded_padded_location = os.path.join(output_base, dataset_name,
                                                       '%s_train_augmented_tgt_encoded_padded.pkl' % dataset_name)

            present_tgt_encoded_padded_location = os.path.join(output_base, dataset_name,
                                                               '%s_train_augmented_present_tgt_encoded_padded.pkl' % dataset_name)
            absent_tgt_encoded_padded_location = os.path.join(output_base, dataset_name,
                                                              '%s_train_augmented_absent_tgt_encoded_padded.pkl' % dataset_name)

            src_attention_mask_location = os.path.join(output_base, dataset_name,
                                                       '%s_train_augmented_src_attention_masks.pkl' % dataset_name)
            tgt_attention_mask_location = os.path.join(output_base, dataset_name,
                                                       '%s_train_augmented_tgt_attention_masks.pkl' % dataset_name)
            present_tgt_attention_mask_location = os.path.join(output_base, dataset_name,
                                                               '%s_train_augmented_present_tgt_attention_masks.pkl' % dataset_name)
            absent_tgt_attention_mask_location = os.path.join(output_base, dataset_name,
                                                              '%s_train_augmented_absent_tgt_attention_masks.pkl' % dataset_name)

        create_attention_masks(src_encoded_padded_location, tgt_encoded_padded_location,present_tgt_encoded_padded_location,absent_tgt_encoded_padded_location, src_attention_mask_location, tgt_attention_mask_location, present_tgt_attention_mask_location, absent_tgt_attention_mask_location)
        #create_attention_masks(src_padded_location, tgt_padded_location,present_tgt_padded_location, absent_tgt_padded_location, src_attention_mask_location, tgt_attention_mask_location, present_tgt_attention_mask_location, absent_tgt_attention_mask_location):


