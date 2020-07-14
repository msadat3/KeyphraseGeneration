import json
from Utils import *
import os
from nltk.probability import FreqDist

target_start_token = "<s>"
target_end_token = "</s>"
separator_token = "<sep>"
present_absent_separator_token = "<eofpr>"
pad_token = "<pad>"

trainOrTest = 'valid'



def create_combined_target_sequence(json_location, output_location):
    with open(json_location, 'r', encoding="utf8", ) as input_json,  open(output_location, 'w') as output_json:
        for json_line in input_json:
            json_dict = json.loads(json_line)
            present_tgts = json_dict['tokenized']['present_tgt']
            absent_tgts = json_dict['tokenized']['absent_tgt']
            combined = [target_start_token]
            for tgt in present_tgts:
                combined += tgt
                combined+=[separator_token]
            combined+=[present_absent_separator_token]
            for tgt in absent_tgts:
                combined += tgt
                combined += [separator_token]
            combined += [target_end_token]
            print(combined)
            json_dict['tokenized']['combined_target_sequence'] = combined
            output_json.write(json.dumps(json_dict) + '\n')


def select_vocab(training_data_location, output_location):
    all_words = []
    with open(training_data_location, 'r', encoding="utf8", ) as input_json:
        for json_line in input_json:
            json_dict = json.loads(json_line)
            src = json_dict['tokenized']['src']
            tgt = json_dict['tokenized']['combined_target_sequence']

            all_words += src
            all_words += tgt
    fdist = FreqDist(all_words)
    vocab = fdist.most_common(50004)
    save_data(vocab,output_location)
    return vocab

def create_vocab_dictionaries(vocab_location,word_to_idx_location, idx_to_word_location):
    vocab = load_data(vocab_location)

    word_to_idx = {}

    word_to_idx[target_start_token] = 0
    word_to_idx[target_end_token] = 1
    word_to_idx[separator_token] = 2
    word_to_idx[target_end_token] = 3
    word_to_idx['<unk>'] = 4
    word_to_idx[pad_token] = 5
    idx = 6

    for v in vocab:
        word, freq = v
        if word not in word_to_idx.keys():
            word_to_idx[word] = idx
        idx+=1

    save_data(word_to_idx,word_to_idx_location)

    idx_to_word = {v: k for k, v in word_to_idx.items()}

    save_data(idx_to_word,idx_to_word_location)

    return word_to_idx, idx_to_word


def create_padded_sequences(data_location, src_output_location, tgt_output_location, src_max_length, tgt_max_length):
    X = []
    y = []

    with open(data_location, 'r', encoding="utf8", ) as input_json:
        for json_line in input_json:
            json_dict = json.loads(json_line)
            src = json_dict['tokenized']['src']
            tgt = json_dict['tokenized']['combined_target_sequence']

            if len(src)>=src_max_length:
                src = src[0:src_max_length]
            else:
                while len(src) != src_max_length:
                    src.append(pad_token)
            if len(tgt)>=tgt_max_length:
                tgt = tgt[0:tgt_max_length]
                i = len(tgt)
                while tgt[i] != separator_token:###to get rid of partial keyphrases
                    tgt[i] = pad_token
                    i+=1
                tgt[i+1] = target_end_token####adding the end token after the last separator token
                print(tgt)
            else:
                while len(tgt) != tgt_max_length:
                    tgt.append(pad_token)








dataset_names = []
if trainOrTest == 'test':
    dataset_names = ['inspec', 'krapivin', 'nus', 'semeval', 'kp20k', 'duc', 'stackexchange']
else:
    dataset_names = ['kp20k']
json_base_dir = 'E:\ResearchData\Keyphrase Generation\data\json\\'

for dataset_name in dataset_names:
    if trainOrTest == 'test':
        input_json_path = os.path.join(json_base_dir, dataset_name, '%s_test_tokenized.json' % dataset_name)
        output_json_path = os.path.join(json_base_dir, dataset_name, '%s_test_tokenized_targets_combined.json' % dataset_name)
    elif trainOrTest == 'valid':
        input_json_path = os.path.join(json_base_dir, dataset_name, '%s_valid_tokenized.json' % dataset_name)
        output_json_path = os.path.join(json_base_dir, dataset_name, '%s_valid_tokenized_targets_combined.json' % dataset_name)
    else:
        input_json_path = os.path.join(json_base_dir, dataset_name, '%s_train_tokenized.json' % dataset_name)
        output_json_path = os.path.join(json_base_dir, dataset_name, '%s_train_tokenized_targets_combined.json' % dataset_name)

    create_combined_target_sequence(input_json_path, output_json_path)

vocab_location = "E:\ResearchData\Keyphrase Generation\DataForExperiments\\vocab.pkl"
vocab = select_vocab("E:\ResearchData\Keyphrase Generation\data\json\kp20k\kp20k_train_tokenized_targets_combined.json",vocab_location)

word_to_idx_location = "E:\ResearchData\Keyphrase Generation\DataForExperiments\\word_to_idx.pkl"
idx_to_word_location = "E:\ResearchData\Keyphrase Generation\DataForExperiments\\idx_to_word.pkl"
word_to_idx,idx_to_word= create_vocab_dictionaries(vocab_location,word_to_idx_location, idx_to_word_location)