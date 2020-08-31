import json
from Utils import *
import os
from nltk.probability import FreqDist

'''target_start_token = "<s>"
target_end_token = "</s>"
separator_token = "<sep>"
present_absent_separator_token = "<eofpr>"
pad_token = "<pad>"'''


target_start_token = ""
target_end_token = ""
separator_token = ";"
present_absent_separator_token = "."
pad_token = "<pad>"


present_max = 8
absent_max = 3

def create_combined_target_sequence(json_location, output_location, present_max, absent_max):
    max_length = -1
    with open(json_location, 'r', encoding="utf8", ) as input_json,  open(output_location, 'w') as output_json:
        for json_line in input_json:
            json_dict = json.loads(json_line)
            present_tgts = json_dict['tokenized']['present_tgt']
            absent_tgts = json_dict['tokenized']['absent_tgt']
            title = json_dict['title']
            if len(present_tgts) > present_max:
                present_tgts = present_tgts[0:present_max]
            if len(absent_tgts) > absent_max:
                absent_tgts = absent_tgts[0:absent_max]
            combined = [target_start_token]
            for tgt in present_tgts:
                combined += tgt
                combined+=[separator_token]
            combined+=[present_absent_separator_token]
            for tgt in absent_tgts:
                combined += tgt
                combined += [separator_token]
            combined += [target_end_token]
            #print(combined)
            if max_length < len(combined):
                max_length = len(combined)
                print(combined)
                print(title)
            json_dict['tokenized']['combined_target_sequence'] = combined
            output_json.write(json.dumps(json_dict) + '\n')
    return max_length



def select_vocab(training_src_location,training_tgt_location, output_location):
    all_words = []
    src_array = load_data(training_src_location)
    tgt_array = load_data(training_tgt_location)

    for i in range(len(src_array)):
        all_words += src_array[i]
        all_words += tgt_array[i]
    fdist = FreqDist(all_words)
    vocab = fdist.most_common(50005)
    save_data(vocab,output_location)
    return vocab

def create_vocab_dictionaries(vocab_location,word_to_idx_location, idx_to_word_location):
    vocab = load_data(vocab_location)

    word_to_idx = {}

    word_to_idx[target_start_token] = 0
    word_to_idx[target_end_token] = 1
    #print(target_end_token, word_to_idx[target_end_token])
    word_to_idx[separator_token] = 2
    word_to_idx[present_absent_separator_token] = 3
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


def create_padded_sequences(data_location, src_output_location, tgt_output_location, src_lengths_output_location, tgt_lengths_output_location, src_max_length, tgt_max_length):
    X = []
    y = []
    X_lengths = []
    y_lengths = []

    with open(data_location, 'r', encoding="utf8", ) as input_json:
        for json_line in input_json:
            json_dict = json.loads(json_line)
            src = json_dict['tokenized']['src']
            tgt = json_dict['tokenized']['combined_target_sequence']
            #print('=====================================', src)

            X_lengths.append(len(src))
            if len(src)>src_max_length:
                src = src[0:src_max_length]
            elif len(src)<src_max_length:
                while len(src) != src_max_length:
                    src.append(pad_token)



            '''if len(tgt)>tgt_max_length:
                tgt = tgt[0:tgt_max_length]
                if tgt[-1] == separator_token:
                    tgt.append(target_end_token)
                else:
                    i = len(tgt) - 1
                    #print(i,len(tgt))
                    while tgt[i] != separator_token:###to get rid of partial keyphrases
                        tgt[i] = pad_token
                        i-=1
                        #print(i, title)
                    tgt[i+1] = target_end_token####adding the end token after the last separator token
                    tgt.append(pad_token)
                   # print(tgt)
            elif len(tgt) == tgt_max_length:
                tgt.append(pad_token)'''
            y_lengths.append(len(tgt))
            if len(tgt)<tgt_max_length:
                while len(tgt) != tgt_max_length:
                    tgt.append(pad_token)

            #print('==============================',src)
            X.append(src)
            y.append(tgt)

    X_lengths = np.asarray(X_lengths)
    X_lengths = np.where(X_lengths < src_max_length, X_lengths, src_max_length)
    y_lengths = np.asarray(y_lengths)
    y_lengths = np.where(y_lengths < tgt_max_length, y_lengths, tgt_max_length)

    save_data(X,src_output_location)
    save_data(y,tgt_output_location)
    save_data(X_lengths, src_lengths_output_location)
    save_data(y_lengths, tgt_lengths_output_location)

def create_numeric_sequence(input_location, output_location, word_to_idx_location):
    text_list = load_data(input_location)
    word_to_idx_dict = load_data(word_to_idx_location)

    numberic_seq = []
    for text in text_list:
        temp = []
        for word in text:
            if word in word_to_idx_dict.keys():
                temp.append(word_to_idx_dict[word])
            else:
                temp.append(word_to_idx_dict['<unk>'])
        numberic_seq.append(temp)

    save_data(numberic_seq, output_location)

def create_numeric_sequence_src(input_location, output_location, extended_vocab_output_location, input_oov_location, word_to_idx_location):
    text_list = load_data(input_location)
    word_to_idx_dict = load_data(word_to_idx_location)

    numberic_seq = []
    numberic_seq_extended_vocab = []
    input_oov_list = []
    for text in text_list:
        temp = []
        temp_extended = []
        input_oov = []
        for word in text:
            if word in word_to_idx_dict.keys():
                temp.append(word_to_idx_dict[word])
                temp_extended.append(word_to_idx_dict[word])
            else:
                temp.append(word_to_idx_dict['<unk>'])
                if word in input_oov:
                    temp_extended.append(len(word_to_idx)+input_oov.index(word))
                else:
                    input_oov.append(word)
                    temp_extended.append(len(word_to_idx) + input_oov.index(word))

        numberic_seq.append(temp)
        numberic_seq_extended_vocab.append(temp_extended)
        input_oov_list.append(input_oov)

    save_data(numberic_seq, output_location)
    save_data(numberic_seq_extended_vocab, extended_vocab_output_location)
    save_data(input_oov_list, input_oov_location)

def create_numeric_sequence_tgt(input_location, output_location, extended_vocab_output_location, input_oov_location, word_to_idx_location):
    text_list = load_data(input_location)
    word_to_idx_dict = load_data(word_to_idx_location)
    input_oov_list = load_data(input_oov_location)

    numberic_seq = []
    numberic_seq_extended_vocab = []

    i = 0
    for text in text_list:
        temp = []
        temp_extended = []
        input_oov = input_oov_list[i]

        for word in text:
            if word in word_to_idx_dict.keys():
                temp.append(word_to_idx_dict[word])
                temp_extended.append(word_to_idx_dict[word])
            else:
                temp.append(word_to_idx_dict['<unk>'])
                if word in input_oov:
                    temp_extended.append(len(word_to_idx)+input_oov.index(word))
                else:
                    temp_extended.append(word_to_idx_dict['<unk>'])

        numberic_seq.append(temp)
        numberic_seq_extended_vocab.append(temp_extended)
        i+=1

    save_data(numberic_seq, output_location)
    save_data(numberic_seq_extended_vocab, extended_vocab_output_location)


keyphrase_max_len = -1
dataset_names = []
trainOrTest_all = ['test','valid','train']
json_base_dir = 'E:\ResearchData\Keyphrase Generation\data\json\\'
for trainOrTest in trainOrTest_all:
    if trainOrTest == 'test':
        dataset_names = ['inspec', 'krapivin', 'nus', 'semeval', 'kp20k', 'duc', 'stackexchange']
    else:
        dataset_names = ['kp20k']

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

        max_temp = create_combined_target_sequence(input_json_path, output_json_path, present_max, absent_max)
        if keyphrase_max_len<max_temp:
            keyphrase_max_len = max_temp
        print(dataset_name,keyphrase_max_len, max_temp)




output_base = "E:\ResearchData\Keyphrase Generation\DataForExperiments_pointer_generator\\"
for trainOrTest in trainOrTest_all:
    if trainOrTest == 'test':
        dataset_names = ['inspec', 'krapivin', 'nus', 'semeval', 'kp20k', 'duc', 'stackexchange']
    else:
        dataset_names = ['kp20k']

    for dataset_name in dataset_names:
        if trainOrTest == 'test':
            input_json_path = os.path.join(json_base_dir, dataset_name, '%s_test_tokenized_targets_combined.json' % dataset_name)
            src_output_path = os.path.join(output_base, dataset_name, '%s_test_src_padded.pkl' % dataset_name)
            tgt_output_path = os.path.join(output_base, dataset_name, '%s_test_tgt_padded.pkl' % dataset_name)

            src_length_output_path = os.path.join(output_base, dataset_name, '%s_test_src_length.pkl' % dataset_name)
            tgt_length_output_path = os.path.join(output_base, dataset_name, '%s_test_tgt_length.pkl' % dataset_name)
        elif trainOrTest == 'valid':
            input_json_path = os.path.join(json_base_dir, dataset_name, '%s_valid_tokenized_targets_combined.json' % dataset_name)
            src_output_path = os.path.join(output_base, dataset_name, '%s_valid_src_padded.pkl' % dataset_name)
            tgt_output_path = os.path.join(output_base, dataset_name, '%s_valid_tgt_padded.pkl' % dataset_name)

            src_length_output_path = os.path.join(output_base, dataset_name, '%s_valid_src_length.pkl' % dataset_name)
            tgt_length_output_path = os.path.join(output_base, dataset_name, '%s_valid_tgt_length.pkl' % dataset_name)
        else:
            input_json_path = os.path.join(json_base_dir, dataset_name, '%s_train_tokenized_targets_combined.json' % dataset_name)
            src_output_path = os.path.join(output_base, dataset_name, '%s_train_src_padded.pkl' % dataset_name)
            tgt_output_path = os.path.join(output_base, dataset_name, '%s_train_tgt_padded.pkl' % dataset_name)

            src_length_output_path = os.path.join(output_base, dataset_name, '%s_train_src_length.pkl' % dataset_name)
            tgt_length_output_path = os.path.join(output_base, dataset_name, '%s_train_tgt_length.pkl' % dataset_name)

        directory = os.path.join(output_base, dataset_name)
        if os.path.isdir(directory) == False:
            os.mkdir(directory)
        create_padded_sequences(input_json_path, src_output_path, tgt_output_path, src_length_output_path, tgt_length_output_path, 300, keyphrase_max_len)
        #create_padded_sequences(data_location, src_output_location, tgt_output_location, src_lengths_output_location, tgt_lengths_output_location, src_max_length, tgt_max_length)



#kp_20K_test_location = "E:\ResearchData\Keyphrase Generation\data\json\kp20k\\kp20k_test_tokenized_targets_combined.json"
#create_padded_sequences(kp_20K_test_location, "E:\ResearchData\Keyphrase Generation\DataForExperiments\\kp20k_test_src_padded.pkl", "E:\ResearchData\Keyphrase Generation\DataForExperiments\\kp20k_test_tgt_padded.pkl", 300, 10)


#kp_20K_train_location = "E:\ResearchData\Keyphrase Generation\data\json\kp20k\\kp20k_train_tokenized_targets_combined.json"
#create_padded_sequences(kp_20K_train_location, "E:\ResearchData\Keyphrase Generation\DataForExperiments\\kp20k_train_src_padded.pkl", "E:\ResearchData\Keyphrase Generation\DataForExperiments\\kp20k_train_tgt_padded.pkl", 300, 10)
#########################################################################################

vocab_location = output_base+"vocab.pkl"
vocab = select_vocab(output_base+'kp20k\\kp20k_train_src_padded.pkl',output_base+'kp20k\\kp20k_train_tgt_padded.pkl',vocab_location)

word_to_idx_location = output_base+"word_to_idx.pkl"
idx_to_word_location = output_base+"idx_to_word.pkl"
word_to_idx,idx_to_word= create_vocab_dictionaries(vocab_location,word_to_idx_location, idx_to_word_location)



####creating numeric sequences
for trainOrTest in trainOrTest_all:
    if trainOrTest == 'test':
        dataset_names = ['inspec', 'krapivin', 'nus', 'semeval', 'kp20k', 'duc', 'stackexchange']
    else:
        dataset_names = ['kp20k']

    for dataset_name in dataset_names:
        if trainOrTest == 'test':
            src_input_path = os.path.join(output_base, dataset_name, '%s_test_src_padded.pkl' % dataset_name)
            tgt_input_path = os.path.join(output_base, dataset_name, '%s_test_tgt_padded.pkl' % dataset_name)

            src_output_path = os.path.join(output_base, dataset_name, '%s_test_src_numeric.pkl' % dataset_name)
            tgt_output_path = os.path.join(output_base, dataset_name, '%s_test_tgt_numeric.pkl' % dataset_name)

            src_output_extended_vocab_path = os.path.join(output_base, dataset_name, '%s_test_src_numeric_extended_vocab.pkl' % dataset_name)
            tgt_output_extended_vocab_path = os.path.join(output_base, dataset_name, '%s_test_tgt_numeric_extended_vocab.pkl' % dataset_name)

            src_oov_location = os.path.join(output_base, dataset_name, '%s_test_src_oov_vocab.pkl' % dataset_name)

        elif trainOrTest == 'valid':
            src_input_path = os.path.join(output_base, dataset_name, '%s_valid_src_padded.pkl' % dataset_name)
            tgt_input_path = os.path.join(output_base, dataset_name, '%s_valid_tgt_padded.pkl' % dataset_name)

            src_output_path = os.path.join(output_base, dataset_name, '%s_valid_src_numeric.pkl' % dataset_name)
            tgt_output_path = os.path.join(output_base, dataset_name, '%s_valid_tgt_numeric.pkl' % dataset_name)

            src_output_extended_vocab_path = os.path.join(output_base, dataset_name,'%s_valid_src_numeric_extended_vocab.pkl' % dataset_name)
            tgt_output_extended_vocab_path = os.path.join(output_base, dataset_name,'%s_valid_tgt_numeric_extended_vocab.pkl' % dataset_name)

            src_oov_location = os.path.join(output_base, dataset_name, '%s_valid_src_oov_vocab.pkl' % dataset_name)
        else:
            src_input_path = os.path.join(output_base, dataset_name, '%s_train_src_padded.pkl' % dataset_name)
            tgt_input_path = os.path.join(output_base, dataset_name, '%s_train_tgt_padded.pkl' % dataset_name)

            src_output_path = os.path.join(output_base, dataset_name, '%s_train_src_numeric.pkl' % dataset_name)
            tgt_output_path = os.path.join(output_base, dataset_name, '%s_train_tgt_numeric.pkl' % dataset_name)

            src_output_extended_vocab_path = os.path.join(output_base, dataset_name, '%s_train_src_numeric_extended_vocab.pkl' % dataset_name)
            tgt_output_extended_vocab_path = os.path.join(output_base, dataset_name, '%s_train_tgt_numeric_extended_vocab.pkl' % dataset_name)

            src_oov_location = os.path.join(output_base, dataset_name, '%s_train_src_oov_vocab.pkl' % dataset_name)
        #directory = os.path.join(output_base, dataset_name)
        #if os.path.isdir(directory) == False:
         #   os.mkdir(directory)
        #create_numeric_sequence(src_input_path, src_output_path, word_to_idx_location)
        #create_numeric_sequence(tgt_input_path, tgt_output_path, word_to_idx_location)
        create_numeric_sequence_src(src_input_path, src_output_path,src_output_extended_vocab_path, src_oov_location, word_to_idx_location)
        create_numeric_sequence_tgt(tgt_input_path, tgt_output_path, tgt_output_extended_vocab_path, src_oov_location, word_to_idx_location)

        #create_numeric_sequence_src(input_location, output_location, extended_vocab_output_location, input_oov_location, word_to_idx_location)


