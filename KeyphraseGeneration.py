import itertools

import torch
from torch.utils.data import TensorDataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Utils import *
from Models import *
from torch import autograd
import torch
import math
import os
from queue import PriorityQueue
import operator
import json

####Load data
from Utils import load_data

base = "E:\ResearchData\Keyphrase Generation\DataForExperiments\\"

X_train = load_data(base+"kp20k\\kp20k_train_src_numeric.pkl")
#X_test = load_data(base+"kp20k\\kp20k_test_src_numeric.pkl")
X_valid = load_data(base+"kp20k\\kp20k_valid_src_numeric.pkl")

X_train_lengths = load_data(base+"kp20k\\kp20k_train_src_length.pkl")
#X_test_lengths = load_data(base+"kp20k\\kp20k_test_src_length.pkl")
X_valid_lengths = load_data(base+"kp20k\\kp20k_valid_src_length.pkl")

y_train_lengths = load_data(base+"kp20k\\kp20k_train_tgt_length.pkl")
#y_test_lengths = load_data(base+"kp20k\\kp20k_test_tgt_length.pkl")
y_valid_lengths = load_data(base+"kp20k\\kp20k_valid_tgt_length.pkl")

y_train = load_data(base+"kp20k\\kp20k_train_tgt_numeric.pkl")
#y_test = load_data(base+"kp20k\\kp20k_test_tgt_numeric.pkl")
y_valid = load_data(base+"kp20k\\kp20k_valid_tgt_numeric.pkl")

word_to_idx = load_data(base+"word_to_idx.pkl")
idx_to_word = load_data(base+"idx_to_word.pkl")

def create_data_loaders(X, X_length, y, y_length, batch_size, device, data_type = 'train'):
    X = torch.tensor(X, dtype=torch.long, device=torch.device(device))
    X_length = torch.tensor(X_length, dtype=torch.long, device=torch.device(device))
    y = torch.tensor(y, dtype=torch.long, device=torch.device(device))
    y_length = torch.tensor(y_length, dtype=torch.long, device=torch.device(device))

    data = TensorDataset(X, X_length, y, y_length)
    if data_type != 'train':
        #data = sorted(data, key=lambda x: x[3], reverse=True)#sorting by target length in descending order
    #return data
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    else:
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    return data_loader

#######parameters
vocab_size = len(word_to_idx)
embedding_size = 100
hidden_size = 150
batch_size = 10
lr = 0.001
num_epochs = 20
pad_idx = word_to_idx["<pad>"]
eos_idx = word_to_idx["</s>"]
sos_idx = word_to_idx["<s>"]
#max_output_length = len(y_train[0])
max_output_length = 56
report_every = 100
validation_every = 10000
device = 'cuda'
abc = None

def train_model(train_data_loader, validation_data_loader, location):

    #vocab_size, embedding_size, hidden_size, pad_idx, eos_idx, sos_idx, max_output_length, device

    model = Seq2Seq(vocab_size, embedding_size, hidden_size, pad_idx, eos_idx, sos_idx, max_output_length, device)

    if device == 'cuda':
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss(ignore_index = pad_idx)

    prev_validation_perplexity = 99999999999999.0
    patience = 5
    not_improving_checkpoints = 0
    train_stop_flag = False
    for epoch in range(num_epochs):
        model.train()
        i = 0
        for src, src_length, tgt, tgt_length in train_data_loader:
            optimizer.zero_grad()
            #print(src_length)
            #(self, input_seq, input_lengths, target_seq, teacher_forcing_ratio = 0.5)
            output = model(input_seq=src, input_lengths=src_length, target_seq=tgt)

            #tgt = tgt[:,1:]
            print('out, tgt', output.shape, tgt.shape)
            loss = criterion(output, tgt)
            with autograd.detect_anomaly():
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
            i+=1
            if (i+1)%report_every == 0:
                print('Epoch', epoch, 'step', i, 'loss', loss.item())
            if (i+1)%validation_every == 0:
                model.eval()
                with torch.no_grad():
                    validation_loss = 0
                    batch_count = 0
                    for val_src, val_src_length, val_tgt, val_tgt_length in validation_data_loader:
                        batch_count+=1
                        val_output = model(input_seq=val_src, input_lengths=val_src_length, target_seq=val_tgt)
                        val_loss_batch = criterion(val_output, val_tgt)

                        validation_loss += val_loss_batch.item()
                        print(validation_loss)
                    validation_perplexity = math.exp(validation_loss/batch_count)
                    print('perp', validation_perplexity)
                    if validation_perplexity < prev_validation_perplexity:
                        print("Validation perplexity improved from ", prev_validation_perplexity, " to ", validation_perplexity)
                        torch.save(model.state_dict(), location)
                        prev_validation_perplexity = validation_perplexity
                        not_improving_checkpoints = 0
                    else:
                        print("Validation perplexity did not improve.")
                        not_improving_checkpoints+=1
                    #return abc
                    #print(abc)
                model.train()
                #quit()
            if not_improving_checkpoints == patience:
                print("Not improving for ", patience, " checkpoints. Sopping training.")
                train_stop_flag = True
                break
        if train_stop_flag == True:
            break

def translate(decoded_seq, idx_to_word):
    translated = [idx_to_word[idx] for idx in decoded_seq]
    return translated
def translate_batch(decoded_batch, idx_to_word):
    translated_batch = []
    for i in range(len(decoded_batch)):
        translated = translate(decoded_batch[i], idx_to_word)
        translated_batch.append(translated)
    return translated_batch
def create_output_seq(translated_output):
    #print(translated_output)
    output_seq = []
    i = 0
    while translated_output[i]!= '</s>' and i+1<len(translated_output):
        i+=1
        #print(i)
        current_token = translated_output[i]
        keyphrase = []
        while current_token != '<sep>' and i+1<len(translated_output):
            if current_token!= '<eofpr>':
                keyphrase.append(current_token)
            i+=1
            current_token = translated_output[i]
        if len(keyphrase)>0 and keyphrase not in output_seq:
            output_seq.append(keyphrase)
    #print(output_seq)
    return output_seq
def create_output_seq_batch(translated_batch):
    output_seq_batch = [create_output_seq(x) for x in translated_batch]
    return output_seq_batch

def calculate_f1_at_k_single(gold, predicted, k):
    #print(gold)
    #print(predicted)
    #if len(predicted) > k:
    #    print(predicted)
    #print(len(predicted))
    num_predicted = len(predicted)
    stemmed_gold = [stem_word_list(x) for x in gold]
    predicted = predicted[0:k]
    stemmed_predicted = [stem_word_list(x) for x in predicted]

   # print(stemmed_gold)
    #print(stemmed_predicted)

    correct = 0
    for pred in stemmed_predicted:
        if pred in stemmed_gold:
            correct+=1
    #print(correct)

    num_correct_at_k = correct
    num_target = len(stemmed_gold)



    if num_correct_at_k == 0 or num_predicted == 0:
        precision_at_k = 0
        recall = 0
        f1_at_k = 0
    else:
        precision_at_k = num_correct_at_k / min(k, num_predicted)
        recall = num_correct_at_k / num_target
        f1_at_k = (2 * precision_at_k * recall) / (precision_at_k + recall)
    #print('precision_at_k, recall, f1_at_k', precision_at_k, recall, f1_at_k)
    return precision_at_k, recall, f1_at_k

def calculate_f1_at_m_single(gold, predicted):
    num_predicted = len(predicted)
    stemmed_gold = [stem_word_list(x) for x in gold]
    k = len(predicted)
    stemmed_predicted = [stem_word_list(x) for x in predicted]

    correct = 0
    for pred in stemmed_predicted:
        if pred in stemmed_gold:
            correct += 1
    #print(correct)

    num_correct_at_k = correct
    num_target = len(stemmed_gold)

    if num_correct_at_k == 0 or num_predicted == 0:
        precision_at_k = 0
        recall = 0
        f1_at_k = 0
    else:
        precision_at_k = num_correct_at_k / min(k, num_predicted)
        recall = num_correct_at_k / num_target
        f1_at_k = (2 * precision_at_k * recall) / (precision_at_k + recall)

    #print('precision_at_k, recall, f1_at_k', precision_at_k, recall, f1_at_k)
    return precision_at_k, recall, f1_at_k


def generate_output_sequences(test_data_loader, model_location, output_seq_save_location, decode_style='greedy',  beam_size = 50, max_len= 56 ):
    model = Seq2Seq(vocab_size, embedding_size, hidden_size, pad_idx, eos_idx, sos_idx, max_len, device)

    if device == 'cuda':
        model.cuda()
    model.load_state_dict(torch.load(model_location))
    model.eval()
    output_sequences = []
    with torch.no_grad():
        batch_count = 0

        for test_src, test_src_length, test_tgt, test_tgt_length in test_data_loader:
            batch_count += 1
            test_decoded = model(input_seq=test_src, input_lengths=test_src_length, target_seq=test_tgt, decode_style = decode_style)
            translated_output = translate_batch(test_decoded, idx_to_word)
            #print(translated_output)
            batch_out_sequence = create_output_seq_batch(translated_output)
            output_sequences += batch_out_sequence

    save_data(output_sequences, output_seq_save_location)
    return output_sequences


def evaluate_output(tgt_gold_seq_location, output_sequences_location, K):
    output_sequences = load_data(output_sequences_location)
    #print(output_sequences)

    sample_idx_start = 0
    precisions_at_k_present = []
    recalls_at_k_present = []
    f1s_at_k_present = []

    precisions_at_k_absent = []
    recalls_at_k_absent = []
    f1s_at_k_absent = []

    precisions_at_m_present = []
    recalls_at_m_present = []
    f1s_at_m_present = []

    precisions_at_m_absent = []
    recalls_at_m_absent = []
    f1s_at_m_absent = []
    i = 0
    with open(tgt_gold_seq_location, 'r', encoding="utf8", ) as input_json:
        for json_line in input_json:
            json_dict = json.loads(json_line)
            src = json_dict['tokenized']['src']
            gold_tgt = json_dict['tokenized']['combined_target_sequence']

            gold_tgt = create_output_seq(gold_tgt)
            predicted_tgt = output_sequences[i]

            present_tgt_flags, occurance_positions, _ = if_present_duplicate_phrases(src, gold_tgt)
            present_tgts = [tgt for tgt, present in zip(gold_tgt, present_tgt_flags) if present]
            absent_tgts = [tgt for tgt, present in zip(gold_tgt, present_tgt_flags) if ~present]

            present_pred_flags, occurance_positions, _ = if_present_duplicate_phrases(src, gold_tgt)
            present_pred = [tgt for tgt, present in zip(predicted_tgt, present_tgt_flags) if present]
            absent_pred = [tgt for tgt, present in zip(predicted_tgt, present_tgt_flags) if ~present]

            #print('present pred', present_pred)
            #print('present tgt', present_tgts)
            precision_at_k, recall_at_k, f1_at_k = calculate_f1_at_k_single(present_tgts, present_pred, K)

            precisions_at_k_present.append(precision_at_k)
            recalls_at_k_present.append(recall_at_k)
            f1s_at_k_present.append(f1_at_k)

            precision_at_k, recall_at_k, f1_at_k = calculate_f1_at_k_single(absent_tgts, absent_pred, K)

            precisions_at_k_absent.append(precision_at_k)
            recalls_at_k_absent.append(recall_at_k)
            f1s_at_k_absent.append(f1_at_k)

            ##########at m
            precision_at_m, recall_at_m, f1_at_m = calculate_f1_at_m_single(present_tgts, present_pred)

            precisions_at_m_present.append(precision_at_m)
            recalls_at_m_present.append(recall_at_m)
            f1s_at_m_present.append(f1_at_m)

            precision_at_m, recall_at_m, f1_at_m = calculate_f1_at_m_single(absent_tgts, absent_pred)

            precisions_at_m_absent.append(precision_at_m)
            recalls_at_m_absent.append(recall_at_m)
            f1s_at_m_absent.append(f1_at_m)
            i+=1

    print('Present keyphrases:')
    print('\n')
    print('Scores at K:')
    print('Preicision: ', sum(precisions_at_k_present)/len(precisions_at_k_present))
    print('Recall: ', sum(recalls_at_k_present) / len(recalls_at_k_present))
    print('F1: ', sum(f1s_at_k_present) / len(f1s_at_k_present))
    #print('\n')

    print('Scores at M:')
    print('Preicision: ', sum(precisions_at_m_present) / len(precisions_at_m_present))
    print('Recall: ', sum(recalls_at_m_present) / len(recalls_at_m_present))
    print('F1: ', sum(f1s_at_m_present) / len(f1s_at_m_present))

    #print('\n')
    print('\n')

    print('Absent keyphrases:')
    print('\n')
    print('Scores at K:')
    print('Preicision: ', sum(precisions_at_k_absent) / len(precisions_at_k_absent))
    print('Recall: ', sum(recalls_at_k_absent) / len(recalls_at_k_absent))
    print('F1: ', sum(f1s_at_k_absent) / len(f1s_at_k_absent))


    print('Scores at M:')
    print('Preicision: ', sum(precisions_at_m_absent) / len(precisions_at_m_absent))
    print('Recall: ', sum(recalls_at_m_absent) / len(recalls_at_m_absent))
    print('F1: ', sum(f1s_at_m_absent) / len(f1s_at_m_absent))

train_data_loader = create_data_loaders(X_train, X_train_lengths, y_train, y_train_lengths, batch_size, device)
validation_data_loader = create_data_loaders(X_valid, X_valid_lengths, y_valid, y_valid_lengths, 10, device, data_type = 'eval')
#test_data_loader = create_data_loaders(X_test, X_test_lengths, y_test, y_test_lengths, 10, device, data_type = 'eval')

model_location = "E:\ResearchData\Keyphrase Generation\Model\Model.pt"

#save_data(train_data_loader, base+'trainloader.pkl')
#save_data(validation_data_loader, base+'validationloader.pkl')
train_model(train_data_loader, validation_data_loader, model_location)

#generate_output_sequences(test_data_loader, location, base+'Test\\',decode_style='greedy',beam_size = 50, max_len= 56)

#generate_output_sequences(test_data_loader, model_location, output_seq_save_location, decode_style='greedy',  beam_size = 50, max_len= 56 ):





#evaluate_output(tgt_gold_seq_location, output_sequences_location, K)


dataset_names = ['inspec', 'krapivin', 'nus', 'semeval', 'duc', 'stackexchange', 'kp20k']
#dataset_names = ['krapivin']
#X_test = load_data(base+"kp20k\\kp20k_test_src_numeric.pkl")
#y_test_lengths = load_data(base+"kp20k\\kp20k_test_tgt_length.pkl")
generated_base = "E:\ResearchData\Keyphrase Generation\Generated_outputs\\"
test_gold_base = 'E:\ResearchData\Keyphrase Generation\data\json\\'
for dataset_name in dataset_names:
    print('======================dataset name===================', dataset_name)
    #input_json_path = os.path.join(json_base_dir, dataset_name, '%s_test_tokenized.json' % dataset_name)
    #output_json_path = os.path.join(json_base_dir, dataset_name, '%s_test_tokenized_targets_combined.json' % dataset_name)

    X_test = load_data(os.path.join(base, dataset_name, '%s_test_src_numeric.pkl' % dataset_name))
    X_test_lengths = load_data(os.path.join(base, dataset_name, '%s_test_src_length.pkl' % dataset_name))
    y_test = load_data(os.path.join(base, dataset_name, '%s_test_tgt_numeric.pkl' % dataset_name))
    y_test_lengths = load_data(os.path.join(base, dataset_name, '%s_test_tgt_numeric.pkl' % dataset_name))
    generated_output_location = os.path.join(generated_base, '%s_test_generated_keyphrases.pkl' % dataset_name)
    tgt_gold_seq_location = os.path.join(test_gold_base, dataset_name, '%s_test_tokenized_targets_combined.json' % dataset_name)

    test_data_loader = create_data_loaders(X_test, X_test_lengths, y_test, y_test_lengths, batch_size, device, data_type = 'eval')

    generate_output_sequences(test_data_loader, model_location, generated_output_location, decode_style='greedy', beam_size=50,
                              max_len=56)

    evaluate_output(tgt_gold_seq_location, generated_output_location, 5)





















'''def calculate_f1_at_k_batch(batch_gold, batch_predicted, k):
    precisions_at_k = []
    recalls_at_k = []
    f1s_at_k = []

    for gold, predicted in zip(batch_gold, batch_predicted):
        precision_at_k, recall, f1_at_k = calculate_f1_at_k_single(gold, predicted, k)
        precisions_at_k.append(precision_at_k)
        recalls_at_k.append(recall)
        f1s_at_k.append(f1_at_k)
    return precisions_at_k, recalls_at_k, f1s_at_k


def calculate_f1_at_m_batch(batch_gold, batch_predicted):
    precisions_at_m = []
    recalls_at_m = []
    f1s_at_m = []

    for gold, predicted in zip(batch_gold, batch_predicted):
        m = len(predicted)
        precision_at_m, recall, f1_at_m = calculate_f1_at_k_single(gold, predicted, m)
        precisions_at_m.append(precision_at_m)
        recalls_at_m.append(recall)
        f1s_at_m.append(f1_at_m)
    return precisions_at_m, recalls_at_m, f1s_at_m'''

