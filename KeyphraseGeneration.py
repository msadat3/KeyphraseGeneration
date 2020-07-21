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
from queue import PriorityQueue
import operator

####Load data
base = "E:\ResearchData\Keyphrase Generation\DataForExperiments\\"

X_train = load_data(base+"kp20k\\kp20k_train_src_numeric.pkl")
X_test = load_data(base+"kp20k\\kp20k_test_src_numeric.pkl")
X_valid = load_data(base+"kp20k\\kp20k_valid_src_numeric.pkl")

X_train_lengths = load_data(base+"kp20k\\kp20k_train_src_length.pkl")
X_test_lengths = load_data(base+"kp20k\\kp20k_test_src_length.pkl")
X_valid_lengths = load_data(base+"kp20k\\kp20k_valid_src_length.pkl")

y_train_lengths = load_data(base+"kp20k\\kp20k_train_tgt_length.pkl")
y_test_lengths = load_data(base+"kp20k\\kp20k_test_tgt_length.pkl")
y_valid_lengths = load_data(base+"kp20k\\kp20k_valid_tgt_length.pkl")

y_train = load_data(base+"kp20k\\kp20k_train_tgt_numeric.pkl")
y_test = load_data(base+"kp20k\\kp20k_test_tgt_numeric.pkl")
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
max_output_length = len(y_train[0])
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
            #print('out, tgt', output.shape, tgt.shape)
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
    for i in range(decoded_batch.shape[0]):
        translated = translate(decoded_batch[i], idx_to_word)
        translate_batch.append(translated)
    return translated_batch
def create_output_seq(translated_output):
    output_seq = []
    i = 0
    while translated_output[i]!= '</s>':
        i+=1
        current_token = translated_output[i]
        keyphrase = []
        while current_token != '<sep>':
            keyphrase.append(current_token)
            i+=1
            current_token = translated_output[i]
        output_seq.append(keyphrase)
    return output_seq
def create_output_seq_batch(translated_batch):
    output_seq_batch = [create_output_seq(x) for x in translated_batch]
    return output_seq_batch

def calculate_f1_at_k(gold, predicted, k):
    num_predicted = len(predicted)
    stemmed_gold = [stem_word_list(x) for x in gold]
    predicted = predicted[0,k]
    stemmed_predicted = [stem_word_list(x) for x in predicted]

    correct = [1 if p in g else 0 for p, g in zip(stemmed_predicted, stemmed_gold)]

    num_correct_at_k = sum(correct)
    num_target = len(stemmed_gold)


    precision_at_k = num_correct_at_k / min(k, num_predicted)
    recall = num_correct_at_k/num_target

    f1_at_k = (2 * precision_at_k * recall) / (precision_at_k + recall)

    return precision_at_k, recall, f1_at_k






train_data_loader = create_data_loaders(X_train, X_train_lengths, y_train, y_train_lengths, batch_size, device)
validation_data_loader = create_data_loaders(X_valid, X_valid_lengths, y_valid, y_valid_lengths, 10, device, data_type = 'eval')
location = "E:\ResearchData\Keyphrase Generation\Model\Model.pt"

#save_data(train_data_loader, base+'trainloader.pkl')
#save_data(validation_data_loader, base+'validationloader.pkl')
train_model(train_data_loader, validation_data_loader, location)





