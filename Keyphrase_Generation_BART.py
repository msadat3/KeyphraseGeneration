from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import autograd
import torch
import math
import os
from Utils import *
import sys
import torch.nn as nn
import inspect
import json
#####################old
# sys.path.append('/home/ubuntu/Keyphrase_Generation/KeyphraseGeneration/transformers_master/src/')

from transformers import BartForConditionalGeneration, BartTokenizer
from CustomBARTModels import BartForConditionalGenerationTwoDecoders, BartForConditionalGeneration_custom

# print(inspect.getsource(BartForConditionalGeneration))

base = "/home/ubuntu/Keyphrase_Generation/DataForExperiments_BART/"

model_location = "/home/ubuntu/Keyphrase_Generation/Bart_model_non_custom/Model.pt"
checkpoint_info_location = "/home/ubuntu/Keyphrase_Generation/Bart_model_non_custom/checkpoints.pkl"

generated_base = "/home/ubuntu/Keyphrase_Generation/Generated_outputs_BART/"
test_gold_base = '/home/ubuntu/Keyphrase_Generation/data/json'
print('lalalal')


print('lalalal')

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
word_to_idx = tokenizer.get_vocab()
idx_to_word = {v: k for k, v in word_to_idx.items()}

target_end_token = "</s>"
separator_token = ","
present_absent_separator_token = ";"
pad_token = "<pad>"


def create_data_loaders_BART(X, X_att_mask, y, y_att_mask, batch_size, device, data_type='train'):
    X = torch.tensor(X, dtype=torch.long)
    X_att_mask = torch.tensor(X_att_mask, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)
    y_att_mask = torch.tensor(y_att_mask, dtype=torch.long)

    data = TensorDataset(X, X_att_mask, y, y_att_mask)
    if data_type != 'train':
        # data = sorted(data, key=lambda x: x[3], reverse=True)#sorting by target length in descending order
        # return data
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    else:
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    return data_loader


def create_chunked_data_loader_BART(X_location, X_att_mask_loc, y_location, y_att_mask_location, batch_size, device,
                                    start_idx, end_idx, data_type='train'):
    X = []
    X_att_mask = []
    y = []
    y_att_mask = []
    with open(X_location, 'r', encoding="utf8", ) as src_json_file, open(X_att_mask_loc, 'r',
                                                                         encoding="utf8", ) as src_att_file, open(
            y_location, 'r', encoding="utf8", ) as tgt_json_file, open(y_att_mask_location, 'r',
                                                                       encoding="utf8", ) as tgt_att_file:

        for i in range(start_idx):
            print(i, 'skipping')
            next(src_json_file)
            next(src_att_file)
            next(tgt_json_file)
            next(tgt_att_file)

        total_lines_read = start_idx
        for json_line in src_json_file:
            json_dict = json.loads(json_line)
            X.append(json_dict['padded'])
            if total_lines_read == end_idx:
                break
            total_lines_read += 1

        total_lines_read = start_idx
        for json_line in src_att_file:
            json_dict = json.loads(json_line)
            X_att_mask.append(json_dict['attention_mask'])
            if total_lines_read == end_idx:
                break
            total_lines_read += 1

        total_lines_read = start_idx
        for json_line in tgt_json_file:
            json_dict = json.loads(json_line)
            y.append(json_dict['padded'])
            if total_lines_read == end_idx:
                break
            total_lines_read += 1

        total_lines_read = start_idx
        for json_line in tgt_att_file:
            json_dict = json.loads(json_line)
            y_att_mask.append(json_dict['attention_mask'])
            if total_lines_read == end_idx:
                break
            total_lines_read += 1

    train_loader = create_data_loaders_BART(X, X_att_mask, y, y_att_mask, batch_size, device, data_type='train')

    return train_loader


def create_data_loaders_BART_dual_decoder(X, X_att_mask, y_present, y_present_att_mask, y_absent, y_absent_att_mask,
                                          batch_size, device, data_type='train'):
    X = torch.tensor(X, dtype=torch.long)
    X_att_mask = torch.tensor(X_att_mask, dtype=torch.long)
    y_present = torch.tensor(y_present, dtype=torch.long)
    y_present_att_mask = torch.tensor(y_present_att_mask, dtype=torch.long)
    y_absent = torch.tensor(y_absent, dtype=torch.long)
    y_absent_att_mask = torch.tensor(y_absent_att_mask, dtype=torch.long)

    data = TensorDataset(X, X_att_mask, y_present, y_present_att_mask, y_absent, y_absent_att_mask)
    if data_type != 'train':
        # data = sorted(data, key=lambda x: x[3], reverse=True)#sorting by target length in descending order
        # return data
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    else:
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    return data_loader


batch_size = 2
accumulation_steps = 10 / batch_size
learning_rate = 2e-5
num_epochs = 3

report_every = 1
validation_every = 10000
device = 'cuda'

checkpoint_info = {}


def train_model(train_data_loader, validation_data_loader, model_location):
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    model.load_state_dict(torch.load(model_location))
    # model = BartForConditionalGeneration_custom(pre_trained_model)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('pytorch_total_params', pytorch_total_params)
    # quit()
    if device == 'cuda':
        model.cuda()
    lr = learning_rate

    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=word_to_idx[pad_token])

    prev_validation_perplexity = 3.730775
    patience = 5
    not_improving_checkpoints = 0
    train_stop_flag = False

    for epoch in range(2, num_epochs):
        model.train()
        i = 0
        step_count = 0
        optimizer.zero_grad()
        for X, X_att_mask, y, y_att_mask in train_data_loader:
            if epoch == 2 and step_count <= 30000:
                print(i, step_count)
                if (i + 1) % accumulation_steps == 0:
                    step_count += 1
                if ((i + 1) == -1):
                    model.eval()
                    with torch.no_grad():
                        validation_loss = 0
                        batch_count = 0
                        for val_X, val_X_att_mask, val_y, val_y_att_mask in validation_data_loader:
                            batch_count += 1
                            input_ids = val_X.to(torch.device(device))
                            attention_mask = val_X_att_mask.to(torch.device(device))
                            val_y = val_y.to(torch.device(device))
                            decoder_input_ids = val_y[:, :-1].contiguous()
                            labels = val_y[:, 1:].contiguous()
                            val_y_att_mask = val_y_att_mask.to(torch.device(device))

                            val_outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                                decoder_input_ids=decoder_input_ids,
                                                decoder_attention_mask=val_y_att_mask[:, :-1].contiguous(),
                                                labels=labels)
                            val_logits = torch.transpose(val_outputs[1], 1, 2)
                            val_loss_batch = criterion(val_logits, labels)

                            validation_loss += val_loss_batch.item()
                            print(validation_loss)
                        validation_perplexity = math.exp(validation_loss / batch_count)
                        checkpoint_info['epoch'] = epoch
                        checkpoint_info['step_count'] = step_count
                        checkpoint_info['i'] = i

                        print('perp', validation_perplexity)
                        if validation_perplexity < prev_validation_perplexity:
                            print("Validation perplexity improved from ", prev_validation_perplexity, " to ",
                                  prev_validation_perplexity)
                            torch.save(model.state_dict(), model_location)
                            prev_validation_perplexity = validation_perplexity
                            not_improving_checkpoints = 0
                            checkpoint_info['best_perp'] = validation_perplexity
                        else:
                            print("Validation perplexity did not improve.")
                            not_improving_checkpoints += 1
                        print(checkpoint_info)
                        save_data(checkpoint_info, checkpoint_info_location)
                    model.train()
                if not_improving_checkpoints == patience:
                    print("Not improving for ", patience, " checkpoints. Sopping training.")
                    train_stop_flag = True
                    break
                i += 1
                continue

            input_ids = X.to(torch.device(device))
            attention_mask = X_att_mask.to(torch.device(device))
            y = y.to(torch.device(device))
            y_att_mask = y_att_mask.to(torch.device(device))
            decoder_input_ids = y[:, :-1].contiguous()
            labels = y[:, 1:].contiguous()
            # print(decoder_input_ids.shape)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                            decoder_attention_mask=y_att_mask[:, :-1].contiguous(), labels=labels)
            # print('outputs', len(outputs))
            # print(len(outputs), type(outputs[0]), type(outputs[1]), type(outputs[2]))
            # quit()
            # print(outputs[1].shape)

            # loss = outputs[0]
            logits = torch.transpose(outputs[1], 1, 2)
            loss = criterion(logits, labels) / accumulation_steps
            loss.backward()
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_data_loader):
                optimizer.step()
                optimizer.zero_grad()
                step_count += 1
                print('Epoch', epoch, 'step', step_count, 'loss', loss.item(), 'current validation perplexity',
                      prev_validation_perplexity)

                if step_count % validation_every == 0 or ((i + 1) == len(train_data_loader)) and step_count != 0:
                    model.eval()
                    with torch.no_grad():
                        validation_loss = 0
                        batch_count = 0
                        for val_X, val_X_att_mask, val_y, val_y_att_mask in validation_data_loader:
                            batch_count += 1
                            input_ids = val_X.to(torch.device(device))
                            attention_mask = val_X_att_mask.to(torch.device(device))
                            val_y = val_y.to(torch.device(device))
                            decoder_input_ids = val_y[:, :-1].contiguous()
                            labels = val_y[:, 1:].contiguous()
                            val_y_att_mask = val_y_att_mask.to(torch.device(device))

                            val_outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                                decoder_input_ids=decoder_input_ids,
                                                decoder_attention_mask=val_y_att_mask[:, :-1].contiguous(),
                                                labels=labels)
                            val_logits = torch.transpose(val_outputs[1], 1, 2)
                            val_loss_batch = criterion(val_logits, labels)

                            validation_loss += val_loss_batch.item()
                            # print(validation_loss)
                        validation_perplexity = math.exp(validation_loss / batch_count)
                        print('perp', validation_perplexity)
                        checkpoint_info['epoch'] = epoch
                        checkpoint_info['step_count'] = step_count
                        checkpoint_info['i'] = i

                        if validation_perplexity < prev_validation_perplexity:
                            print("Validation perplexity improved from ", prev_validation_perplexity, " to ",
                                  validation_perplexity)
                            torch.save(model.state_dict(), model_location)
                            prev_validation_perplexity = validation_perplexity
                            not_improving_checkpoints = 0
                            checkpoint_info['best_perp'] = validation_perplexity
                        else:
                            print("Validation perplexity did not improve.")
                            not_improving_checkpoints += 1
                        save_data(checkpoint_info, checkpoint_info_location)
                    model.train()
                if not_improving_checkpoints == patience:
                    print("Not improving for ", patience, " checkpoints. Sopping training.")
                    train_stop_flag = True
                    break
            i += 1
        if train_stop_flag == True:
            break


def train_model_dual_decoders(train_data_loader, validation_data_loader, model_location):
    pre_trained_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    model = BartForConditionalGenerationTwoDecoders(pre_trained_model)

    # model.load_state_dict(torch.load(model_location))
    if device == 'cuda':
        model.cuda()

    lr = learning_rate
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=word_to_idx[pad_token])

    prev_validation_perplexity = 99999999999999.0
    patience = 5
    not_improving_checkpoints = 0
    train_stop_flag = False

    for epoch in range(0, num_epochs):
        model.train()
        i = 0
        step_count = 0
        optimizer.zero_grad()
        for X, X_att_mask, y_present, y_present_att_mask, y_absent, y_absent_att_mask in train_data_loader:
            if epoch == 100 and step_count < 10000:

                if (i + 1) % accumulation_steps == 0:
                    step_count += 1
                    print(step_count)
                if step_count == 10000:
                    model.eval()
                    with torch.no_grad():
                        present_validation_loss = 0
                        absent_validation_loss = 0
                        batch_count = 0
                        for val_X, val_X_att_mask, val_y_present, val_y_present_att_mask, val_y_absent, val_y_absent_att_mask in validation_data_loader:
                            batch_count += 1
                            input_ids = val_X.to(torch.device(device))
                            attention_mask = val_X_att_mask.to(torch.device(device))

                            val_y_present = val_y_present.to(torch.device(device))
                            val_y_present_att_mask = val_y_present_att_mask.to(torch.device(device))
                            val_y_absent = val_y_absent.to(torch.device(device))
                            val_y_absent_att_mask = val_y_absent_att_mask.to(torch.device(device))
                            decoder_input_ids_present = val_y_present[:, :-1].contiguous()
                            decoder_input_ids_absent = val_y_absent[:, :-1].contiguous()
                            labels_present = val_y_present[:, 1:].contiguous()
                            labels_absent = val_y_absent[:, 1:].contiguous()

                            val_outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                                decoder_present_input_ids=decoder_input_ids_present,
                                                decoder_present_attention_mask=val_y_present_att_mask[:,
                                                                               :-1].contiguous(),
                                                decoder_absent_input_ids=decoder_input_ids_absent,
                                                decoder_absent_attention_mask=val_y_absent_att_mask[:,
                                                                              :-1].contiguous(),
                                                labels_present=labels_present, labels_absent=labels_absent)

                            val_present_logits = torch.transpose(val_outputs[0], 1, 2)
                            val_absent_logits = torch.transpose(val_outputs[1], 1, 2)
                            present_val_loss_batch = criterion(val_present_logits, labels_present)
                            absent_val_loss_batch = criterion(val_absent_logits, labels_absent)

                            present_validation_loss += present_val_loss_batch.item()
                            absent_validation_loss += absent_val_loss_batch.item()
                        # print(present_val_loss_batch, present_validation_loss,absent_val_loss_batch, absent_validation_loss, batch_count)
                        # print(validation_loss)
                        present_perplexity = math.exp(present_validation_loss / batch_count)
                        absent_perplexity = math.exp(absent_validation_loss / batch_count)
                        validation_perplexity = present_perplexity + absent_perplexity
                        print('perp', present_perplexity, absent_perplexity, validation_perplexity)
                        if validation_perplexity < prev_validation_perplexity:
                            print("Validation perplexity improved from ", prev_validation_perplexity, " to ",
                                  validation_perplexity)
                            torch.save(model.state_dict(), model_location)
                            prev_validation_perplexity = validation_perplexity
                            not_improving_checkpoints = 0
                        else:
                            print("Validation perplexity did not improve.")
                            lr = lr / 2
                            optimizer = optim.Adam(model.parameters(), lr=lr)
                            for param_group in optimizer.param_groups:
                                print(param_group['lr'])
                            not_improving_checkpoints += 1
                    model.train()
                i += 1
                continue

            input_ids = X.to(torch.device(device))
            attention_mask = X_att_mask.to(torch.device(device))
            y_present = y_present.to(torch.device(device))
            y_present_att_mask = y_present_att_mask.to(torch.device(device))
            y_absent = y_absent.to(torch.device(device))
            y_absent_att_mask = y_absent_att_mask.to(torch.device(device))
            decoder_input_ids_present = y_present[:, :-1].contiguous()
            decoder_input_ids_absent = y_absent[:, :-1].contiguous()
            labels_present = y_present[:, 1:].contiguous()
            labels_absent = y_absent[:, 1:].contiguous()
            # print(decoder_input_ids.shape)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            decoder_present_input_ids=decoder_input_ids_present,
                            decoder_present_attention_mask=y_present_att_mask[:, :-1].contiguous(),
                            decoder_absent_input_ids=decoder_input_ids_absent,
                            decoder_absent_attention_mask=y_absent_att_mask[:, :-1].contiguous(),
                            labels_present=labels_present, labels_absent=labels_absent)
            # print('outputs', len(outputs))
            # print(len(outputs), type(outputs[0]), type(outputs[1]), type(outputs[2]))
            # quit()
            # print(outputs[1].shape)

            # loss = outputs[0]
            present_logits = torch.transpose(outputs[0], 1, 2)
            absent_logits = torch.transpose(outputs[1], 1, 2)
            # print(present_logits.shape)
            # print(absent_logits.shape)
            present_loss = criterion(present_logits, labels_present) / accumulation_steps
            absent_loss = criterion(absent_logits, labels_absent) / accumulation_steps

            loss = (0.8 * present_loss) + (0.2 * absent_loss)
            # print(loss)
            loss.backward()
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_data_loader):
                optimizer.step()
                optimizer.zero_grad()
                step_count += 1
                print('Epoch', epoch, 'step', step_count, 'present loss', present_loss.item(), 'absent loss',
                      absent_loss.item(), 'current validation perplexity', prev_validation_perplexity)

                if (step_count % validation_every == 0 or (i + 1) == len(train_data_loader)) and step_count != 0:
                    model.eval()
                    with torch.no_grad():
                        present_validation_loss = 0
                        absent_validation_loss = 0
                        batch_count = 0
                        for val_X, val_X_att_mask, val_y_present, val_y_present_att_mask, val_y_absent, val_y_absent_att_mask in validation_data_loader:
                            batch_count += 1
                            input_ids = val_X.to(torch.device(device))
                            attention_mask = val_X_att_mask.to(torch.device(device))

                            val_y_present = val_y_present.to(torch.device(device))
                            val_y_present_att_mask = val_y_present_att_mask.to(torch.device(device))
                            val_y_absent = val_y_absent.to(torch.device(device))
                            val_y_absent_att_mask = val_y_absent_att_mask.to(torch.device(device))
                            decoder_input_ids_present = val_y_present[:, :-1].contiguous()
                            decoder_input_ids_absent = val_y_absent[:, :-1].contiguous()
                            labels_present = val_y_present[:, 1:].contiguous()
                            labels_absent = val_y_absent[:, 1:].contiguous()

                            val_outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                                decoder_present_input_ids=decoder_input_ids_present,
                                                decoder_present_attention_mask=val_y_present_att_mask[:,
                                                                               :-1].contiguous(),
                                                decoder_absent_input_ids=decoder_input_ids_absent,
                                                decoder_absent_attention_mask=val_y_absent_att_mask[:,
                                                                              :-1].contiguous(),
                                                labels_present=labels_present, labels_absent=labels_absent)

                            val_present_logits = torch.transpose(val_outputs[0], 1, 2)
                            val_absent_logits = torch.transpose(val_outputs[1], 1, 2)
                            present_val_loss_batch = criterion(val_present_logits, labels_present)
                            absent_val_loss_batch = criterion(val_absent_logits, labels_absent)

                            present_validation_loss += present_val_loss_batch.item()
                            absent_validation_loss += absent_val_loss_batch.item()
                        # print(present_val_loss_batch, present_validation_loss,absent_val_loss_batch, absent_validation_loss, batch_count)
                        # print(validation_loss)
                        present_perplexity = math.exp(present_validation_loss / batch_count)
                        absent_perplexity = math.exp(absent_validation_loss / batch_count)
                        validation_perplexity = present_perplexity + absent_perplexity
                        print('perp', present_perplexity, absent_perplexity, validation_perplexity)
                        if validation_perplexity < prev_validation_perplexity:
                            print("Validation perplexity improved from ", prev_validation_perplexity, " to ",
                                  validation_perplexity)
                            torch.save(model.state_dict(), model_location)
                            prev_validation_perplexity = validation_perplexity
                            not_improving_checkpoints = 0
                        else:
                            print("Validation perplexity did not improve.")
                            lr = lr / 2
                            optimizer = optim.Adam(model.parameters(), lr=lr)
                            for param_group in optimizer.param_groups:
                                print(param_group['lr'])
                            not_improving_checkpoints += 1
                    model.train()
                if not_improving_checkpoints == patience:
                    print("Not improving for ", patience, " checkpoints. Sopping training.")
                    train_stop_flag = True
                    break
            i += 1
        if train_stop_flag == True:
            break


def train_model_augmented(X_location, X_att_mask_loc, y_location, y_att_mask_location, validation_data_loader,
                          model_location):
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    #model.load_state_dict(torch.load(model_location))
    # model = BartForConditionalGeneration_custom(pre_trained_model)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('pytorch_total_params', pytorch_total_params)
    # quit()
    if device == 'cuda':
        model.cuda()
    lr = learning_rate

    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=word_to_idx[pad_token])

    prev_validation_perplexity = 99999999
    patience = 5
    not_improving_checkpoints = 0
    train_stop_flag = False

    num_chunks = 5
    start_idx = 0
    end_idx = start_idx + 500000

    for epoch in range(0, num_epochs):
        model.train()
        i = 0
        step_count = 0
        optimizer.zero_grad()
        for chunk in range(num_chunks):
            train_data_loader = create_chunked_data_loader_BART(X_location, X_att_mask_loc, y_location, y_att_mask_location, batch_size, device,
                                        start_idx, end_idx, data_type='train')
            for X, X_att_mask, y, y_att_mask in train_data_loader:
                if epoch == -1 and step_count <= 30000:
                    print(i, step_count)
                    if (i + 1) % accumulation_steps == 0:
                        step_count += 1
                    if ((i + 1) == -1):
                        model.eval()
                        with torch.no_grad():
                            validation_loss = 0
                            batch_count = 0
                            for val_X, val_X_att_mask, val_y, val_y_att_mask in validation_data_loader:
                                batch_count += 1
                                input_ids = val_X.to(torch.device(device))
                                attention_mask = val_X_att_mask.to(torch.device(device))
                                val_y = val_y.to(torch.device(device))
                                decoder_input_ids = val_y[:, :-1].contiguous()
                                labels = val_y[:, 1:].contiguous()
                                val_y_att_mask = val_y_att_mask.to(torch.device(device))

                                val_outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                                    decoder_input_ids=decoder_input_ids,
                                                    decoder_attention_mask=val_y_att_mask[:, :-1].contiguous(),
                                                    labels=labels)
                                val_logits = torch.transpose(val_outputs[1], 1, 2)
                                val_loss_batch = criterion(val_logits, labels)

                                validation_loss += val_loss_batch.item()
                                print(validation_loss)
                            validation_perplexity = math.exp(validation_loss / batch_count)
                            checkpoint_info['epoch'] = epoch
                            checkpoint_info['step_count'] = step_count
                            checkpoint_info['i'] = i

                            print('perp', validation_perplexity)
                            if validation_perplexity < prev_validation_perplexity:
                                print("Validation perplexity improved from ", prev_validation_perplexity, " to ",
                                      prev_validation_perplexity)
                                torch.save(model.state_dict(), model_location)
                                prev_validation_perplexity = validation_perplexity
                                not_improving_checkpoints = 0
                                checkpoint_info['best_perp'] = validation_perplexity
                            else:
                                print("Validation perplexity did not improve.")
                                not_improving_checkpoints += 1
                            print(checkpoint_info)
                            save_data(checkpoint_info, checkpoint_info_location)
                        model.train()
                    if not_improving_checkpoints == patience:
                        print("Not improving for ", patience, " checkpoints. Sopping training.")
                        train_stop_flag = True
                        break
                    i += 1
                    continue

                input_ids = X.to(torch.device(device))
                attention_mask = X_att_mask.to(torch.device(device))
                y = y.to(torch.device(device))
                y_att_mask = y_att_mask.to(torch.device(device))
                decoder_input_ids = y[:, :-1].contiguous()
                labels = y[:, 1:].contiguous()
                # print(decoder_input_ids.shape)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                                decoder_attention_mask=y_att_mask[:, :-1].contiguous(), labels=labels)
                # print('outputs', len(outputs))
                # print(len(outputs), type(outputs[0]), type(outputs[1]), type(outputs[2]))
                # quit()
                # print(outputs[1].shape)

                # loss = outputs[0]
                logits = torch.transpose(outputs[1], 1, 2)
                loss = criterion(logits, labels) / accumulation_steps
                loss.backward()
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_data_loader):
                    optimizer.step()
                    optimizer.zero_grad()
                    step_count += 1
                    print('Epoch', epoch, 'step', step_count, 'loss', loss.item(), 'current validation perplexity',
                          prev_validation_perplexity)

                    if step_count % validation_every == 0 or ((i + 1) == len(train_data_loader)) and step_count != 0:
                        model.eval()
                        with torch.no_grad():
                            validation_loss = 0
                            batch_count = 0
                            for val_X, val_X_att_mask, val_y, val_y_att_mask in validation_data_loader:
                                batch_count += 1
                                input_ids = val_X.to(torch.device(device))
                                attention_mask = val_X_att_mask.to(torch.device(device))
                                val_y = val_y.to(torch.device(device))
                                decoder_input_ids = val_y[:, :-1].contiguous()
                                labels = val_y[:, 1:].contiguous()
                                val_y_att_mask = val_y_att_mask.to(torch.device(device))

                                val_outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                                    decoder_input_ids=decoder_input_ids,
                                                    decoder_attention_mask=val_y_att_mask[:, :-1].contiguous(),
                                                    labels=labels)
                                val_logits = torch.transpose(val_outputs[1], 1, 2)
                                val_loss_batch = criterion(val_logits, labels)

                                validation_loss += val_loss_batch.item()
                                # print(validation_loss)
                            validation_perplexity = math.exp(validation_loss / batch_count)
                            print('perp', validation_perplexity)
                            checkpoint_info['epoch'] = epoch
                            checkpoint_info['step_count'] = step_count
                            checkpoint_info['i'] = i

                            if validation_perplexity < prev_validation_perplexity:
                                print("Validation perplexity improved from ", prev_validation_perplexity, " to ",
                                      validation_perplexity)
                                torch.save(model.state_dict(), model_location)
                                prev_validation_perplexity = validation_perplexity
                                not_improving_checkpoints = 0
                                checkpoint_info['best_perp'] = validation_perplexity
                            else:
                                print("Validation perplexity did not improve.")
                                not_improving_checkpoints += 1
                            save_data(checkpoint_info, checkpoint_info_location)
                        model.train()
                    if not_improving_checkpoints == patience:
                        print("Not improving for ", patience, " checkpoints. Sopping training.")
                        train_stop_flag = True
                        break
                i += 1
            if train_stop_flag == True:
                break
            start_idx = end_idx+1
            end_idx = min((start_idx+500000),2700000)



X_train = load_data(os.path.join(base, "kp20k", "kp20k_train_src_encoded_padded.pkl"))
# X_test = load_data(base+"kp20k\\kp20k_test_src_numeric.pkl")
print(len(X_train))
X_valid = load_data(os.path.join(base, "kp20k", "kp20k_valid_src_encoded_padded.pkl"))

#X_train_att_mask = load_data(os.path.join(base, "kp20k", "kp20k_train_src_attention_masks.pkl"))
# X_test_lengths = load_data(base+"kp20k\\kp20k_test_src_length.pkl")
X_valid_att_mask = load_data(os.path.join(base, "kp20k", "kp20k_valid_src_attention_masks.pkl"))

'''y_train_present = load_data(os.path.join(base,"kp20k","kp20k_train_present_tgt_encoded_padded.pkl"))
#X_test = load_data(base+"kp20k\\kp20k_test_src_numeric.pkl")
y_valid_present = load_data(os.path.join(base,"kp20k","kp20k_valid_present_tgt_encoded_padded.pkl"))

y_train_absent = load_data(os.path.join(base,"kp20k","kp20k_train_absent_tgt_encoded_padded.pkl"))
#X_test = load_data(base+"kp20k\\kp20k_test_src_numeric.pkl")
y_valid_absent = load_data(os.path.join(base,"kp20k","kp20k_valid_absent_tgt_encoded_padded.pkl"))

y_train_present_att_mask = load_data(os.path.join(base,"kp20k","kp20k_train_present_tgt_attention_masks.pkl"))
#X_test_lengths = load_data(base+"kp20k\\kp20k_test_src_length.pkl")
y_valid_present_att_mask = load_data(os.path.join(base,"kp20k","kp20k_valid_present_tgt_attention_masks.pkl"))

y_train_absent_att_mask = load_data(os.path.join(base,"kp20k","kp20k_train_absent_tgt_attention_masks.pkl"))
#X_test_lengths = load_data(base+"kp20k\\kp20k_test_src_length.pkl")
y_valid_absent_att_mask = load_data(os.path.join(base,"kp20k","kp20k_valid_absent_tgt_attention_masks.pkl"))'''

#####single encoder
#y_train = load_data(os.path.join(base, "kp20k", "kp20k_train_tgt_encoded_padded.pkl"))
# X_test = load_data(base+"kp20k\\kp20k_test_src_numeric.pkl")
y_valid = load_data(os.path.join(base, "kp20k", "kp20k_valid_tgt_encoded_padded.pkl"))

#y_train_att_mask = load_data(os.path.join(base, "kp20k", "kp20k_train_tgt_attention_masks.pkl"))
# X_test_lengths = load_data(base+"kp20k\\kp20k_test_src_length.pkl"
y_valid_att_mask = load_data(os.path.join(base, "kp20k", "kp20k_valid_tgt_attention_masks.pkl"))


X_train_location = os.path.join(base, "kp20k", "kp20k_train_src_encoded_padded.json")
X_train_att_location = os.path.join(base, "kp20k", "kp20k_train_src_attention_masks.json")
y_train_location = os.path.join(base, "kp20k", "kp20k_train_tgt_encoded_padded.json")
y_train_att_location = os.path.join(base, "kp20k", "kp20k_train_tgt_attention_masks.pkl")
#train_data_loader = create_data_loaders_BART(X_train, X_train_att_mask, y_train, y_train_att_mask, batch_size, device,
                                            # data_type='train')
validation_data_loader = create_data_loaders_BART(X_valid, X_valid_att_mask, y_valid, y_valid_att_mask, batch_size,
                                                  device, data_type='eval')
train_model_augmented(validation_data_loader, model_location)


# train_data_loader = create_data_loaders_BART_dual_decoder(X_train, X_train_att_mask, y_train_present, y_train_present_att_mask,y_train_absent, y_train_absent_att_mask, batch_size, device, data_type = 'train')
# validation_data_loader = create_data_loaders_BART_dual_decoder(X_valid, X_valid_att_mask, y_valid_present, y_valid_present_att_mask,y_valid_absent, y_valid_absent_att_mask, batch_size, device, data_type = 'eval')
# train_model_dual_decoders(train_data_loader, validation_data_loader, model_location)

def create_output_seq_tgt(translated_output):
    # print(translated_output)
    output_seq = []
    i = 0
    while translated_output[i] != '</s>' and i + 1 < len(translated_output):
        i += 1
        # print(i)
        current_token = translated_output[i]
        keyphrase = []
        while current_token != '<sep>' and i + 1 < len(translated_output):
            if current_token != '<eofpr>':
                keyphrase.append(current_token)
            i += 1
            current_token = translated_output[i]
        if len(keyphrase) > 0 and keyphrase not in output_seq:
            output_seq.append(keyphrase)
    # print(output_seq)
    return output_seq


def create_output_seq(translated_output):
    # print(translated_output)
    output_seq = []
    i = 0
    '''while translated_output[i]!= '</s>' and i+1<len(translated_output):
        i+=1
        #print(i)
    for i in range(0, len(translated_output)-1)
        current_token = translated_output[i]
        keyphrase = []
        while current_token != ',' and i+1<len(translated_output):
            if current_token!= ';':
                keyphrase.append(current_token)
            i+=1
            current_token = translated_output[i]
        if len(keyphrase)>0 and keyphrase not in output_seq:
            output_seq.append(keyphrase)'''

    Keyphrases = translated_output.split(',')

    for key in Keyphrases:
        if ';' in key:
            output_seq.append([';'])
            key = key.replace(';', '')
        key = meng17_tokenize(key)
        if len(key) > 0 and key not in output_seq:
            output_seq.append(key)

    # print(output_seq)
    return output_seq


def create_output_seq_batch(translated_batch):
    output_seq_batch = [create_output_seq(x) for x in translated_batch]
    return output_seq_batch


def generate_output_sequences(model_location, test_data_loader, output_seq_save_location, beam_size=50, max_len=150):
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    if device == 'cuda':
        model.cuda()
    model.load_state_dict(torch.load(model_location))

    model.eval()
    output_sequences = []
    with torch.no_grad():
        batch_count = 0
        # val_src, val_src_extended, val_src_length, val_tgt, val_tgt_extended, val_tgt_length, val_extended_vocab_sizes
        for test_X, test_X_att_mask, test_y, test_y_att_mask in test_data_loader:
            batch_count += 1
            # model(val_src, val_src_extended, val_src_length, val_tgt, val_extended_vocab_sizes)

            input_ids = test_X.to(torch.device(device))
            attention_mask = test_X_att_mask.to(torch.device(device))
            # decoder_input_ids = test_y[:, :-1].contiguous()

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_len,
                num_beams=beam_size,
                length_penalty=1.0,
                early_stopping=True
            )
            translated_output = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g
                                 in generated_ids]

            # print('translated_output', translated_output)
            # print(translated_output)
            batch_out_sequence = create_output_seq_batch(translated_output)
            # print(batch_out_sequence)
            # print(len(output_sequences))
            output_sequences += batch_out_sequence

    save_data(output_sequences, output_seq_save_location)
    return output_sequences


import copy


def generate_output_sequences_dual_decoder(model_location, test_data_loader, output_seq_save_location, decoder_type,
                                           beam_size=50, max_len=150):
    pre_trained_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    # shared = copy.deepcopy(pre_trained_model.model.shared)
    # print(shared)

    trained_model = BartForConditionalGenerationTwoDecoders(pre_trained_model)

    if device == 'cuda':
        trained_model.cuda()
        # shared.cuda()
    trained_model.load_state_dict(torch.load(model_location))
    # print(trained_model.state_dict()['final_logits_bias_present'])

    # for p1, p2 in zip(shared.parameters(), trained_model.model.shared.parameters()):
    # if p1.data.ne(p2.data).sum() > 0:
    #    print('False======================')
    # else:
    #   print('True======================')

    model = pre_trained_model
    model.model.shared.load_state_dict(trained_model.model.shared.state_dict())
    # model.model.encoder = trained_model.model.encoder
    model.model.encoder.load_state_dict(trained_model.model.encoder.state_dict())
    if decoder_type == 'present':
        # model.model.decoder = trained_model.model.decoder_present
        model.model.decoder.load_state_dict(trained_model.model.decoder_present.state_dict())
    else:
        # model.model.decoder = trained_model.model.decoder_absent
        model.model.decoder.load_state_dict(trained_model.model.decoder_absent.state_dict())

    if device == 'cuda':
        model.cuda()
    # print(model)
    # print(model.state_dict()['final_logits_bias'])
    model.eval()
    output_sequences = []
    with torch.no_grad():
        batch_count = 0
        # val_src, val_src_extended, val_src_length, val_tgt, val_tgt_extended, val_tgt_length, val_extended_vocab_sizes
        for test_X, test_X_att_mask, test_y_present, test_y_present_att_mask, test_y_absent, test_y_absent_att_mask in test_data_loader:
            batch_count += 1
            # model(val_src, val_src_extended, val_src_length, val_tgt, val_extended_vocab_sizes)

            input_ids = test_X.to(torch.device(device))
            attention_mask = test_X_att_mask.to(torch.device(device))
            # decoder_input_ids = test_y[:, :-1].contiguous()
            # print(input_ids)
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_len,
                num_beams=beam_size,
                length_penalty=1.0,
                early_stopping=True
            )
            # print(generated_ids)
            translated_output = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g
                                 in generated_ids]

            # print('translated_output', translated_output)
            # print(translated_output)
            batch_out_sequence = create_output_seq_batch(translated_output)
            # print(batch_out_sequence)
            # print(len(output_sequences))
            output_sequences += batch_out_sequence

    save_data(output_sequences, output_seq_save_location)
    return output_sequences


def calculate_f1_at_k_single(gold, predicted, k):
    # print(gold)
    # print(predicted)
    num_predicted = len(predicted)
    stemmed_gold = [stem_word_list(x) for x in gold]
    predicted = predicted[0:k]
    stemmed_predicted = [stem_word_list(x) for x in predicted]

    # print(stemmed_gold)
    # print(stemmed_predicted)

    correct = 0
    for pred in stemmed_predicted:
        if pred in stemmed_gold:
            correct += 1
    # print(correct)

    num_correct_at_k = correct
    num_target = len(stemmed_gold)

    if num_correct_at_k == 0 or num_predicted == 0:
        precision_at_k = 0
        recall = 0
        f1_at_k = 0
    else:
        precision_at_k = num_correct_at_k / k
        recall = num_correct_at_k / num_target
        f1_at_k = (2 * precision_at_k * recall) / (precision_at_k + recall)
    # print('precision_at_k, recall, f1_at_k', precision_at_k, recall, f1_at_k)
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
    # print(correct)

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

    # print('precision_at_k, recall, f1_at_k', precision_at_k, recall, f1_at_k)
    return precision_at_k, recall, f1_at_k


def evaluate_output(tgt_gold_seq_location, output_sequences_location, K):
    output_sequences = load_data(output_sequences_location)
    # print(len(output_sequences))

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
            # print("=============================================")
            json_dict = json.loads(json_line)
            src = json_dict['tokenized']['src']
            # gold_tgt = json_dict['tokenized']['combined_target_sequence']
            gold_tgt = json_dict['tokenized']['tgt']
            # print('gold seq',gold_tgt)
            # gold_tgt = create_output_seq_tgt(gold_tgt)
            # print('gold list',gold_tgt)
            predicted_tgt = output_sequences[i]

            present_tgt_flags, occurance_positions, _ = if_present_duplicate_phrases(src, gold_tgt)
            present_tgts = [tgt for tgt, present in zip(gold_tgt, present_tgt_flags) if present]
            absent_tgts = [tgt for tgt, present in zip(gold_tgt, present_tgt_flags) if ~present]

            # print('present gold', len(present_tgts))
            # print('absent gold', len(absent_tgts))

            # print('predicted list',predicted_tgt)

            present_pred_flags, occurance_positions, _ = if_present_duplicate_phrases(src, predicted_tgt)
            present_pred = [tgt for tgt, present in zip(predicted_tgt, present_pred_flags) if present]
            absent_pred = [tgt for tgt, present in zip(predicted_tgt, present_pred_flags) if ~present and tgt[0] != ';']
            '''present_pred = []
            j = 0
            if len(predicted_tgt)>0:
                key = predicted_tgt[j]
                while key!=[';'] and j<len(predicted_tgt):
                    present_pred.append(key)
                    j+=1
                    if j < len(predicted_tgt):
                        key = predicted_tgt[j]
                j+=1
                if j<len(predicted_tgt):
                    absent_pred = predicted_tgt[j:len(predicted_tgt)]
                else:
                    absent_pred = []
            else:
                present_pred = []
                absent_pred = []'''
            # print('present pred', present_pred)
            # print('absent pred', absent_pred)

            # print('present pred', len(present_pred))
            # print('present tgt', len(present_tgts))
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
            i += 1

    print('Present keyphrases:')
    print('\n')
    print('Scores at K:')
    print('Preicision: ', sum(precisions_at_k_present) / len(precisions_at_k_present))
    print('Recall: ', sum(recalls_at_k_present) / len(recalls_at_k_present))
    print('F1: ', sum(f1s_at_k_present) / len(f1s_at_k_present))
    print('\n')

    print('Scores at M:')
    print('Preicision: ', sum(precisions_at_m_present) / len(precisions_at_m_present))
    print('Recall: ', sum(recalls_at_m_present) / len(recalls_at_m_present))
    print('F1: ', sum(f1s_at_m_present) / len(f1s_at_m_present))

    print('\n')
    print('\n')

    print('Absent keyphrases:')
    print('\n')
    print('Scores at K:')
    print('Preicision: ', sum(precisions_at_k_absent) / len(precisions_at_k_absent))
    print('Recall: ', sum(recalls_at_k_absent) / len(recalls_at_k_absent))
    print('F1: ', sum(f1s_at_k_absent) / len(f1s_at_k_absent))
    print('\n')

    print('Scores at M:')
    print('Preicision: ', sum(precisions_at_m_absent) / len(precisions_at_m_absent))
    print('Recall: ', sum(recalls_at_m_absent) / len(recalls_at_m_absent))
    print('F1: ', sum(f1s_at_m_absent) / len(f1s_at_m_absent))


def evaluate_output_only_present_or_absent(tgt_gold_seq_location, output_sequences_location, K, presentOrAbsent):
    output_sequences = load_data(output_sequences_location)
    # print(len(output_sequences))

    sample_idx_start = 0
    precisions_at_k = []
    recalls_at_k = []
    f1s_at_k = []

    precisions_at_m = []
    recalls_at_m = []
    f1s_at_m = []

    i = 0
    with open(tgt_gold_seq_location, 'r', encoding="utf8", ) as input_json:
        for json_line in input_json:
            # print("=============================================")
            json_dict = json.loads(json_line)
            src = json_dict['tokenized']['src']
            # gold_tgt = json_dict['tokenized']['combined_target_sequence']
            gold_tgt_both = json_dict['tokenized']['tgt']
            # print('gold seq',gold_tgt)
            # gold_tgt = create_output_seq_tgt(gold_tgt)
            # print('gold list',gold_tgt)
            predicted_tgt = output_sequences[i]

            present_tgt_flags, occurance_positions, _ = if_present_duplicate_phrases(src, gold_tgt_both)
            if presentOrAbsent == 'present':
                gold_tgts = [tgt for tgt, present in zip(gold_tgt_both, present_tgt_flags) if present]
            else:
                gold_tgts = [tgt for tgt, present in zip(gold_tgt_both, present_tgt_flags) if ~present]

            # print('present gold', len(present_tgts))
            # print('absent gold', len(absent_tgts))

            # print('predicted list',predicted_tgt)

            # print('present pred', len(present_pred))
            # print('absent pred', len(absent_pred))

            # print('present pred', len(present_pred))
            # print('present tgt', len(present_tgts))
            precision_at_k, recall_at_k, f1_at_k = calculate_f1_at_k_single(predicted_tgt, gold_tgts, K)

            precisions_at_k.append(precision_at_k)
            recalls_at_k.append(recall_at_k)
            f1s_at_k.append(f1_at_k)

            ##########at m
            precision_at_m, recall_at_m, f1_at_m = calculate_f1_at_m_single(predicted_tgt, gold_tgts)

            precisions_at_m.append(precision_at_m)
            recalls_at_m.append(recall_at_m)
            f1s_at_m.append(f1_at_m)

            i += 1

    print(presentOrAbsent, ' keyphrases:')
    print('\n')
    print('Scores at K:')
    print('Preicision: ', sum(precisions_at_k) / len(precisions_at_k))
    print('Recall: ', sum(recalls_at_k) / len(recalls_at_k))
    print('F1: ', sum(f1s_at_k) / len(f1s_at_k))
    print('\n')

    print('Scores at M:')
    print('Preicision: ', sum(precisions_at_m) / len(precisions_at_m))
    print('Recall: ', sum(recalls_at_m) / len(recalls_at_m))
    print('F1: ', sum(f1s_at_m) / len(f1s_at_m))


dataset_names = ['inspec', 'krapivin', 'nus', 'semeval', 'duc', 'kp20k']

'''for dataset_name in dataset_names:
    print('======================dataset name===================', dataset_name)
    # input_json_path = os.path.join(json_base_dir, dataset_name, '%s_test_tokenized.json' % dataset_name)
    # output_json_path = os.path.join(json_base_dir, dataset_name, '%s_test_tokenized_targets_combined.json' % dataset_name)
    generated_present_output_location = os.path.join(generated_base, '%s_test_generated_present_keyphrases.pkl' % dataset_name)
    generated_absent_output_location = os.path.join(generated_base,
                                                     '%s_test_generated_absent_keyphrases.pkl' % dataset_name)
    tgt_gold_seq_location = os.path.join(test_gold_base, dataset_name,
                                         '%s_test_tokenized_targets_combined.json' % dataset_name)
    # if dataset_name== 'kp20k' or dataset_name== 'duc':
    X_test = load_data(os.path.join(base, dataset_name, '%s_test_src_encoded_padded.pkl' % dataset_name))
    X_test_att_mask = load_data(os.path.join(base, dataset_name, "%s_test_src_attention_masks.pkl" % dataset_name))
    y_test_present = load_data(os.path.join(base,dataset_name,"%s_test_present_tgt_encoded_padded.pkl" % dataset_name))
    y_test_present_att_mask = load_data(os.path.join(base,dataset_name,"%s_test_present_tgt_attention_masks.pkl" % dataset_name))

    y_test_absent = load_data(
        os.path.join(base, dataset_name, "%s_test_absent_tgt_encoded_padded.pkl" % dataset_name))
    y_test_absent_att_mask = load_data(
        os.path.join(base, dataset_name, "%s_test_absent_tgt_attention_masks.pkl" % dataset_name))

    test_data_loader = create_data_loaders_BART_dual_decoder(X_test, X_test_att_mask, y_test_present, y_test_present_att_mask,y_test_absent, y_test_absent_att_mask, batch_size, device, data_type = 'eval')
    # X, X_att_mask, y, y_att_mask, batch_size, device, data_type = 'train'):
    #generate_output_sequences_dual_decoder(model_location, test_data_loader, output_seq_save_location, decoder_type, beam_size=50, max_len=150)
    generate_output_sequences_dual_decoder(model_location, test_data_loader, generated_present_output_location,'present', beam_size=10, max_len=80)
    # model_location, test_data_loader, output_seq_save_location,beam_size = 50, max_len= 150
    evaluate_output_only_present_or_absent(tgt_gold_seq_location, generated_present_output_location, 5, 'present')

    generate_output_sequences_dual_decoder(model_location, test_data_loader, generated_absent_output_location,
                                           'absent', beam_size=10, max_len=80)

    evaluate_output_only_present_or_absent(tgt_gold_seq_location, generated_absent_output_location, 5, 'absent')'''

for dataset_name in dataset_names:
    print('======================dataset name===================', dataset_name)
    # input_json_path = os.path.join(json_base_dir, dataset_name, '%s_test_tokenized.json' % dataset_name)
    # output_json_path = os.path.join(json_base_dir, dataset_name, '%s_test_tokenized_targets_combined.json' % dataset_name)
    generated_output_location = os.path.join(generated_base, '%s_test_generated_keyphrases.pkl' % dataset_name)
    tgt_gold_seq_location = os.path.join(test_gold_base, dataset_name,
                                         '%s_test_tokenized_targets_combined.json' % dataset_name)
    # if dataset_name == 'krapivin':
    # evaluate_output(tgt_gold_seq_location, generated_output_location, 5)
    if dataset_name != 'kp20k':
        X_test = load_data(os.path.join(base, dataset_name, '%s_test_src_encoded_padded.pkl' % dataset_name))
        X_test_att_mask = load_data(os.path.join(base, dataset_name, "%s_test_src_attention_masks.pkl" % dataset_name))
        y_test = load_data(os.path.join(base, dataset_name, "%s_test_tgt_encoded_padded.pkl" % dataset_name))
        y_test_att_mask = load_data(os.path.join(base, dataset_name, "%s_test_tgt_attention_masks.pkl" % dataset_name))
        test_data_loader = create_data_loaders_BART(X_test, X_test_att_mask, y_test, y_test_att_mask, batch_size,
                                                    device, data_type='eval')
        print('asldjas;ld')
        generate_output_sequences(model_location, test_data_loader, generated_output_location, beam_size=10,
                                  max_len=200)
        # evaluate_output_only_present_or_absent(tgt_gold_seq_location, generated_output_location, 50, 'absent')
        evaluate_output(tgt_gold_seq_location, generated_output_location, 5)
        # evaluate_output(tgt_gold_seq_location, output_sequences_location, K)
        # evaluate_output_only_present_or_absent(tgt_gold_seq_location, output_sequences_location, K, presentOrAbsent)

