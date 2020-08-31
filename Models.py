import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import torch
from queue import PriorityQueue
import operator
from Utils import *
import os

#base = "/home/ubuntu/Keyphrase_Generation/DataForExperiments_pointer_generator/"

base = "E:\ResearchData\Keyphrase Generation\DataForExperiments_pointer_generator\\"


word_2_idx = load_data(os.path.join(base,"word_to_idx.pkl"))
class GRU_Encoder(nn.Module):
    def __init__(self, embedding, embedding_size, hidden_size):
        super(GRU_Encoder, self).__init__()
        #self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.embedding = embedding

        self.GRU = nn.GRU(embedding_size, hidden_size, bidirectional=True)

    def forward(self, input_seq, input_lengths):
        #print('input shape', input_seq.shape)
        embed = self.embedding(input_seq)
        embed = torch.transpose(embed, 0, 1)
        #print('embed shape', embed.shape)
        #print('Embedding', embed)
        embed = torch.nn.utils.rnn.pack_padded_sequence(embed, input_lengths, enforce_sorted=False)
        hidden_states, last_hidden_state = self.GRU(embed)
        hidden_states, _ = torch.nn.utils.rnn.pad_packed_sequence(hidden_states)

        #print('last_hidden_state shape', last_hidden_state.shape)
        #print('last_hidden_state 0 shape', last_hidden_state[0, :, :].shape)
        last_hidden_state = torch.cat([last_hidden_state[0, :, :], last_hidden_state[1, :, :]], dim=1).unsqueeze(0)
        #print('last_hidden_state shape', last_hidden_state.shape)
        #print('last_hidden_state shape', last_hidden_state.shape)
       # print(last_hidden_state)
        return hidden_states, last_hidden_state

class GRU_Decoder(nn.Module):
    def __init__(self, embedding, vocab_size, embedding_size, hidden_size):
        super(GRU_Decoder, self).__init__()
        #self.embedding_size = embedding_size
        self.embedding = embedding

        self.GRU = nn.GRU(embedding_size+(2*hidden_size), hidden_size, bidirectional=False)

        self.attention = BahdanauAttention(hidden_size)

        self.linear = nn.Linear(embedding_size+(3*hidden_size), hidden_size)

        self.output_layer = nn.Linear(hidden_size,vocab_size)

    def forward(self, input_token, encoder_hidden_states, projected_encoder_hidden_states, src_mask, prev_hidden):
        #print('decoder input shape', input_token.shape)
        embed = self.embedding(input_token)
        #print('decoder embedding', embed.shape)
        embed = torch.transpose(embed, 0, 1)

        #print('prev_hidden shape', prev_hidden.shape)
        context_vector, _ = self.attention(query= prev_hidden, proj_key=projected_encoder_hidden_states, value=encoder_hidden_states, mask=src_mask)

        context_vector = torch.transpose(context_vector,0,1)
        #print('decoder embed shape', embed.shape)
        #print('context_vector embed shape', context_vector.shape)
        rnn_input = torch.cat([embed, context_vector], dim=2)
        rnn_output,prev_hidden = self.GRU(rnn_input, prev_hidden)
        #print('prev_hidden',prev_hidden)
        #print('decoder rnn out',  rnn_output.shape)
        concat = torch.cat([embed,rnn_output,context_vector],dim=2)

        #print('concat shaoe', concat.shape)
        linear_output = self.linear(concat)

        out = self.output_layer(linear_output)

        output = F.log_softmax(out, dim=2)
        #print('output', output.shape)
        #sum = torch.sum(output, dim=2)
        #print('attention sum', sum)
        #print('decoder out shaoe======================================', output.shape)
        return output, prev_hidden



class GRU_Decoder_pointer_generator(nn.Module):
    def __init__(self, embedding, vocab_size, embedding_size, hidden_size, device, coverage_enabled = False):
        super(GRU_Decoder_pointer_generator, self).__init__()
        #self.embedding_size = embedding_size
        self.embedding = embedding

        self.GRU = nn.GRU(embedding_size+(2*hidden_size), hidden_size, bidirectional=False)

        self.attention = BahdanauAttention(hidden_size, coverage_enabled= coverage_enabled)

        self.linear = nn.Linear(embedding_size+(3*hidden_size), hidden_size)

        self.pointer_generator_switch = nn.Linear(embedding_size+(3*hidden_size), 1)

        self.output_layer = nn.Linear(hidden_size,vocab_size)

        self.batch_size = None
        self.device = device

    def forward(self, input_token, encoder_hidden_states, projected_encoder_hidden_states, src_mask, prev_hidden, encoder_batch_extended_vocab, batch_max_oov, coverage = None):
        #print('decoder input shape', input_token.shape)
        self.batch_size = input_token.shape[0]
        embed = self.embedding(input_token)
        #print('decoder embedding', embed.shape)
        embed = torch.transpose(embed, 0, 1)

        #print('prev_hidden shape', prev_hidden.shape)
        if coverage != None:
            context_vector, attention_dist, coverage, min_between_attention_coverage = self.attention(query= prev_hidden, proj_key=projected_encoder_hidden_states, value=encoder_hidden_states, mask=src_mask, coverage = coverage)
        else:
            context_vector, attention_dist = self.attention(query= prev_hidden, proj_key=projected_encoder_hidden_states, value=encoder_hidden_states, mask=src_mask)
        #print(attention_dist.shape, coverage.shape)
        #min_between_attention_coverage = torch.min(attention_dist, coverage)

       # print('attention_dist shape just', attention_dist.shape)
        context_vector = torch.transpose(context_vector,0,1)
        #print('decoder embed shape', embed.shape)
        #print('context_vector embed shape', context_vector.shape)
        rnn_input = torch.cat([embed, context_vector], dim=2)
        rnn_output,prev_hidden = self.GRU(rnn_input, prev_hidden)
        #print('prev_hidden',prev_hidden)
        #print('decoder rnn out',  rnn_output.shape)
        concat = torch.cat([embed,rnn_output,context_vector],dim=2)

       # print('concat shape', concat.shape)
        p_gen = torch.sigmoid(self.pointer_generator_switch(concat))
       # print('p_gen shape', p_gen.shape)
        #print(p_gen)
        #print('concat shaoe', concat.shape)
        linear_output = self.linear(concat)

        out = self.output_layer(linear_output)

        encoder_batch_extended_vocab = encoder_batch_extended_vocab.unsqueeze(0)
        generation_prob = F.softmax(out, dim=2)
        #print('gen_prob 0',generation_prob[0][0][encoder_batch_extended_vocab[0][0][0]])
       # sum = torch.sum(attention_dist, dim=1)
        #print('attention_dist sum', sum)
       # print('pgrn', p_gen)

       # print('generation_prob shape', generation_prob.shape)
        generation_prob = p_gen * generation_prob
       # print('gen_prob 0', generation_prob[0][0][encoder_batch_extended_vocab[0][0][0]])
        attention_dist = (1 - p_gen) * attention_dist

        #print('attention_dist shape', attention_dist.shape)

        #print('batch_max_oov', batch_max_oov)
        zero_generation_probs_for_oovs = torch.zeros(self.batch_size, batch_max_oov).to(self.device)
        #zero_generation_probs_for_oovs = zero_generation_probs_for_oovs + 1e-1
        zero_generation_probs_for_oovs = zero_generation_probs_for_oovs.unsqueeze(0)
        #print('zero_generation_probs_for_oovs', zero_generation_probs_for_oovs.shape)

        generation_prob = torch.cat([generation_prob, zero_generation_probs_for_oovs], dim=2)
        generation_prob = generation_prob + 1e-12###for avoiding -inf in log space
        #print('generation_prob shape after zero added', generation_prob.shape)
       # print('gen_prob after adding zeroes and exp',generation_prob[0][0][encoder_batch_extended_vocab[0][0][0]])

        #print('encoder_batch_extended_vocab shape after ', encoder_batch_extended_vocab.shape)


        #print('attention_dist after mult',attention_dist)
        #print('attension 0', attention_dist[0][0][0])

        output = generation_prob.scatter_add_(2, encoder_batch_extended_vocab, attention_dist)
       # print('output 0', output[0][0][encoder_batch_extended_vocab[0][0][0]])

        #print('output', output.shape, output)
        #sum = torch.sum(output, dim=2)
        #print('output sum', sum)
        output = torch.log(output)
        #print('output after log', output.shape, output)
        #sum = torch.sum(output, dim=2)
        #print('output sum', sum)
        #print('decoder out shaoe======================================', output.shape)
        if coverage != None:
            return output, prev_hidden, coverage, min_between_attention_coverage
        else:
            return output, prev_hidden


class Seq2Seq_pointer_generator(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, pad_idx, eos_idx, sos_idx, unk_idx, max_output_length, device, coverage_enabled = False):
        super(Seq2Seq_pointer_generator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.encoder = GRU_Encoder(self.embedding, embedding_size, hidden_size)

        self.project_encoder_states = nn.Linear(2*hidden_size, hidden_size)
        self.bridge = nn.Linear(2 * hidden_size, hidden_size, bias=True)

        self.coverage_enabled = coverage_enabled

        self.decoder = GRU_Decoder_pointer_generator(self.embedding, vocab_size, embedding_size, hidden_size, device, coverage_enabled=self.coverage_enabled)

        self.pad_idx = pad_idx
        self.eos_idx = eos_idx
        self.sos_idx = sos_idx
        self.unk_idx = unk_idx
        self.max_output_length = max_output_length
        self.vocab_size = vocab_size
        self.device = device
        self.batch_size = None
        self.max_input_length_batch = None

    def create_mask(self, input_sequence):
        return (input_sequence != self.pad_idx).permute(1, 0)
    def forward(self, input_seq, input_seq_extended_with_oov, input_lengths, target_seq, extended_vocab_sizes, teacher_forcing_ratio = 0.5,decode_style='training'):

        self.batch_size = len(input_lengths)


        encoder_hidden_states, encoder_last_hidden_state = self.encoder(input_seq, input_lengths)

        projected_encoder_hidden_states = self.project_encoder_states(encoder_hidden_states)##for efficiency. could be done in decoder at every step also

        #print('projected shape',projected_encoder_hidden_states.shape)

        #print('projected', projected_encoder_hidden_states)

        try:
            if target_seq == None:
                #inference = True
                output_tokens = torch.zeros((input_seq.shape[0], self.max_output_length)).long().fill_(self.sos_idx).to(self.device)
            else:
                output_tokens = target_seq
        except:
            #inference = False
            #print('wtf')
            output_tokens = target_seq
            #print('output tokens shape',output_tokens.shape)
        #print(output_tokens)
        decoder_hidden = self.decoder_init_hidden(encoder_last_hidden_state)


        #print('decoder output probbailities', decoder_output_probabilities.shape)
        self.max_input_length_batch = max(input_lengths)

        src_mask = self.create_mask(input_seq)
        #print(src_mask)
        if decode_style == "training":
            if self.coverage_enabled:
                coverage_vectors = torch.zeros(self.max_input_length_batch, self.batch_size).to(self.device)
                batch_coverage_loss = torch.zeros(self.batch_size).to(self.device)
            #print('traom',coverage_vectors.shape)
            decoder_input = output_tokens[:, 0]

            #print(extended_vocab_sizes.shape)
            batch_max_oov = max(extended_vocab_sizes)
            decoder_output_probabilities = torch.zeros(self.max_output_length, self.batch_size,
                                                       self.vocab_size + batch_max_oov).to(self.device)

            for t in range(1, self.max_output_length):
                #print(decoder_input.shape)
                #print('decoder_input', decoder_input)
                decoder_input = decoder_input.unsqueeze(1)

                #(self, input_token, encoder_hidden_states, projected_encoder_hidden_states, src_mask, prev_hidden):
                #print(decoder_hidden.shape)
                #print('actu', decoder_input.shape)
                #input_token, encoder_hidden_states, projected_encoder_hidden_states, src_mask, prev_hidden, encoder_batch_extended_vocab
                #print('train',input_seq_extended_with_oov.shape)
                if self.coverage_enabled:
                    output, decoder_hidden, coverage_vectors, min_between_attention_coverage = self.decoder(decoder_input, encoder_hidden_states, projected_encoder_hidden_states, src_mask, decoder_hidden, input_seq_extended_with_oov, batch_max_oov, coverage_vectors)
                    batch_coverage_loss_step = torch.sum(min_between_attention_coverage, dim=1)
                    batch_coverage_loss += batch_coverage_loss_step
                else:
                    output, decoder_hidden = self.decoder(decoder_input, encoder_hidden_states, projected_encoder_hidden_states, src_mask, decoder_hidden, input_seq_extended_with_oov, batch_max_oov)
                #print(coverage_vectors)
               # print('min_between_attention_coverage', min_between_attention_coverage.shape, min_between_attention_coverage)

                #print('batch_coverage_loss_step',batch_coverage_loss_step.shape, batch_coverage_loss_step)
                #print('batch_coverage_loss', batch_coverage_loss.shape)

                #print(batch_coverage_loss)


                decoder_output_probabilities[t] = output

                #teacher_force = random.random() < teacher_forcing_ratio
                #print(torch.argmax(output,dim=2))
                teacher_force = True
                if teacher_force:
                    #print('teacher force===================================')
                    decoder_input = target_seq[:,t]
                else:
                    #print('argmaxxxxxxxxx===================================')
                    decoder_input = torch.argmax(output,dim=2)
                    decoder_input = torch.transpose(decoder_input,0,1)
                    decoder_input = decoder_input.squeeze()
                #print(t)
            #print('divinding by', self.max_output_length)

            #print('after divide', batch_coverage_loss)

            decoder_output_probabilities = torch.transpose(decoder_output_probabilities, 0,1)
            decoder_output_probabilities = torch.transpose(decoder_output_probabilities, 1, 2)
            if self.coverage_enabled:
                batch_coverage_loss = batch_coverage_loss/self.max_output_length
                #print(batch_coverage_loss)
                batch_coverage_loss = torch.sum(batch_coverage_loss) / self.batch_size

                return decoder_output_probabilities, batch_coverage_loss
            else:
                return decoder_output_probabilities

        elif decode_style == 'greedy':
            #greedy_decode(self, decoder_hidden, encoder_hidden_states, projected_encoder_hidden_states, src_mask)
            decoded_outputs = self.greedy_decode(decoder_hidden,encoder_hidden_states, projected_encoder_hidden_states, src_mask, input_seq_extended_with_oov, extended_vocab_sizes)

            return decoded_outputs
        elif decode_style == 'beam':
            decoded = self.beam_decode(encoder_hidden_states, projected_encoder_hidden_states, src_mask, decoder_hidden, input_seq_extended_with_oov,
                        word_2_idx['<s>'], word_2_idx['</s>'], extended_vocab_sizes, device='cuda', beam_width=50, max_len=56)
            #print('balsalbalsal')
           # print(decoded)
            return decoded
    def decoder_init_hidden(self, encoder_final_state):
            """Returns the initial decoder state,
            conditioned on the final encoder state."""

            return torch.tanh(self.bridge(encoder_final_state))

    def greedy_decode(self, decoder_hidden, encoder_hidden_states, projected_encoder_hidden_states, src_mask, input_seq_extended_with_oov, extended_vocab_sizes):

        decoded_outputs = []
        for i in range(self.batch_size):
            decoder_input = torch.tensor([self.sos_idx], dtype=torch.long, device=torch.device(self.device))
            decoded_output = []
            decoded_output.append(self.sos_idx)
            #print('decoder hidden batch', decoder_hidden.shape)
            #print('encoder_hidden_states batch', encoder_hidden_states.shape)
            #print('projected_encoder_hidden_states batch', projected_encoder_hidden_states.shape)
            decoder_hidden_for_idx = decoder_hidden[:, i, :].unsqueeze(1)
            encoder_hidden_states_for_idx = encoder_hidden_states[:, i, :].unsqueeze(1)
            projected_encoder_hidden_states_for_idx = projected_encoder_hidden_states[:, i, :].unsqueeze(1)

            input_seq_extended_with_oov_for_idx = input_seq_extended_with_oov[i, :].unsqueeze(0)

            num_oov = extended_vocab_sizes[i]
            if self.coverage_enabled:
                coverage_vector_for_idx = torch.zeros(self.max_input_length_batch, 1).to(self.device)
            #print(coverage_vector_for_idx.shape)
            #print('decoder hidden idx', decoder_hidden_for_idx.shape)
            #print('encoder_hidden_states idx', encoder_hidden_states_for_idx.shape)
            #print('projected_encoder_hidden_states idx', projected_encoder_hidden_states_for_idx.shape)
            #print('decoder_input before loop', decoder_input.shape, decoder_input)
            for t in range(1, self.max_output_length):
                # print(decoder_input.shape)
                #print('decoder_input before unsqueeze', decoder_input.shape, decoder_input)
                if decoder_input.item() >= self.vocab_size: ###if oov
                   # print('oov')
                    decoder_input = torch.tensor([self.unk_idx], dtype=torch.long, device=torch.device(self.device))
                decoder_input = decoder_input.unsqueeze(1)
                #print(decoder_input)
                #print('decoder_input', decoder_input.shape, decoder_input)

                # input_token, encoder_hidden_states, projected_encoder_hidden_states, src_mask, prev_hidden, encoder_batch_extended_vocab
                # print(decoder_hidden.shape)
                # print('actu', decoder_input.shape)
                #self.decoder(decoder_input, encoder_hidden_states, projected_encoder_hidden_states, src_mask, decoder_hidden, input_seq_extended_with_oov, batch_max_oov, coverage_vectors)

                if self.coverage_enabled:
                    output, decoder_hidden_for_idx, coverage_vector_for_idx, _ = self.decoder(decoder_input, encoder_hidden_states_for_idx,
                                                              projected_encoder_hidden_states_for_idx, src_mask[i],
                                                              decoder_hidden_for_idx, input_seq_extended_with_oov_for_idx, num_oov, coverage_vector_for_idx)
                else:
                    output, decoder_hidden_for_idx = self.decoder(decoder_input, encoder_hidden_states_for_idx,
                                                              projected_encoder_hidden_states_for_idx, src_mask[i],
                                                              decoder_hidden_for_idx, input_seq_extended_with_oov_for_idx, num_oov)

                # decoder_output_probabilities[t] = output

                #print('output shape', output.shape)
                decoder_input = torch.argmax(output, dim=2)
                #print('decoder_input shape just after', decoder_input.shape, decoder_input)

                decoder_input = decoder_input.squeeze(dim=1)
                #print('decoder_input shape after squeeze', decoder_input.shape, decoder_input)
                # decoded_outputs[t] = decoder_input
                decoded_output.append(decoder_input.item())
                if decoder_input.item() == self.eos_idx:
                    #print(len(decoded_output))
                    break
                #print('decoder_input before transpose shape', decoder_input.shape, decoder_input)
                #decoder_input = torch.transpose(decoder_input, 0, 1)
                #decoder_input = decoder_input.squeeze()
                #print('decoder_input shape', decoder_input.shape)
            #print(decoded_output)
            decoded_outputs.append(decoded_output)
        return decoded_outputs

    def beam_decode(self, encoder_hidden_states, projected_encoder_hidden_states, src_masks, decoder_hiddens, input_seq_extended_with_oov,
                    SOS_token, EOS_token, extended_vocab_sizes, device='cuda', beam_width=50, max_len = 56):
        '''
        :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
        :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
        :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
        :return: decoded_batch
        '''

        topk = beam_width  # how many sentence do you want to generate
        decoded_batch = []

        # decoding goes sentence by sentence
        for idx in range(self.batch_size):
            #print(idx)
            #print('src mask len', src_masks.shape)
            #print(idx)
            if isinstance(decoder_hiddens, tuple):  # LSTM case
                decoder_hidden = (
                decoder_hiddens[0][:, idx, :].unsqueeze(0), decoder_hiddens[1][:, idx, :].unsqueeze(0))
            else:
                decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)

            projected_encoder_hidden_state = projected_encoder_hidden_states[:, idx, :].unsqueeze(1)
            input_seq_extended_with_oov_for_idx = input_seq_extended_with_oov[idx, :].unsqueeze(0)
            num_oov = extended_vocab_sizes[idx]
            if self.coverage_enabled:
                coverage_vector_for_idx = torch.zeros(self.max_input_length_batch, 1).to(self.device)
            #print('decoder_hidden', decoder_hidden.shape)
            #print('decoder_hiddens', decoder_hiddens.shape)

            #print('projected_encoder_hidden_states',projected_encoder_hidden_states.shape)
            #print('projected_encoder_hidden_state', projected_encoder_hidden_state.shape)
            encoder_hidden_state = encoder_hidden_states[:, idx, :].unsqueeze(1)
            #print('encoder_hidden_states', encoder_hidden_states.shape)
            #print('encoder_hidden_state', encoder_hidden_state.shape)
            #print(src_masks.shape)
            src_mask = src_masks[:,idx].unsqueeze(1)
            #print(src_mask.shape)
            # Start with the start of the sentence token
            #decoder_input = torch.LongTensor([[SOS_token]], device=device)
            decoder_input = torch.tensor([[SOS_token]], dtype=torch.long, device=torch.device(device))
            #print('input',decoder_input.shape)
            # Number of sentence to generate
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))

            # starting node -  hidden vector, previous node, word id, logp, length
            if self.coverage_enabled:
                node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1, coverage_vector=coverage_vector_for_idx)
            else:
                node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
            nodes = PriorityQueue()

            # start the queue
            nodes.put((-node.eval(), node))
            qsize = 1

            # start beam search
            while True:
                # give up when decoding takes too long
                #if qsize > 20000: break

                # fetch the best node
                try:
                    score, n = nodes.get()
                except Exception as e:
                    #print([(node.wordid, node.eval()) for node in nodes])
                    print(e)
                decoder_input = n.wordid
                #print(decoder_input.shape)
                decoder_hidden = n.h
                coverage_vector_for_idx = n.coverage_vector

                if (n.wordid.item() == EOS_token or n.leng == max_len) and n.prevNode != None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    #print(n.leng)

                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                # decode for one step using decoder
                if decoder_input.squeeze().item() >= self.vocab_size:####decoder gnerated a oov word
                    decoder_input = torch.tensor([[self.unk_idx]], dtype=torch.long, device=torch.device(device))
                if self.coverage_enabled:
                    decoder_output, decoder_hidden, coverage_vector_for_idx, _ = self.decoder(decoder_input, encoder_hidden_state, projected_encoder_hidden_state, src_mask, decoder_hidden, input_seq_extended_with_oov_for_idx, num_oov, coverage_vector_for_idx)
                else:
                    decoder_output, decoder_hidden = self.decoder(decoder_input, encoder_hidden_state, projected_encoder_hidden_state, src_mask, decoder_hidden, input_seq_extended_with_oov_for_idx, num_oov)


                # PUT HERE REAL BEAM SEARCH OF TOP
                #print('decoderoutput',decoder_output.shape)
                decoder_output = decoder_output.squeeze()
                decoder_output = decoder_output.squeeze()
                log_prob, indexes = torch.topk(decoder_output, beam_width)
                #print('indexes', indexes.shape, indexes)
                #print('log_prob', log_prob.shape)

                nextnodes = []

                for new_k in range(beam_width):
                    #print('new', new_k)
                    decoded_t = indexes[new_k].view(1, -1)
                    #print(log_prob)

                    log_p = log_prob[new_k].item()
                    if self.coverage_enabled:
                        node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1, coverage_vector_for_idx)
                    else:
                        node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                    score = -node.eval()
                    nextnodes.append((score, node))
                #print(len(nextnodes))
                # put them into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    #print(score, nn.leng, nn.logp, nn.prevNode, nn.wordid)
                    #print(idx)
                    try:
                        nodes.put((score, nn))
                    except Exception as e:
                        print(e)
                    # increase qsize
                qsize += len(nextnodes) - 1

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            utterances = []
            #print("=========================================")
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = []
                utterance.append(n.wordid.squeeze(dim=1).item())
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance.append(n.wordid.squeeze(dim=1).item())


                utterance = utterance[::-1]
                #print(score, utterance)
                utterances.append(utterance)
            #print(idx,utterances)
            decoded_batch.append(utterances[0])

        #print(decoded_batch)
        return decoded_batch



class BahdanauAttention(nn.Module):
#inspired from: https://bastings.github.io/annotated_encoder_decoder/
    """Implements Bahdanau (MLP) attention"""

    def __init__(self, hidden_size, coverage_enabled=False):
        super(BahdanauAttention, self).__init__()

        # We assume a bi-directional encoder so key_size is 2*hidden_size
        #key_size = 2 * hidden_size if key_size is None else key_size
        query_size = hidden_size

        #self.key_layer = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)###for concat size is doubleed
        self.coverage_enabled = coverage_enabled
        if self.coverage_enabled:
            self.coverage_layer = nn.Linear(1,hidden_size)

        # to store attention scores
        self.alphas = None

    def forward(self, query=None, proj_key=None, value=None, mask=None, coverage = None):
        assert mask is not None, "mask is required"
        assert proj_key is not None, "proj_key is required"
        assert value is not None, "value is required"

        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computated.
        #print(query)
        #print(query.shape)
        query = self.query_layer(query)
        #print('query',query)
        # Calculate scores.
        #print('wuery shape',query.shape)

        #torch.cat([query, proj_key], dim=2)
        if self.coverage_enabled:
            #print('coverage shape before', coverage.shape)
            #print(coverage)
            coverage = coverage.unsqueeze(2)
            coverage_projected = self.coverage_layer(coverage)
            #print('coverage_projected shape', coverage_projected.shape)
            scores = self.energy_layer(torch.tanh(query + proj_key+coverage_projected))
        else:
            scores = self.energy_layer(torch.tanh(query+proj_key))#have to figure out how t
        #scores = scores.squeeze(2).unsqueeze(1)
        #print('scores shape',scores.shape)

        # Mask out invalid positions.
        # The mask marks valid positions so we invert it using `mask & 0`.
        scores = scores.squeeze()
        #print(mask)
        #mask = mask[0:scores.shape[0],:]

        #print('mask', mask.shape, mask)
        #scores.data.masked_fill_(mask == False, -float('inf'))

        #print('scores shape', scores.shape)
        # Turn scores to probabilities.
        alphas = F.softmax(scores, dim=0)
        self.alphas = alphas

        # The context vector is the weighted sum of the values.
        value = torch.transpose(value, 0, 1)
        #print('alphas',alphas.shape)
        if len(alphas.shape) == 1:
            #print(sum(alphas))
            alphas = alphas.unsqueeze(1)
        alphas = torch.transpose(alphas, 0, 1)
        #print('decoder alphas shape', alphas.shape)
        #print('decoder alphas', alphas)
        #sum = torch.sum(alphas, dim=1)
        #print('attention sum', sum)
        alphas = alphas.unsqueeze(1)

        #print('alpha shape', alphas.shape)
        #print('value shape', value.shape)

        context = torch.bmm(alphas, value)
        #print('context',context)

        # context shape: [B, 1, 2D], alphas shape: [B, 1, M]
        alphas = alphas.squeeze()


        if self.coverage_enabled:

            coverage = coverage.squeeze(2)
            coverage = torch.transpose(coverage, 0,1)
           # print('end', context.shape, alphas.shape, coverage.shape)
            min_between_attention_coverage = torch.min(alphas, coverage)

            coverage = coverage + alphas
            #print(coverage)

            #print('attention', alphas)
            #print('coverage', coverage)

           # print('min_between_attention_coverage', alphas)
            coverage = torch.transpose(coverage, 0,1)
            #print('coverage final', coverage.shape, coverage)
            return context, alphas, coverage, min_between_attention_coverage
        else:
           # print('attention', alphas)
            return context, alphas









































































class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, pad_idx, eos_idx, sos_idx, max_output_length, device):
        super(Seq2Seq, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.encoder = GRU_Encoder(self.embedding, embedding_size, hidden_size)

        self.project_encoder_states = nn.Linear(2*hidden_size, hidden_size)
        self.bridge = nn.Linear(2 * hidden_size, hidden_size, bias=True)

        self.decoder = GRU_Decoder(self.embedding, vocab_size, embedding_size, hidden_size)

        self.pad_idx = pad_idx
        self.eos_idx = eos_idx
        self.sos_idx = sos_idx
        self.max_output_length = max_output_length
        self.vocab_size = vocab_size
        self.device = device
        self.batch_size = None

    def create_mask(self, input_sequence):
        return (input_sequence != self.pad_idx).permute(1, 0)
    def forward(self, input_seq, input_lengths, target_seq, teacher_forcing_ratio = 0.5, decode_style='training'):

        self.batch_size= len(input_lengths)

        encoder_hidden_states, encoder_last_hidden_state = self.encoder(input_seq, input_lengths)

        projected_encoder_hidden_states = self.project_encoder_states(encoder_hidden_states)

        #print('projected shape',projected_encoder_hidden_states.shape)

        #print('projected', projected_encoder_hidden_states)

        try:
            if target_seq == None:
                #inference = True
                output_tokens = torch.zeros((input_seq.shape[0], self.max_output_length)).long().fill_(self.sos_idx).to(self.device)
        except:
            #inference = False
            output_tokens = target_seq
            #print('output tokens shape',output_tokens.shape)
        #print(output_tokens)
        decoder_hidden = self.decoder_init_hidden(encoder_last_hidden_state)
        decoder_output_probabilities = torch.zeros(self.max_output_length, self.batch_size, self.vocab_size).to(self.device)

        #print('decoder output probbailities', decoder_output_probabilities.shape)
        decoder_input = output_tokens[:, 0]
        src_mask = self.create_mask(input_seq)
        #print(src_mask)
        if decode_style == "training":
            for t in range(1, self.max_output_length):
                #print(decoder_input.shape)
                #print('decoder_input', decoder_input)
                decoder_input = decoder_input.unsqueeze(1)

                #(self, input_token, encoder_hidden_states, projected_encoder_hidden_states, src_mask, prev_hidden):
                #print(decoder_hidden.shape)
                #print('actu', decoder_input.shape)
                output, decoder_hidden = self.decoder(decoder_input, encoder_hidden_states, projected_encoder_hidden_states, src_mask, decoder_hidden)

                decoder_output_probabilities[t] = output

                #teacher_force = random.random() < teacher_forcing_ratio
                #print(torch.argmax(output,dim=2))
                teacher_force = True
                if teacher_force:
                    #print('teacher force===================================')
                    decoder_input = target_seq[:,t]
                else:
                    #print('argmaxxxxxxxxx===================================')
                    decoder_input = torch.argmax(output,dim=2)
                    decoder_input = torch.transpose(decoder_input,0,1)
                    decoder_input = decoder_input.squeeze()
                #print('decoder_input',decoder_input)

            decoder_output_probabilities = torch.transpose(decoder_output_probabilities, 0,1)
            decoder_output_probabilities = torch.transpose(decoder_output_probabilities, 1, 2)
            return decoder_output_probabilities
        elif decode_style == 'greedy':
            #greedy_decode(self, decoder_hidden, encoder_hidden_states, projected_encoder_hidden_states, src_mask)
            decoded_outputs = self.greedy_decode(decoder_hidden,encoder_hidden_states, projected_encoder_hidden_states, src_mask)

            return decoded_outputs
        elif decode_style == 'beam':
            decoded = self.beam_decode(encoder_hidden_states, projected_encoder_hidden_states, src_mask, decoder_hidden,
                        word_2_idx['<s>'], word_2_idx['</s>'], device='cuda', beam_width=50, max_len=56)
            #print('balsalbalsal')
           # print(decoded)
            return decoded
    def decoder_init_hidden(self, encoder_final_state):
            """Returns the initial decoder state,
            conditioned on the final encoder state."""

            return torch.tanh(self.bridge(encoder_final_state))

    def greedy_decode(self, decoder_hidden, encoder_hidden_states, projected_encoder_hidden_states, src_mask):

        decoded_outputs = []
        for i in range(self.batch_size):
            decoder_input = torch.tensor([self.sos_idx], dtype=torch.long, device=torch.device(self.device))
            decoded_output = []
            decoded_output.append(self.sos_idx)
            #print('decoder hidden batch', decoder_hidden.shape)
            #print('encoder_hidden_states batch', encoder_hidden_states.shape)
            #print('projected_encoder_hidden_states batch', projected_encoder_hidden_states.shape)
            decoder_hidden_for_idx = decoder_hidden[:, i, :].unsqueeze(1)
            encoder_hidden_states_for_idx = encoder_hidden_states[:, i, :].unsqueeze(1)
            projected_encoder_hidden_states_for_idx = projected_encoder_hidden_states[:, i, :].unsqueeze(1)

            #print('decoder hidden idx', decoder_hidden_for_idx.shape)
            #print('encoder_hidden_states idx', encoder_hidden_states_for_idx.shape)
            #print('projected_encoder_hidden_states idx', projected_encoder_hidden_states_for_idx.shape)
            #print('decoder_input before loop', decoder_input.shape, decoder_input)
            for t in range(1, self.max_output_length):
                # print(decoder_input.shape)
                #print('decoder_input before unsqueeze', decoder_input.shape, decoder_input)
                decoder_input = decoder_input.unsqueeze(1)
                #print('decoder_input', decoder_input.shape, decoder_input)

                # (self, input_token, encoder_hidden_states, projected_encoder_hidden_states, src_mask, prev_hidden):
                # print(decoder_hidden.shape)
                # print('actu', decoder_input.shape)
                output, decoder_hidden_for_idx = self.decoder(decoder_input, encoder_hidden_states_for_idx,
                                                              projected_encoder_hidden_states_for_idx, src_mask[i],
                                                              decoder_hidden_for_idx)

                # decoder_output_probabilities[t] = output

                #print('output shape', output.shape)
                decoder_input = torch.argmax(output, dim=2)
                #print('decoder_input shape just after', decoder_input.shape, decoder_input)

                decoder_input = decoder_input.squeeze(dim=1)
                #print('decoder_input shape after squeeze', decoder_input.shape, decoder_input)
                # decoded_outputs[t] = decoder_input
                decoded_output.append(decoder_input.item())
                if decoder_input.item() == self.eos_idx:
                    break
                #print('decoder_input before transpose shape', decoder_input.shape, decoder_input)
                #decoder_input = torch.transpose(decoder_input, 0, 1)
                #decoder_input = decoder_input.squeeze()
                #print('decoder_input shape', decoder_input.shape)
            #print(decoded_output)
            decoded_outputs.append(decoded_output)
        return decoded_outputs

    def beam_decode(self, encoder_hidden_states, projected_encoder_hidden_states, src_masks, decoder_hiddens,
                    SOS_token, EOS_token, device='cuda', beam_width=50, max_len = 56):
        '''
        :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
        :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
        :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
        :return: decoded_batch
        '''

        topk = 1  # how many sentence do you want to generate
        decoded_batch = []

        # decoding goes sentence by sentence
        for idx in range(self.batch_size):
            #print('src mask len', src_masks.shape)
            #print(idx)
            if isinstance(decoder_hiddens, tuple):  # LSTM case
                decoder_hidden = (
                decoder_hiddens[0][:, idx, :].unsqueeze(0), decoder_hiddens[1][:, idx, :].unsqueeze(0))
            else:
                decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)

            projected_encoder_hidden_state = projected_encoder_hidden_states[:, idx, :].unsqueeze(1)
            #print('decoder_hidden', decoder_hidden.shape)
            #print('decoder_hiddens', decoder_hiddens.shape)

            #print('projected_encoder_hidden_states',projected_encoder_hidden_states.shape)
            #print('projected_encoder_hidden_state', projected_encoder_hidden_state.shape)
            encoder_hidden_state = encoder_hidden_states[:, idx, :].unsqueeze(1)
            #print('encoder_hidden_states', encoder_hidden_states.shape)
            #print('encoder_hidden_state', encoder_hidden_state.shape)
            #print(src_masks.shape)
            src_mask = src_masks[:,idx].unsqueeze(1)
            #print(src_mask.shape)
            # Start with the start of the sentence token
            #decoder_input = torch.LongTensor([[SOS_token]], device=device)
            decoder_input = torch.tensor([[SOS_token]], dtype=torch.long, device=torch.device(device))
            #print('input',decoder_input.shape)
            # Number of sentence to generate
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))

            # starting node -  hidden vector, previous node, word id, logp, length
            node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
            nodes = PriorityQueue()

            # start the queue
            nodes.put((-node.eval(), node))
            qsize = 1

            # start beam search
            while True:
                # give up when decoding takes too long
                if qsize > 2000: break

                # fetch the best node
                score, n = nodes.get()
                decoder_input = n.wordid
                #print(decoder_input.shape)
                decoder_hidden = n.h

                if (n.wordid.item() == EOS_token or n.leng == max_len) and n.prevNode != None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                # decode for one step using decoder
                decoder_output, decoder_hidden = self.decoder(decoder_input, encoder_hidden_state,
                                                         projected_encoder_hidden_state, src_mask, decoder_hidden)

                # PUT HERE REAL BEAM SEARCH OF TOP
                #print('decoderoutput',decoder_output.shape)
                decoder_output = decoder_output.squeeze()
                decoder_output = decoder_output.squeeze()
                log_prob, indexes = torch.topk(decoder_output, beam_width)
                #print('indexes', indexes.shape)
                #print('log_prob', log_prob.shape)

                nextnodes = []

                for new_k in range(beam_width):
                    #print('new', new_k)
                    decoded_t = indexes[new_k].view(1, -1)
                    #print(log_prob)

                    log_p = log_prob[new_k].item()

                    node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                    score = -node.eval()
                    nextnodes.append((score, node))

                # put them into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    #print(score, nn.leng, nn.logp, nn.prevNode, nn.wordid)
                    #print(idx)
                    try:
                        nodes.put((score, nn))
                    except Exception as e:
                        print(e)
                    # increase qsize
                qsize += len(nextnodes) - 1

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            utterances = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = []
                utterance.append(n.wordid.squeeze(dim=1).item())
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance.append(n.wordid.squeeze(dim=1).item())

                utterance = utterance[::-1]
                utterances.append(utterance)
            #print(idx,utterances)
            decoded_batch.append(utterances[0])

        #print(decoded_batch)
        return decoded_batch




class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length, coverage_vector = None):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length
        self.coverage_vector = coverage_vector

    def __lt__(self, other):
        return self.leng < other.leng

    def __eq__(self, other):
        if (other == None):
            return False
        if (not isinstance(other, BeamSearchNode)):
            return False
        return self.leng == other.leng

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

       # return self.logp


