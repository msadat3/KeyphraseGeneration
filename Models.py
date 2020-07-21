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

word_2_idx = load_data("E:\ResearchData\Keyphrase Generation\DataForExperiments\\word_to_idx.pkl")
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
        #print('decoder embedding', embed)
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

    def create_mask(self, input_sequence):
        return (input_sequence != self.pad_idx).permute(1, 0)
    def forward(self, input_seq, input_lengths, target_seq, teacher_forcing_ratio = 0.5, decode_style='training'):

        batch_size = len(input_lengths)

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
        decoder_output_probabilities = torch.zeros(self.max_output_length, batch_size, self.vocab_size).to(self.device)

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

            decoder_output_probabilities = torch.transpose(decoder_output_probabilities, 0,1)
            decoder_output_probabilities = torch.transpose(decoder_output_probabilities, 1, 2)
            return decoder_output_probabilities
        elif decode_style == 'greedy':
            decoded_outputs = torch.zeros(self.max_output_length, batch_size).to(self.device)
            decoded_outputs[0] = decoder_input
            for t in range(1, self.max_output_length):
                #print(decoder_input.shape)
                #print('decoder_input', decoder_input)

                decoder_input = decoder_input.unsqueeze(1)

                #(self, input_token, encoder_hidden_states, projected_encoder_hidden_states, src_mask, prev_hidden):
                #print(decoder_hidden.shape)
                #print('actu', decoder_input.shape)
                output, decoder_hidden = self.decoder(decoder_input, encoder_hidden_states, projected_encoder_hidden_states, src_mask, decoder_hidden)

                #decoder_output_probabilities[t] = output

                decoder_input = torch.argmax(output, dim=2)

                decoded_outputs[t] = decoder_input

                decoder_input = torch.transpose(decoder_input, 0, 1)
                decoder_input = decoder_input.squeeze()
            print('decoded shape',decoded_outputs.shape)
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
        for idx in range(src_masks.shape[1]):
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
                    #print(score)
                    nodes.put((score, nn))
                    # increase qsize
                qsize += len(nextnodes) - 1

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            utterances = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = []
                utterance.append(n.wordid)
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance.append(n.wordid)

                utterance = utterance[::-1]
                utterances.append(utterance)

            decoded_batch.append(utterances)

        return decoded_batch

class BahdanauAttention(nn.Module):
#inspired from: https://bastings.github.io/annotated_encoder_decoder/
    """Implements Bahdanau (MLP) attention"""

    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()

        # We assume a bi-directional encoder so key_size is 2*hidden_size
        #key_size = 2 * hidden_size if key_size is None else key_size
        query_size = hidden_size

        #self.key_layer = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)###for concat size is doubleed

        # to store attention scores
        self.alphas = None

    def forward(self, query=None, proj_key=None, value=None, mask=None):
        assert mask is not None, "mask is required"

        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computated.
        query = self.query_layer(query)

        # Calculate scores.
        #print('wuery shape',query.shape)
        #print('proj_key shape', proj_key.shape)
        #torch.cat([query, proj_key], dim=2)
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
       # print(context.shape, alphas.shape)
        return context, alphas



class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
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

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward