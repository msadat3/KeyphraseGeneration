##Few functions are adapted from: https://github.com/memray/OpenNMT-kpg-release/tree/master/onmt/keyphrase
from nltk.stem.porter import *
import numpy as np
import pickle

stemmer = PorterStemmer()
DIGIT_token = "<digit>"

def stem_word_list(word_list):
    return [stemmer.stem(w.strip()) for w in word_list]

def if_present_duplicate_phrases(src_seq, tgt_seqs, stemming=True, lowercase=True):
    """
    Check if each given target sequence verbatim appears in the source sequence
    :param src_seq:
    :param tgt_seqs:
    :param stemming:
    :param lowercase:
    :param check_duplicate:
    :return:
    """
    if lowercase:
        src_seq = [w.lower() for w in src_seq]
    if stemming:
        src_seq = stem_word_list(src_seq)

    present_indices = []
    present_flags = []
    duplicate_flags = []
    phrase_set = set()  # some phrases are duplicate after stemming, like "model" and "models" would be same after stemming, thus we ignore the following ones

    for tgt_seq in tgt_seqs:
        if lowercase:
            tgt_seq = [w.lower() for w in tgt_seq]
        if stemming:
            tgt_seq = stem_word_list(tgt_seq)

        # check if the phrase appears in source text
        # iterate each word in source
        match_flag, match_pos_idx = if_present_phrase(src_seq, tgt_seq)

        # if it reaches the end of source and no match, means it doesn't appear in the source
        present_flags.append(match_flag)
        present_indices.append(match_pos_idx)

        # check if it is duplicate
        if '_'.join(tgt_seq) in phrase_set:
            duplicate_flags.append(True)
        else:
            duplicate_flags.append(False)
        phrase_set.add('_'.join(tgt_seq))

    assert len(present_flags) == len(present_indices)

    return np.asarray(present_flags), \
           np.asarray(present_indices), \
           np.asarray(duplicate_flags)

def if_present_phrase(src_str_tokens, phrase_str_tokens):
    """
    :param src_str_tokens: a list of strings (words) of source text
    :param phrase_str_tokens: a list of strings (words) of a phrase
    :return:
    """
    match_flag = False
    match_pos_idx = -1
    for src_start_idx in range(len(src_str_tokens) - len(phrase_str_tokens) + 1):
        match_flag = True
        # iterate each word in target, if one word does not match, set match=False and break
        for seq_idx, seq_w in enumerate(phrase_str_tokens):
            src_w = src_str_tokens[src_start_idx + seq_idx]
            if src_w != seq_w:
                match_flag = False
                break
        if match_flag:
            match_pos_idx = src_start_idx
            break

    return match_flag, match_pos_idx

def meng17_tokenize(text):
    '''
    The tokenizer used in Meng et al. ACL 2017
    parse the feed-in text, filtering and tokenization
    keep [_<>,\(\)\.\'%], replace digits to <digit>, split by [^a-zA-Z0-9_<>,\(\)\.\'%]
    :param text:
    :return: a list of tokens
    '''
    # remove line breakers
    text = re.sub(r'[\r\n\t]', ' ', text)
    # pad spaces to the left and right of special punctuations
    text = re.sub(r'[_<>,\(\)\.\'%]', ' \g<0> ', text)
    # tokenize by non-letters (new-added + # & *, but don't pad spaces, to make them as one whole word)
    text = text.lower()
    tokens = list(filter(lambda w: len(w) > 0, re.split(r'[^a-zA-Z0-9_<>,#&\+\*\(\)\.\']', text)))

    return tokens

def replace_numbers_to_DIGIT(tokens, k=2):
    # replace big numbers (contain more than k digit) with <digit>
    tokens = [w if not re.match('^\d{%d,}$' % k, w) else DIGIT_token for w in tokens]

    return tokens

def match_keyphrase(keyphrase_1, keyphrase_2):
    keyphrase_1 = stem_word_list(keyphrase_1)
    keyphrase_2 = stem_word_list(keyphrase_2)

    if keyphrase_1 == keyphrase_2:
        return True
    else:
        return False

def remove_duplicate_keyphrases(keyphrases):
    unique_keyphrases = []
    changed = 0
    for i in range(len(keyphrases)):
        match_flag = 0
        for j in range(len(unique_keyphrases)):
            if match_keyphrase(keyphrases[i], unique_keyphrases[j]):
                #print(keyphrases)
                #print(keyphrases[i], unique_keyphrases[j])
                match_flag = 1
                changed = 1
        if match_flag == 0:
            unique_keyphrases.append(keyphrases[i])

    #if changed == 1:
       # print(unique_keyphrases)
    return unique_keyphrases


def load_data(location):
    with open(location, 'rb') as file:
        data = pickle.load(file)
        return data

def save_data(data, location):
    with open(location, 'wb') as file:
        pickle.dump(data,file)