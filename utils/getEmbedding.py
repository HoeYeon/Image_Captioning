from config import config, rnnConfig
from utils.load_data import *
import numpy as np
from utils.preprocessing import *
from pickle import load

#caption type dict
'''def capToList(caption):
    all_train_captions = []
    for key, val in caption.items():
        for cap in val:
            all_train_captions.append(cap)
    return all_train_captions

def getVoca(all_train_captions,word_count_threshold=5):
    word_counts = {}
    for sent in all_train_captions:
        for w in sent.split(' '):
            word_counts[w] = word_counts.get(w,0) + 1
    voca = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print('preprocessed words %d -> %d' % (len(word_counts), len(voca)))
    return voca

def word2Idx(voca):
    word2ix = {}

    ix = 1
    for w in vocab:
        word2ix[w] = ix
        ix += 1
    return word2ix'''

def word2Idx():
    tokenizer = load(open(config['tokenizer_path'], 'rb'))
    return tokenizer.word_index

def getGlove():
    glove_dir = 'glove'
    embedding_index = {}
    f= open(os.path.join(glove_dir, 'glove.6B.200d.txt'),encoding='utf-8')

    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:],dtype='float32')
        embedding_index[word] = coefs
    f.close()
    print('We have %s word vectors'% len(embedding_index))
    return embedding_index

def setGlove():
    embedding_dim = 200
    wordtoix = word2Idx()

    embedding_index = getGlove()
    vocab_size = len(wordtoix)+1
    print('vocab_size:',vocab_size)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    for word, i in wordtoix.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix

