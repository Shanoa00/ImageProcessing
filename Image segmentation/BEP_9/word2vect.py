from pickle import TRUE
import re
from numpy.testing._private.utils import assert_array_almost_equal
import torch
import numpy as np
import string
from torch.autograd import Variable
import torch.functional as F
import torch.nn.functional as F

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def tokenized_corpus(corpus):
    corpus_raw= corpus.lower()

    #all_words= corpus_raw.replace([".",","], "").split()
    # all_words= re.sub('[^A-Za-z0-9]+', ' ', corpus_raw).split()
    return corpus_raw.translate(str.maketrans('', '', string.punctuation)).split() #faster!!

def wordtonum(tokenized_text):
    words= set(tokenized_text) #duplicate words are removed
    vocab_size= len(words)

    word2int= {}
    int2word= {}
    onehot= np.eye(vocab_size, dtype=int)

    for i,word in enumerate(words):
        print("i:",i, " word:", word)
        word2int[word] = i
        int2word[i] = word
    return word2int, int2word, vocab_size

def centre_word(tokenized_text):
    window_size= 2
    idx_pairs = []
    word2int, int2word, vocab_size= wordtonum(tokenized_text)
    indices = [word2int[word] for word in tokenized_text]
    # for each word, threated as center word
    for center_word_pos in range(len(indices)):
        # for each window position
        for w in range(-window_size, window_size + 1):
            context_word_pos = center_word_pos + w
            # make soure not jump out sentence
            if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                continue
            context_word_idx = indices[context_word_pos]
            idx_pairs.append((indices[center_word_pos], context_word_idx))

    # for i in idx_pairs:
    #     print(int2word[i[0]],",", int2word[i[1]])
    return np.array(idx_pairs), word2int, int2word, vocab_size # it will be useful to have this as numpy array
    
#input layer:
def get_input_layer(word_idx, vocab_size):
    x = torch.zeros(vocab_size).float()
    x[word_idx] = 1.0
    return x

def similarity(v,u):
  return torch.dot(v,u)/(torch.norm(v)*torch.norm(u))

corpus_raw= 'Word vectors represent a significant leap forward in advancing our ability to analyze relationships across words, sentences, \
            and documents. In doing so, they advance technology by providing machines with much more information about words than has previously been possible \
            using traditional representations of words.'
    
def main():
    TRAIN_MODEL= False
    tokenized_text= tokenized_corpus(corpus_raw)
    idx_pairs, word2int, int2word, vocab_size= centre_word(tokenized_text)
    
    #hidden layer 
    embedding_dims = 5
    W1 = Variable(torch.randn(embedding_dims, vocab_size).float(), requires_grad=True)
    W2 = Variable(torch.randn(vocab_size, embedding_dims).float(), requires_grad=True)
    num_epochs = 1000
    learning_rate = 0.001
    if TRAIN_MODEL:
        for epo in range(num_epochs):
            loss_val = 0
            for data, target in idx_pairs:
                x = Variable(get_input_layer(data,vocab_size)).float()
                y_true = Variable(torch.from_numpy(np.array([target])).long())

                z1 = torch.matmul(W1, x)
                z2 = torch.matmul(W2, z1)

                log_softmax = F.log_softmax(z2, dim=0)

                loss = F.nll_loss(log_softmax.view(1,-1), y_true)
                loss_val += loss.item()
                loss.backward()
                W1.data -= learning_rate * W1.grad.data
                W2.data -= learning_rate * W2.grad.data

                W1.grad.data.zero_()
                W2.grad.data.zero_()
            if epo % 50 == 0:    
                print(f'Loss at epo {epo}: {loss_val/len(idx_pairs)}')
        print("")
        torch.save(W1, "W1")
        torch.save(W2, "W2")
    else:
        W1= torch.load("W1")
        W2= torch.load("W2")

        word1 = 'advance'
        word2 = 'technology'
        w1v = torch.matmul(W1,get_input_layer(word2int[word1],vocab_size))
        w2v = torch.matmul(W1,get_input_layer(word2int[word2],vocab_size))
        print("test1:", similarity(w1v, w2v))

        word1 = 'word'
        word2 = 'vectors'
        #from 0~1: strong relationship between words
        #0: no relationship 
        #from 0~-1: opposite relationship between words
        w1v = torch.matmul(W1,get_input_layer(word2int[word1.lower()],vocab_size))
        w2v = torch.matmul(W1,get_input_layer(word2int[word2.lower()],vocab_size))
        print("test2:", similarity(w1v, w2v))

if __name__=="__main__":
    main()
