import re
from numpy.testing._private.utils import assert_array_almost_equal
import tensorflow as tf
import numpy as np
import string

#corpus_raw= 'He is the king. The king is royal. She is the royal  queen'
corpus_raw= 'Word vectors represent a significant leap forward in advancing our ability to analyze relationships across words, sentences, and documents. In doing so, they advance technology by providing machines with much more information about words than has previously been possible using traditional representations of words.'
corpus_raw= corpus_raw.lower()

#all_words= corpus_raw.replace([".",","], "").split()
# all_words= re.sub('[^A-Za-z0-9]+', ' ', corpus_raw).split()
all_words= corpus_raw.translate(str.maketrans('', '', string.punctuation)).split() #faster!!

words= set(all_words) #duplicate words are removed
vocab_size= len(words)

word2int= {}
int2word= {}
onehot= np.eye(vocab_size, dtype=int)

for i,word in enumerate(words):
    print("i:",i, " word:", word)
    word2int[word] = i
    int2word[i] = word

#num= word2int['royal']
#print(onehot[num])
#print(int2word[num])

#test
for w in all_words:
    num= word2int[w]
    print("onehot:", onehot[num], " word:", w)

#https://jalammar.github.io/illustrated-word2vec/
#https://dzone.com/articles/introduction-to-word-vectors
#https://www.analyticsvidhya.com/blog/2018/02/natural-language-processing-for-beginners-using-textblob/
#https://towardsdatascience.com/learn-word2vec-by-implementing-it-in-tensorflow-45641adaf2ac
#https://towardsdatascience.com/nlp-101-word2vec-skip-gram-and-cbow-93512ee24314


