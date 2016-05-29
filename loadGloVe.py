from numbers import Number
from pandas import DataFrame
import sys, codecs
import numpy as np
import time

def find_word_vector_matrix(vector_file, vocab, vocab_dim):
  '''Read a GloVe array from sys.argv[1] and return its vectors and labels as arrays'''
  # vocab_dim = 50 embed_size of your word vectors
  # vocab list of words
  embedding_weights = np.zeros((1, vocab_dim))
  start = time.time()
      
  word_vectors = []
  #labels_array = []
  
  index_dict = {}
  for i, word in enumerate(vocab):
    index_dict[word] = i+1 # adding 1 to account for 0th index (for masking)dict((c, i + 1) )
    with codecs.open(vector_file, 'r', 'utf-8') as f:
      for c, r in enumerate(f):
        sr = r.split()
        if sr[0] == word.lower():
          word_vectors.append( np.array([float(i) for i in sr[1:]]))
          break
        #embedding_weights = np.append(embedding_weights, [np.array([float(i) for i in sr[1:]])], axis = 0)
      #if c == n_words - 1:
       # break
        #return np.array( numpy_arrays ), index_dict#abels_array
  embedding_weights = np.append(embedding_weights, np.array(word_vectors), axis = 0)
  print 'Total loading time: {}'.format(time.time() - start)
  #n_symbols = len(index_dict) + 1 
  
  return embedding_weights, index_dict#labels_array

def build_word_vector_matrix(vector_file, n_words):
  '''Read a GloVe array from sys.argv[1] and return its vectors and labels as arrays'''
  vocab_dim = 50 # embed_size of your word vectors
  embedding_weights = np.zeros((1, vocab_dim))
  
  word_vectors = []
  #labels_array = []
  index_dict = {}
  with codecs.open(vector_file, 'r', 'utf-8') as f:
    start = time.time()
    for c, r in enumerate(f):
      sr = r.split()
      word_vectors.append( np.array([float(i) for i in sr[1:]]) )
      #embedding_weights = np.append(embedding_weights, [np.array([float(i) for i in sr[1:]])], axis = 0)
      index_dict[sr[0]] = c+1 # adding 1 to account for 0th index (for masking)

      if c == n_words - 1:
        break
        #return np.array( numpy_arrays ), index_dict#abels_array
  embedding_weights = np.append(embedding_weights, np.array(word_vectors), axis = 0)
  print 'Total loading time: {}'.format(time.time() - start)
  #n_symbols = len(index_dict) + 1 
  
  return embedding_weights, index_dict#labels_array

def build_word_vector_matrix2(vector_file, n_words):
  '''Read a GloVe array from sys.argv[1] and return its vectors and labels as arrays'''
  word_vectors = []
  #labels_array = []
  index_dict = {}
  #word_vectors = {}
  with codecs.open(vector_file, 'r', 'utf-8') as f:
    for c, r in enumerate(f):
      sr = r.split()
      #labels_array.append(sr[0])
      word_vectors.append( np.array([float(i) for i in sr[1:]]) )
      index_dict[sr[0]] = c+1

      if c == n_words:
        break
        #return np.array( numpy_arrays ), index_dict#abels_array
  
  vocab_dim = len(np.array(word_vectors)[0]) # dimensionality of your word vectors
  n_symbols = len(index_dict) + 1 # adding 1 to account for 0th index (for masking)
  embedding_weights = np.zeros((n_symbols, vocab_dim))
  
  for word,index in index_dict.items():
    embedding_weights[index,:] = np.array(word_vectors)[index - 1, :]
  
  return embedding_weights, index_dict#labels_array

if __name__ == "__main__":
  input_vector_file = sys.argv[1] # The Glove file to analyze (e.g. glove.6B.300d.txt)
  n_words           = int(sys.argv[2]) # The number of lines to read from the input file
  df, labels_array  = build_word_vector_matrix(input_vector_file, n_words)
  
  #print(index_dict)
  #print(labels_array)
  print(df[0])
  #for c in cluster_to_words:
   # print cluster_to_words[c]
    #print "\n"