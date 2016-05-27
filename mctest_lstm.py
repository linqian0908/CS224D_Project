from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import sys, re

import cPickle

from keras.preprocessing import sequence
from keras.initializations import uniform
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU

def flatten(a):
    ret = []
    for item in a:
        if type(item) == list:
            for k in item:
                ret.append(k)
        else:
            ret.append(k)
    return ret

def get_dataset(questions):
    X = []
    y = []
    for question in questions:
        statements = []
        statements += flatten(question[2])
        statements += question[3]
        X.append(statements)
        y.append(question[4])
    return X,y

if __name__ == "__main__":
    train_file = sys.argv[1]
    dev_file = train_file.replace('train','dev')
    test_file = train_file.replace('train', 'test')

    print("Loading pickled train dataset")
    f = file(train_file, 'rb')
    obj = cPickle.load(f)
    train_dataset, train_questions, word_to_id, num_words, null_word_id, _, _ = obj
    
    print("Loading pickled development dataset")
    f = file(dev_file,'rb')
    obj = cPickle.load(f)
    dev_dataset, dev_questions, _, _, _, _, _ = obj
    
    print("Loading pickled test dataset")
    f = file(test_file, 'rb')
    obj = cPickle.load(f)
    test_dataset, test_questions, _, _, _, _, _ = obj

    nb_epoch = 10
    if len(sys.argv) > 2:
        nb_epoch = int(sys.argv[2])

    X_train, y_train = get_dataset(train_questions)
    X_dev, y_dev = get_dataset(dev_questions)
    X_test, y_test = get_dataset(test_questions)

    id_to_word = dict([(v, k) for k, v in word_to_id.iteritems()])

    y_train_cat = np_utils.to_categorical(y_train, nb_classes=num_words)
    y_dev_cat = np_utils.to_categorical(y_dev, nb_classes=num_words)
    y_test_cat = np_utils.to_categorical(y_test, nb_classes=num_words)

    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')

    print("Pad sequences (samples x time)")
    X_train = sequence.pad_sequences(X_train)
    X_test = sequence.pad_sequences(X_test)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    print('Build model...')
    batch_size = 10
    in_embedding_size = 100
    out_embedding_size = 100

    model = Sequential()
    model.add(Embedding(num_words, in_embedding_size))
    model.add(LSTM(out_embedding_size))
    model.add(Dropout(0.5))
    model.add(Dense(num_words))
    model.add(Activation('softmax'))

    #sgd_optimizer = SGD(lr=0.006, momentum=0.9, decay=0.99, nesterov=True)
    #adg_optimizer = Adagrad()
    rms_optimizer = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms_optimizer,metrics=['accuracy'])

    print("Train...")
    model.fit(X_train, y_train_cat, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_dev,y_dev_cat))
    score = model.evaluate(X_test, y_test_cat, batch_size=batch_size)
    print('Test score:', score)

    classes_proba = model.predict_proba(X_test, batch_size=batch_size)
    for i in range(5):
        probs = sorted(zip(range(len(classes_proba)), classes_proba[i].tolist()), key=lambda x: x[1], reverse=True)
        print('Test sample %d (Correct label: %s)' % (i, id_to_word[y_test[i]]))
        for j, p in probs[:5]:
            print(id_to_word[j].ljust(20) + ': ' + str(p))

    classes = np_utils.probas_to_classes(classes_proba)

    correct, wrong = 0, 0
    for (i,q) in enumerate(test_questions):
        options = q[5]
        options_probs = classes_proba[i][options]
        best_idx = np.argmax(options_probs)
        predicted = options[best_idx]
        print('Test sample %d (Correct label: %s)' % (i, id_to_word[y_test[i]]))
        for k in range(len(options)):
            print(id_to_word[options[k]].ljust(20) + ': ' + str(options_probs[k]))

        if predicted == y_test[i]:
            correct += 1
        else:
            wrong += 1

    print('%d correct, %d wrong' % (correct, wrong))
