# CS224D_Project
cs224d project 2016 spring

##
mctest_memnn.py requires theano.config.floatX=float32, can be done in two ways
1. echo -e "\n[global]\nfloatX=float32\n" >> ~/.theanorc
this permanently create the .theanorc config file in $HOME directory (if you have not setup the file before it likely does not exist). Can also use this file to specify GPU

2. sudo THEANO_FLAGS='floatX=float32' python mctest_memnn.py data/MCTest/mc160.train.pickle 1 50
use the flag to run program each time. this will not affact other programs running on the theano backend

## MCTest statistics
➜  data_utils git:(master) sudo python mctest_parse.py ../data/MCTest mc500
Train file: mc500.train.tsv
Parsing questions ../data/MCTest/mc500.train.tsv ../data/MCTest/mc500.train.ans
There are 1200 questions
There are 5697 statements
There are 3974 words
Ignored 0 questions which had more than 1 word answers
Ignored 0 questions which had an unknown answer word
Max statement length:  46
Max number of statements:  55
Final processing...
Parsing questions ../data/MCTest/mc500.dev.tsv ../data/MCTest/mc500.dev.ans
There are 200 questions
There are 1002 statements
There are 4252 words
Ignored 0 questions which had more than 1 word answers
Ignored 0 questions which had an unknown answer word
Max statement length:  52
Max number of statements:  59
Final processing...
Parsing questions ../data/MCTest/mc500.test.tsv ../data/MCTest/mc500.test.ans
There are 600 questions
There are 2778 statements
There are 4820 words
Ignored 0 questions which had more than 1 word answers
Ignored 0 questions which had an unknown answer word
Max statement length:  48
Max number of statements:  44
Final processing...
Pickling train... mc500.train.pickle
Pickling train... mc500.dev.pickle
Pickling test... mc500.test.pickle
Pickling stop words... stopwords.pickle

➜  data_utils git:(master) sudo python mctest_parse.py ../data/MCTest mc160
Train file: mc160.train.tsv
Parsing questions ../data/MCTest/mc160.train.tsv ../data/MCTest/mc160.train.ans
There are 280 questions
There are 1386 statements
There are 2086 words
Ignored 0 questions which had more than 1 word answers
Ignored 0 questions which had an unknown answer word
Max statement length:  45
Max number of statements:  42
Final processing...
Parsing questions ../data/MCTest/mc160.dev.tsv ../data/MCTest/mc160.dev.ans
There are 120 questions
There are 567 statements
There are 2515 words
Ignored 0 questions which had more than 1 word answers
Ignored 0 questions which had an unknown answer word
Max statement length:  34
Max number of statements:  48
Final processing...
Parsing questions ../data/MCTest/mc160.test.tsv ../data/MCTest/mc160.test.ans
There are 240 questions
There are 1101 statements
There are 3151 words
Ignored 0 questions which had more than 1 word answers
Ignored 0 questions which had an unknown answer word
Max statement length:  54
Max number of statements:  41
Final processing...
Pickling train... mc160.train.pickle
Pickling train... mc160.dev.pickle
Pickling test... mc160.test.pickle
Pickling stop words... stopwords.pickle

## How to run mctest:
1. data preparation
cd to data_utils
python mctest_parse ../data/MCTest mc160
python mctest_parse ../data/MCTest mc500

2. baseline, lstm and memnn
python mctest_baseline.py data/MCTest/mc500.train.pickle
python mctest_lstm.py data/MCTest/mc500.train.pickle [# of epoch]
python mctest_memnn.py data/MCTest/mc500.train.pickle [# of epoch] [embedding size]

change mc500 to mc160 for the other set

##
return by parse_mc_test_dataset:
def parse_mc_test_dataset(questions_file, answers_file, word_id=0, word_to_id={}, update_word_ids=True, pad=True, add_pruning=False):
return dataset, questions_seq, word_to_id, word_id, null_word_id, max_stmts, max_words
dataset: a list of  (list of tokens in each story)

##
data structure of "question" used in MCtest
question[0]: number of questions
question[1]: -1
question[2]: list of (list of indexed tokens in each sentence)
question[3]: list of indexed token for question
question[4]: int of 0-3 indicating correct answer
question[5]: list of (list of indexed tokens in option)
