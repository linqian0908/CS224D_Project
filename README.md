# CS224D_Project
cs224d project 2016 spring

##
mctest_memnn.py requires theano.config.floatX=float32, can be done in two ways
1. echo -e "\n[global]\nfloatX=float32\n" >> ~/.theanorc
this permanently create the .theanorc config file in $HOME directory (if you have not setup the file before it likely does not exist). Can also use this file to specify GPU

2. sudo THEANO_FLAGS='floatX=float32' python mctest_memnn.py data/MCTest/mc160.train.pickle 1 50
use the flag to run program each time. this will not affact other programs running on the theano backend

##
How to run mctest:
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
