# CS224D_Project
cs224d project 2016 spring

mctest_memnn.py requires theano.config.floatX=float32, can be done in two ways
1. echo -e "\n[global]\nfloatX=float32\n" >> ~/.theanorc
this permanently create the .theanorc config file in $HOME directory (if you have not setup the file before it likely does not exist). Can also use this file to specify GPU

2. sudo THEANO_FLAGS='floatX=float32' python mctest_memnn.py data/MCTest/mc160.train.pickle 1 50
use the flag to run program each time. this will not affact other programs running on the theano backend

