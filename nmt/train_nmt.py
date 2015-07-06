import numpy as np
import fcntl  # copy
import itertools
import sys, os
import argparse
import time
import datetime
from nmt import train

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-dw', '--dim_word', required=False, default='50', help='Size of the word representation')
parser.add_argument('-d', '--dim_model', required=False, default='200', help='Size of the hidden representation')
parser.add_argument('-data', '--dataset', required=False, default='sub_europarl', help='ex: sub_europarl, europarl')



args = parser.parse_args()

dim_word = int(args.dim_word)
dim_model = int(args.dim_model)
dataset = args.dataset






#Create names and folders
####################################################################################
dirPath = 'saved_models/'
if not os.path.exists(dirPath):
    try:
        os.makedirs(dirPath)
    except OSError as e:
        print e
        print 'Exeption was catch, will continue script \n'

if dataset == "sub_europarl":
    dirModelName = "model_gru_sub_europarl_enfr_" + "_".join([str(dim_word), str(dim_model)])
elif dataset == "europarl":
    dirModelName = "model_gru_europarl_enfr_" + "_".join([str(dim_word), str(dim_model)])
elif dataset == "small_europarl_enfr":
    dirModelName = "model_gru_small_europarl_enfr_" + "_".join([str(dim_word), str(dim_model)])
elif dataset == "wmt_all_enfr":
    dirModelName = "model_gru_wmt_all_enfr_" + "_".join([str(dim_word), str(dim_model)])
elif dataset == "de_en":
    dirModelName = "model_gru_statmt_deen_" + "_".join([str(dim_word), str(dim_model)])
else:
    sys.exit("Wrong dataset")

dirPath = os.path.join(dirPath, dirModelName)
if not os.path.exists(dirPath):
    try:
        os.makedirs(dirPath)
    except OSError as e:
        print e
        print 'Exeption was catch, will continue script \n'

modelName = dirPath = os.path.join(dirPath, dirModelName + ".npz")

###################################################################################


if dataset == "sub_europarl":
    n_words_src = 1025
    n_words_trg = 1153
    dataset = 'stan'
    dictionary_trg='../../data/vocab_and_data_wmt_all_enfr/vocab_sub_europarl.fr.pkl'
    dictionary_src='../../data/vocab_and_data_wmt_all_enfrx/vocab_sub_europarl.en.pkl'

    batch_size = 64
    nb_batch_epoch = 4

elif dataset == "europarl":
    n_words_src=30000
    n_words_trg=30000
    dictionary_trg='../../data/vocab_and_data_europarl/vocabEuroparl.fr.pkl'
    dictionary_src='../../data/vocab_and_data_europarl/vocabEuroparl.en.pkl'

    sizeTrainset = 1405407.0
    batch_size = 64
    nb_batch_epoch = np.ceil(sizeTrainset/batch_size)

elif dataset == "small_europarl_enfr":
    n_words_src=30000
    n_words_trg=30000
    dictionary_trg='../../data/vocab_and_data_small_europarl_v7_enfr/vocab.fr.pkl'
    dictionary_src='../../data/vocab_and_data_small_europarl_v7_enfr/vocab.en.pkl'

    sizeTrainset = 500000.0
    batch_size = 64
    nb_batch_epoch = np.ceil(sizeTrainset/batch_size)
    
elif dataset == "wmt_all_enfr":
    n_words_src=30000
    n_words_trg=30000
    dictionary_trg='../../data/vocab_and_data_wmt_all_enfr/vocab_wmt_all.fr.pkl'
    dictionary_src='../../data/vocab_and_data_wmt_all_enfr/vocab_wmt_all.en.pkl'

    sizeTrainset = 12075604.0
    batch_size = 64
    nb_batch_epoch = np.ceil(sizeTrainset/batch_size)

elif dataset == "de_en":
    n_words_src=30000
    n_words_trg=30000
    dictionary_trg='../../data/vocab_and_data_de_en/vocab.en.pkl'
    dictionary_src='../../data/vocab_and_data_de_en/vocab.de.pkl'

    sizeTrainset = 4535522.0
    batch_size = 64
    nb_batch_epoch = np.ceil(sizeTrainset/batch_size)

trainerr, validerr, testerr = train(saveto=modelName,
                                    reload_=False,
                                    dim_word=dim_word,
                                    dim=dim_model,
                                    encoder='gru',
                                    decoder='gru_cond_simple',
                                    hiero=None, #'gru_hiero', # or None
                                    max_epochs=100,
                                    n_words_src=n_words_src,
                                    n_words=n_words_trg,
                                    optimizer='adadelta',
                                    decay_c=0.,
                                    alpha_c=0.,
                                    diag_c=0.,# not used with adadelta
                                    lrate=0.,
                                    patience=10,
                                    maxlen=50,
                                    batch_size=batch_size,
                                    valid_batch_size=batch_size,
                                    validFreq=nb_batch_epoch, # freq in batch of computing cost for train, valid and test
                                    dispFreq=nb_batch_epoch, # freq of diplaying the cost of one batch (e.g.: 1 is diplaying the cost of each batch)
                                    saveFreq=nb_batch_epoch, # freq of saving the model per batch
                                    sampleFreq=nb_batch_epoch, # freq of sampling per batch
                                    dataset=dataset,
                                    dictionary=dictionary_trg,
                                    dictionary_src=dictionary_src,
                                    use_dropout=False)
