#! /usr/bin/sh
python preprocess.py -train_src data/src_train.txt -train_tgt data/tgt_train.txt -valid_src data/src_dev.txt -valid_tgt data/tgt_dev.txt -save_data data/demo -train_cue data/cue_train.txt -valid_cue data/cue_dev.txt
