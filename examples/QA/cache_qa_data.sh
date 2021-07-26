#BASE=/path/to/base/directory
#SCRIPT=/path/to/scripts/for/gule/experiments
#DATA=/path/to/GLUE/data/directory
#TLM=/path/to/pretrained/TLM/directory
#CNN=/path/to/pretrained/CNN/directory
#CNN_VOCAB=/path/to/cnn/vocabulary/directory
#fastText=/path/to/fastText/embedding/vector/file
#OUTPUT=/path/to/cached/data/directory

BASE=../..
SCRIPT=$BASE/src/examples.openqa/
DATA=$BASE/dataset/openqa
TLM=$BASE/pretrained_models/albert-xxlarge-v2
CNN=$BASE/cnn_models
CNN_VOCAB=vocab
fastText=$BASE/fastText/fastText.enwiki.300d.txt
OUTPUT=cached

QA_DSET=$1

## mkdir for cache directory
cd cached
sh mkdir.sh
cd ..


CUDA_VISIBLE_DEVICES=0 python $SCRIPT/run_openqa_preprocess.py  \
     --model_type albert  \
     --model_name_or_path $TLM \
     --config_name $TLM/config.json \
     --do_lower_case  \
     --train_file $DATA/${QA_DSET}_train.npm.json \
     --dev_file $DATA/${QA_DSET}_dev.npm.json \
     --predict_file $DATA/${QA_DSET}_test.npm.json \
     --max_seq_length 384  \
     --doc_stride 128  \
     --cnn_model $CNN/cnn_1.2.3.4.100.pt   \
     --emb_file $fastText \
     --prep_vocab_file $CNN_VOCAB/${QA_DSET}_ot_vocab_file.bin  \
     --feat_dir $OUTPUT \
     --threads 24  \
     --overwrite_cache \
