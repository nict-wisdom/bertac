#BASE=/path/to/base/directory
#SCRIPT=/path/to/scripts/for/gule/experiments
#DATA=/path/to/GLUE/data/directory
#TLM=/path/to/pretrained/TLM/directory
#CNN=/path/to/pretrained/CNN/directory
#CNN_VOCAB=/path/to/cnn/vocabulary/directory
#fastText=/path/to/fastText/embedding/vector/file
#OUTPUT=/path/to/cached/data/directory

BASE=../..
SCRIPT=$BASE/src/examples.glue/
DATA=$BASE/dataset/GLUE
TLM=$BASE/pretrained_models/albert-xxlarge-v2
CNN=$BASE/cnn_models
CNN_VOCAB=vocab
fastText=$BASE/fastText/fastText.enwiki.300d.txt
OUTPUT=cached
TASK=$1
task="$(tr [A-Z] [a-z] <<< "$TASK")"

## mkdir for cache directory
cd cached
sh mkdir.sh
cd ..

CUDA_VISIBLE_DEVICES=0 python $SCRIPT/run_glue_preprocess.py \
     --model_type albert \
     --model_name_or_path $TLM\
     --config_name $TLM/config.json\
     --do_lower_case \
     --data_dir $DATA/$TASK\
     --max_seq_length 128 \
     --cnn_model $CNN/cnn_1.2.3.4.100.pt \
     --cnn_stem enwiki \
     --emb_file $fastText\
     --emb_dim 300 \
     --prep_vocab_file $CNN_VOCAB/${TASK}_ot_vocab_file.bin \
     --feat_dir $OUTPUT \
     --overwrite_cache \
     --task_name $task \
