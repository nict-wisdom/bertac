#BASE=/path/to/base/directory
#SCRIPT=/path/to/scripts/for/gule/experiments
#DATA=/path/to/GLUE/data/directory
#TLM=/path/to/pretrained/TLM/directory
#CNN=/path/to/pretrained/CNN/directory
#CNN_VOCAB=/path/to/cnn/vocabulary/directory
#fastText=/path/to/fastText/embedding/vector/file
#CACHE=/path/to/cached/data/directory
#OUTPUT=/path/to/outout/data/directory

TASK=$1
task="$(tr [A-Z] [a-z] <<< "$TASK")"

BASE=../..
SCRIPT=$BASE/src/examples.glue/
DATA=$BASE/dataset/GLUE
TLM=$BASE/pretrained_models/albert-xxlarge-v2
CNN=$BASE/cnn_models
CNN_VOCAB=vocab
fastText=$BASE/fastText/fastText.enwiki.300d.txt
CACHE=cached
OUTPUT=model/$TASK

## mkdir for cache directory
cd model
sh mkdir.sh
cd ..



CUDA_VISIBLE_DEVICES=0 python $SCRIPT/run_glue.py  \
     --model_type albert  \
     --model_name_or_path $TLM \
     --config_name $TLM/config.json  \
     --do_train  \
     --do_eval  \
     --do_lower_case  \
     --data_dir $DATA/$TASK \
     --learning_rate 9e-06  \
     --num_train_epochs 1  \
     --max_seq_length 128  \
     --per_gpu_train_batch_size 16  \
     --per_gpu_eval_batch_size 16  \
     --overwrite_output_dir   \
     --cnn_model $CNN/cnn_1.2.3.4.100.pt   \
     --cnn_stem enwiki   \
     --emb_file $fastText \
     --emb_dim 300   \
     --num_of_TIERs 3  \
     --prep_vocab_file $CNN_VOCAB/${TASK}_ot_vocab_file.bin  \
     --feat_dir $CACHE \
     --task_name $task \
     --output_dir $OUTPUT/albert-xxlarge-v2.1.2.3.4.100.TIER3.9e-06.bs16.e6 \
     --fp16  \
     --fp16_opt_level O1 \
