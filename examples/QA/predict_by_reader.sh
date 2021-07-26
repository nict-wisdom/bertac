#BASE=/path/to/base/directory
#SCRIPT=/path/to/scripts/for/QA/experiments
#DATA=/path/to/QA/data/directory
#CNN=/path/to/pretrained/CNN/directory
#CNN_VOCAB=/path/to/cnn/vocabulary/directory
#fastText=/path/to/fastText/embedding/vector/file
#CACHE=/path/to/cached/data/directory
#OUTPUT=/path/to/outout/data/directory

QA_DSET=$1

BASE=../..
SCRIPT=$BASE/src/examples.openqa/
DATA=$BASE/dataset/openqa
CNN=$BASE/cnn_models
CNN_VOCAB=vocab
fastText=$BASE/fastText/fastText.enwiki.300d.txt
CACHE=cached
OUTPUT=model/reader

CUDA_VISIBLE_DEVICES=0 python $SCRIPT/run_openqa_reader.py  \
     --model_type albert  \
     --model_name_or_path $OUTPUT/albert-xxlarge-v2.${QA_DSET}.1.2.3.4.100.TIER3.1e-05.e2  \
     --config_name $OUTPUT/albert-xxlarge-v2.${QA_DSET}.1.2.3.4.100.TIER3.1e-05.e2 \
     --do_eval   \
     --do_lower_case  \
     --predict_file $DATA/${QA_DSET}_test.npm.json \
     --max_seq_length 384  \
     --doc_stride 128  \
     --cnn_model $CNN/cnn_1.2.3.4.100.pt   \
     --emb_file $fastText \
     --prep_vocab_file $CNN_VOCAB/${QA_DSET}_ot_vocab_file.bin  \
     --max_seq_length 384  \
     --doc_stride 128  \
     --output_dir $OUTPUT/albert-xxlarge-v2.${QA_DSET}.1.2.3.4.100.TIER3.1e-05.e2  \
     --per_gpu_eval_batch_size 80 \
     --overwrite_output_dir   \
     --num_of_TIERs 3  \
     --feat_dir $CACHE \
     --gradient_accumulation_steps 2  \
     --fp16  \
