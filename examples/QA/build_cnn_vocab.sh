#BASE=/path/to/base/directory
#SCRIPT=/path/to/scripts/for/qa/experiments
#TLM=/path/to/pretrained/TLM/directory
#DATA=/path/to/QA/data/directory
#OUTPUT=/path/to/output/directory

BASE=../..
SCRIPT=$BASE/src/examples.openqa/
DATA=$BASE/dataset/openqa
TLM=$BASE/pretrained_models/albert-xxlarge-v2
OUTPUT=vocab
QA_DSET=$1

## mkdir for output directory
if [ ! -d "$OUTPUT" ]; then
	mkdir $OUTPUT
fi;


CUDA_VISIBLE_DEVICES=0 python $SCRIPT/make_openqa_cnn_vocab.py  \
     --model_name_or_path $TLM \
     --model_type albert  \
     --do_lower_case  \
     --train_file $DATA/${QA_DSET}_train.npm.json  \
     --dev_file $DATA/${QA_DSET}_dev.npm.json \
     --predict_file $DATA/${QA_DSET}_test.npm.json \
     --max_seq_length 384  \
     --doc_stride 128  \
     --output_dir $OUTPUT \
     --ostem $QA_DSET \
