#BASE=/path/to/base/directory
#SCRIPT=/path/to/scripts/for/gule/experiments
#TLM=/path/to/pretrained/TLM/directory
#DATA=/path/to/GLUE/data/directory
#OUTPUT=/path/to/output/directory

BASE=../..
SCRIPT=$BASE/src/examples.glue/
DATA=$BASE/dataset/GLUE
TLM=$BASE/pretrained_models/albert-xxlarge-v2
OUTPUT=vocab
TASK=$1
task="$(tr [A-Z] [a-z] <<< "$TASK")"

## mkdir for output directory
if [ ! -d "$OUTPUT" ]; then
	mkdir $OUTPUT
fi;

CUDA_VISIBLE_DEVICES=0 python $SCRIPT/make_glue_cnn_vocab.py  \
  --model_name_or_path $TLM \
  --do_lower_case  \
  --data_dir $DATA/$TASK \
  --max_seq_length 128 \
  --output_dir $OUTPUT \
  --ostem $TASK \
  --model_type albert \
  --task_name $task \
