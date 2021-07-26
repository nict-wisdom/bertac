GLUE experiments
================
You can run GLUE experiments with BERTAC from the command line. This example is for training BERTAC models for eight GLUE tasks (all the tasks except for WNLI) and for evaluating them with the development data. 


## Run GLUE example

1. [Get the data](#get_the_data)
2. [Convert the data to the BERTAC format](#convert_the_data)
3. [Build CNN vocabularies](#build_cnn_vocab)
4. [Cache the train/dev data](#caching)
5. [Run GLUE tasks](#run_task)

### <a name="get_the_data"></a> 1. Get the data

```bash
cd dataset/glue_preprocess
sh download_data.sh  
```
`download_data.sh` downloads datasets of seven tasks (MNLI, QNLI, QQP, RTE, SST, CoLA, STS except for MRPC). 

For legal reasons, you need to download MRPC data manually. Please download it from the [Microsoft download center](https://www.microsoft.com/en-us/download/details.aspx?id=52398) and store it into `download/MRPC/{train,dev}.tsv`.

### <a name="convert_the_data"></a> 2. Convert the data to the BERTAC format 

```bash
sh convert_to_bertac_format.sh
cd ../../examples/GLUE
```
All the downloaded data are converted into the BERTAC format. The resulting files are stored in `dataset/GLUE/{CoLA,MNLI,MRPC,QNLI,QQP,RTE,SST-2,STS-B}`.

### <a name="build_cnn_vocab"></a> 3. Build CNN vocabularies

For efficiency in training/evaluating BERTAC models, this example first builds CNN vocabularies from the training/development data of each GLUE task. The CNN vocabularies are given to the pretrained CNN in a BERTAC model. 

```bash
BASE=/path/to/base/directory
SCRIPT=/path/to/scripts/for/gule/experiments
TLM=/path/to/pretrained/TLM/directory
DATA=/path/to/GLUE/data/directory
OUTPUT=/path/to/output/directory

TASK=STS-B
task="$(tr [A-Z] [a-z] <<< "$TASK")"

CUDA_VISIBLE_DEVICES=0 python $SCRIPT/make_glue_cnn_vocab.py  \
  --model_name_or_path $TLM \
  --model_type albert \
  --do_lower_case  \
  --task_name $task \
  --data_dir $DATA/$TASK \
  --max_seq_length 128 \
  --output_dir $OUTPUT \
  --ostem $TASK \

```
or

```bash
sh build_cnn_vocab.sh STS-B
```

You can use the following arguments to run this script. 
* `--model_name_or_path`: the directory of pretrained ALBERT or RoBERTa (the source code supports ALBERT and RoBERTa).
* `--model_type`: the pretrained TLM model type (Since the source code supports ALBERT and RoBERTa, it should be `albert` or `roberta`). 
* `--do_lower_case`: set the flag if you are using an uncased model.
* `--task_name`: the name of a GLUE task (CoLA, MRPC, etc.). 
* `--data_dir`: the directory of the GLUE dataset for the task specified by `--task_name`.
* `--max_seq_length`: the maximum input sequence length after tokenization. Input sequences longer than this number will be truncated, those shorter will be padded.
* `--output_dir`: the directory used to store resulting CNN vocabulary files.
* `--ostem`: the identifier of CNN vocabulary files. The CNN vocabulary file is store with the name `${ostem}_ot_vocab_file.bin` into the directory specified by `--output_dir`.
 
### <a name="caching"></a> 4. Cache the train/dev data
For efficiency, all the GLUE data are cached before training and evaluation.
 
```bash
BASE=/path/to/base/directory
SCRIPT=/path/to/scripts/for/gule/experiments
DATA=/path/to/GLUE/data/directory
TLM=/path/to/pretrained/TLM/directory
CNN=/path/to/pretrained/CNN/directory
CNN_VOCAB=/path/to/cnn/vocabulary/directory
fastText=/path/to/fastText/embedding/vector/file
OUTPUT=/path/to/cached/data/directory

TASK=STS-B
task="$(tr [A-Z] [a-z] <<< "$TASK")"

CUDA_VISIBLE_DEVICES=0 python $SCRIPT/run_glue_preprocess.py \
     --model_type albert \
     --model_name_or_path $TLM\
     --config_name $TLM/config.json\
     --do_lower_case \
     --data_dir $DATA/$TASK\
     --max_seq_length 128 \
     --cnn_model $CNN/cnn_1.2.3.4.100.pt \
     --emb_file $fastText\
     --prep_vocab_file $CNN_VOCAB/${TASK}_ot_vocab_file.bin \
     --feat_dir $OUTPUT \
     --overwrite_cache \
     --task_name $task \

```

This script takes the same arguments related to TLMs as those used in the script `make_glue_cnn_vocab.py` used in [Step 3: Build CNN vocabularies](#build_cnn_vocab) and additionally takes the following ones.
* `--config_name`: the configuration file name of ALBERT or RoBERTa.
* `--cnn_model`: the file name of the pretrained CNN model. 
* `--emb_file`: the file name of the fastText embedding vectors. 
* `--prep_vocab_file`: the CNN vocabulary file of the task specified by `--task_name`. The vocabulary file is the one that was built by [Step 3: Build CNN vocabularies](#build_cnn_vocab). 
* `--feat_dir`: the directory used to store resulting cached data.
* `--overwrite_cache`: overwrite the existing cached data if this flag is set. 
   
The file name of the pretrained CNN that we provided is in the format of `cnn_{filter_windows}.{number_of_filters}.pt` (e.g., `cnn_1.2.3.4.100.pt`), where `filter_windows` (`1.2.3.4`) and `number_of_filters` (`100`) are the hyperparameters used for pretraining the CNN.

### <a name="run_task"></a> 5. Run GLUE tasks 
 
```bash
BASE=/path/to/base/directory
SCRIPT=/path/to/scripts/for/gule/experiments
DATA=/path/to/GLUE/data/directory
TLM=/path/to/pretrained/TLM/directory
CNN=/path/to/pretrained/CNN/directory
CNN_VOCAB=/path/to/cnn/vocabulary/directory
fastText=/path/to/fastText/embedding/vector/file
CACHE=/path/to/cached/data/directory
OUTPUT=/path/to/outout/data/directory

TASK=$1
task="$(tr [A-Z] [a-z] <<< "$TASK")"

CUDA_VISIBLE_DEVICES=0 python $SCRIPT/run_glue.py  \
     --model_type albert  \
     --model_name_or_path $TLM \
     --config_name $TLM/config.json  \
     --do_train  \
     --do_eval  \
     --do_lower_case  \
     --data_dir $DATA/$TASK \
     --learning_rate 9e-06  \
     --num_train_epochs 6  \
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
```

* `--do_train`: run training if this flag is set.
* `--do_eval`: run evaluation if this flag is set.
* `--learning_rate`: the learning rate of training. 
* `--num_train_epochs`: the number of training epochs.
* `--per_gpu_train_batch_size`: the batch size for training.
* `--per_gpu_eval_batch_size`: the batch size for evaluation.
* `--overwrite_output_dir`: if this flag is set, the existing output model and evaluation data in the directory specified by `--output_dir` is overwritten.
* `--num_of_TIERs`: the number of TIER layers. 
* `--output_dir`: the directory used to store the resulting model and its evaluation result.
* `--fp16`: use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit if this flag is set. 

The above example is for training (`--do_train`) and evaluating (`--do_eval`) a BERTAC model with three TIER layers (`--num_of_TIERs 3`). Here, a learning rate of 9e-6 (`--learning_rate`), a batch size of 16 (`--per_gpu_train_batch_size` and  `--per_gpu_train_batch_size`), the pretrained CNN model  `cnn_1.2.3.4.100.pt` (`--cnn_model`), and six training-epochs are used. The trained BERTAC model and its evaluation result are stored in the directory of specified by `--output_dir`.