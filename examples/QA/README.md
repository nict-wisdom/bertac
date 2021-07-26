<a name="top"></a> Open-domain QA experiments
================
You can run open-domain QA experiments with BERTAC from the command line. This example is for training BERTAC QA models for Quasar-T and SearchQA benchmarks. 

## Run open-domain QA example 
1. [Get the data](#get_the_data)
2. [Convert the data to the BERTAC format](#convert_the_data)
3. [Build CNN vocabularies](#build_cnn_vocab)
4. [Cache the train/dev data](#caching)
5. [Training](#training)
6. [Prediction](#prediction)
7. [Answer extraction and evaluation](#answer_extraction)

This example is built on the modified version of `examples/run_squad.py` in the HuggingFace Transformers version 2.4.1 that is originally designed for training/evaluating a SQuAD QA model. The modified code works for training/evaluating a BERTAC QA model that exploits a *passage selector* and an *answer span selector* for extracting answers from retrieved passages.  

### <a name="get_the_data"></a> 1. Get the data 
This example starts from downloading the Quasar-T and SearchQA datasets. 

```bash
cd dataset/openqa_preprocess
sh ./download_data.sh  
```
The script shown above downloads these datasets and stores them into `dataset/openqa_preprocess/{quasart,searchqa}`. It also downloads the [Stanford CoreNLP library](https://stanfordnlp.github.io/CoreNLP/) (stanford-corenlp-full-2017-06-09) that is used for conversion of the data into the BERTAC format in the next step. 

### <a name="convert_the_data"></a> 2. Convert the data to the BERTAC format 

```bash
sh ./convert_to_bertac_format.sh
cd ../../examples/QA
```

As the first step of preprocessing, the original Quasar-T and SearchQA data are converted to the BERTAC format. The resulting train/dev/test data are stored in `dataset/openqa`.

### <a name="build_cnn_vocab"></a> 3. Build CNN vocabularies

Then, CNN vocabularies are built from each dataset (Quasar-T or SearchQA) for efficiency in training and evaluating BERTAC QA models.

```bash
BASE=/path/to/base/directory
SCRIPT=/path/to/scripts/for/qa/experiments
TLM=/path/to/pretrained/TLM/directory
DATA=/path/to/QA/data/directory
OUTPUT=/path/to/output/directory

QA_DSET=$1

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


```
or

```
sh build_cnn_vocab.sh quasart
```

You can use the following arguments to run this script. 
* `--model_name_or_path`: the directory of the pretrained ALBERT or RoBERTa (the current code supports ALBERT and RoBERTa).
* `--model_type`: the pretrained TLM model type (Since the source code supports ALBERT and RoBERTa, it should be `albert` or `roberta`). 
* `--do_lower_case`: set the flag if you are using an uncased model.
* `--train_file`: the file name of the training data.
* `--dev_file`: the file name of the development data.
* `--predict_file`: the file name of the test data.
* `--max_seq_length`: the maximum input sequence length after tokenization. Input sequences longer than this number will be truncated, those shorter will be padded.
* `--doc_stride`: the number of strides between chunks when splitting up a long document into chunks. 
* `--output_dir`: the directory used to store the resulting CNN vocabulary file.
* `--ostem`: the identifier of the resulting CNN vocabulary file. The CNN vocabulary file is store with the name `${ostem}_ot_vocab_file.bin` into the directory specified by `--output_dir`.

### <a name="caching"></a> 4. Cache the train/dev/test data
For efficiency, all the data is cached before training and prediction.
 
```bash
BASE=/path/to/base/directory
SCRIPT=/path/to/scripts/for/qa/experiments
DATA=/path/to/QA/data/directory
TLM=/path/to/pretrained/TLM/directory
CNN=/path/to/pretrained/CNN/directory
CNN_VOCAB=/path/to/cnn/vocabulary/directory
fastText=/path/to/fastText/embedding/vector/file
OUTPUT=/path/to/cached/data/directory

TQA_DSET=$1


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

```

This script takes the same arguments related to TLMs and QA data as those used in the script of [Step 3: Build CNN vocabularies](#build_cnn_vocab) and additionally takes the following ones.
* `--config_name`: the config file name of ALBERT or RoBERTa.
* `--cnn_model`: the file name of a pretrained CNN model. 
* `--emb_file`: the file name of the fastText embedding vectors. 
* `--prep_vocab_file`: the CNN vocabulary file of the QA dataset specified by `$QA_DSET`, i.e., the output of [Step 3: Build CNN vocabularies](#build_cnn_vocab). 
* `--feat_dir`: the directory used to store resulting cached data.
* `--overwrite_cache`: overwrite the existing cached data if this flag is set. 
   
The file name of pretrained CNNs we provided is in the format of `cnn_{filter_windows}.{number_of_filters}.pt` (e.g., `cnn_1.2.3.4.100.pt`), where `filter_windows` (`1.2.3.4`) and `number_of_filters` (`100`) are the hyperparameters used for pretraining the CNN.

### <a name="training"></a> 5. Training
As mentioned above, the BERTAC QA model exploits a *passage selector* to choose
relevant passages from retrieved passages and an *answer span selector* to identify the answer span in the selected passages. These two selectors are independently trained and their prediction results are combined to extract answers from given passages.

`run_openqa_selector.py` and `run_openqa_reader.py` used in Steps 5 and 6 inherit the arguments related to TLMs, QA data, and pretrained CNNs from the script `run_openqa_preprocess.py` used in [Step 4: Cache the train/dev/test data](#caching). 


#### <a name="training_ps"></a> 5.1. Passage selector
```bash
BASE=/path/to/base/directory
SCRIPT=/path/to/scripts/for/QA/experiments
DATA=/path/to/QA/data/directory
TLM=/path/to/pretrained/TLM/directory
CNN=/path/to/pretrained/CNN/directory
CNN_VOCAB=/path/to/cnn/vocabulary/directory
fastText=/path/to/fastText/embedding/vector/file
CACHE=/path/to/cached/data/directory
OUTPUT=/path/to/trained/model/directory

QA_DSET=$1

CUDA_VISIBLE_DEVICES=0 python $SCRIPT/run_openqa_selector.py  \
     --model_type albert  \
     --model_name_or_path $TLM \
     --config_name $TLM/config.json \
     --do_train   \
     --do_lower_case  \
     --train_file $DATA/${QA_DSET}_train.npm.json \
     --max_seq_length 384  \
     --doc_stride 128  \
     --cnn_model $CNN/cnn_1.2.3.4.100.pt   \
     --emb_file $fastText \
     --prep_vocab_file $CNN_VOCAB/${QA_DSET}_ot_vocab_file.bin  \
     --learning_rate 1e-05  \
     --num_train_epochs 2  \
     --max_seq_length 384  \
     --doc_stride 128  \
     --per_gpu_train_batch_size 6  \
     --output_dir $OUTPUT/albert-xxlarge-v2.${QA_DSET}.1.2.3.4.100.TIER3.1e-05.e2  \
     --overwrite_output_dir   \
     --num_of_TIERs 3  \
     --feat_dir $CACHE \
     --gradient_accumulation_steps 2  \
     --fp16  \

```

* `--do_train`: run training if this flag is set.
* `--learning_rate`: the learning rate of training. 
* `--num_train_epochs`: the number of training epochs.
* `--per_gpu_train_batch_size`: the batch size for training.
* `--output_dir`: the directory used to store the resulting model.
* `--overwrite_output_dir`: if this flag is set, the existing output model and evaluation data in the directory specified by `--output_dir` is overwritten.
* `--num_of_TIERs`: the number of TIER layers.
* `--feat_dir`: the directory of cached data.
* `--gradient_accumulation_steps`: the number of update steps to accumulate before performing a backward/update pass.
* `--fp16`: use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit if this flag is set. 

In the above example, the passage selector is trained (`--do_train`) under the following  settings: TIER layer of 3 (`--num_of_TIERs 3`), learning rate of 1e-05 (`--learning rate 1e-05`), batch size of 6 (`--per_gpu_train_batch_size 6`), pretrained CNN model of cnn_1.2.3.4.100.pt (`--cnn_model`), and six training-epochs. The trained model is stored in the directory specified by `--output_dir`.

#### <a name="training_as"></a> 5.2. Answer span selector 
 
```bash
BASE=/path/to/base/directory
SCRIPT=/path/to/scripts/for/QA/experiments
DATA=/path/to/QA/data/directory
TLM=/path/to/pretrained/TLM/directory
CNN=/path/to/pretrained/CNN/directory
CNN_VOCAB=/path/to/cnn/vocabulary/directory
fastText=/path/to/fastText/embedding/vector/file
CACHE=/path/to/cached/data/directory
OUTPUT=/path/to/trained/model/directory

QA_DSET=$1

CUDA_VISIBLE_DEVICES=0 python $SCRIPT/run_openqa_reader.py  \
     --model_type albert  \
     --model_name_or_path $TLM \
     --config_name $TLM/config.json \
     --do_train   \
     --do_lower_case  \
     --train_file $DATA/${QA_DSET}_train.npm.json \
     --max_seq_length 384  \
     --doc_stride 128  \
     --cnn_model $CNN/cnn_1.2.3.4.100.pt   \
     --emb_file $fastText \
     --prep_vocab_file $CNN_VOCAB/${QA_DSET}_ot_vocab_file.bin  \
     --learning_rate 1e-05  \
     --num_train_epochs 2  \
     --max_seq_length 384  \
     --doc_stride 128  \
     --output_dir $OUTPUT/albert-xxlarge-v2.${QA_DSET}.1.2.3.4.100.TIER3.1e-05.e2  \
     --per_gpu_train_batch_size 6  \
     --overwrite_output_dir   \
     --num_of_TIERs 3  \
     --feat_dir $CACHE \
     --gradient_accumulation_steps 2  \
     --fp16  \

```

As shown above, the answer span selector can be trained with the same setting as that for training a passage selector in [Step 5.1](#training_ps). 

**Note**: Though the examples for training the two selectors shown above are in a single-GPU setting, it is strongly recommended to run the scripts in a multi-GPU setting for data-parallel. It took around 2 days with eight V100 GPUs to run the script for training a passage selector in [Step 5.1](#training_ps) when using SearchQA training data (`QA_DSET=searchqa`) and ALBERT-xxlarge as a base TLM. 

### <a name="prediction"></a> 6. Prediction
Similarly to the training, prediction by the passage selection and answer span selection models is done independently. Their prediction results are then combined in the next step: [Step 7. Answer extraction and evaluation](#evaluation) to extract answers.

#### 6.1. Passage selector

```bash
BASE=/path/to/base/directory
SCRIPT=/path/to/scripts/for/QA/experiments
DATA=/path/to/QA/data/directory
CNN=/path/to/pretrained/CNN/directory
CNN_VOCAB=/path/to/cnn/vocabulary/directory
fastText=/path/to/fastText/embedding/vector/file
CACHE=/path/to/cached/data/directory
OUTPUT=/path/to/trained/model/directory

QA_DSET=$1

CUDA_VISIBLE_DEVICES=0 python $SCRIPT/run_openqa_selector.py  \
     --model_type albert  \
     --model_name_or_path $OUTPUT/albert-xxlarge-v2.toy3.1.2.3.4.100.TIER3.1e-05.e2 \
     --config_name $OUTPUT/albert-xxlarge-v2.${QA_DSET}.1.2.3.4.100.TIER3.1e-05.e2/config.json \
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
     --per_gpu_eval_batch_size 80  \
     --overwrite_output_dir   \
     --num_of_TIERs 3  \
     --feat_dir $CACHE \
     --fp16  \

```

* `--do_eval`: run prediction if this flag is set.

The above example for prediction by a passage selector takes arguments similar to its training. Instead of the arguments required for training only (including `--do_train`, `--train_file`, `--learning_rate`, `--training_epochs`, `--per_gpu_train_batch_size`, and `--gradient_accumulation_steps`), the script for prediction takes `--do_eval`, `--predict_file`, and `--per_gpu_train_batch_size` as arguments. 

The prediction result is stored in the directory specified by `--output_dir` with the file name `predictions_test.json`. Note that prediction results on the development data can be obtained by setting `--predict_file` with the file name of the development data and activating the mode for prediction on the development data with `--pred_dev`. In this case, the result is stored with the name `predictions_dev.json`.

#### 6.2. Answer span selector

```bash
BASE=/path/to/base/directory
SCRIPT=/path/to/scripts/for/QA/experiments
DATA=/path/to/QA/data/directory
CNN=/path/to/pretrained/CNN/directory
CNN_VOCAB=/path/to/cnn/vocabulary/directory
fastText=/path/to/fastText/embedding/vector/file
CACHE=/path/to/cached/data/directory
OUTPUT=/path/to/trained/model/directory

QA_DSET=$1

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
     --fp16  \
```

As in the previous example for prediction by a passage selector, the above example takes `--do_eval`, `--predict_file`, and `--per_gpu_train_batch_size` as arguments. The prediction result is stored in the directory specified by `--output_dir` with the name `nbest_predictions_test.json`. Prediction results on the development data can be obtained in the same way as that explained in the passage selector: `--pred_dev` and `--predict_file`. 

### <a name="answer_extraction"></a> 7. Answer extraction and evaluation


```bash
STEM={quasart|searchqa}
ASSM=answer_span_selector_model_name 
    (e.g., albert-xxlarge-v2.${STEM}.1.2.3.4.100.TIER3.1e-05.e2)
PSM=passage_selector_model_name 
    (e.g., albert-xxlarge-v2.${STEM}.1.2.3.4.100.TIER3.1e-05.e2)
MODEL=/path/to/trained/model/directory 
    (e.g., ./model)
GT_DIR=/path/to/ground/truth/file/directory 
    (e.g., ../../dataset/openqa_preprocess/download)
SCRIPT=eval_scripts

## JSON to TXT conversion for Answer span selector
python $SCRIPT/nbestjson2txt.py  \
   --data_dir $MODEL/reader/$ASSM \
   --input_file nbest_predictions_test.json  \
   --output_file 3best_predictions_test.txt \

## Answer extraction using the prediction results of the two selectors
perl $SCRIPT/answer_extraction.pl $STEM $MODEL $PSM $ASSM test

## Evaluation 
GT_FILE=$GT_DIR/$STEM/test.txt
EXTRACTED_ANSWERS=extracted_answers/test/$STEM/$PSM.$ASSM.out
EVAL_OUT=extracted_answers/test/$STEM/$PSM.$ASSM.eval
python $SCRIPT/evaluate.py --gt $GT_FILE --pred $EXTRACTED_ANSWERS > $EVAL_OUT

```

In the above script, `nbestjson2txt.py` converts the JSON file containing the prediction output by a given answer span selector into plain text. Then `answer_extraction.pl` extracts answers for each question by aggregating the given prediction results of passage selector and answer span selector. The answer extraction process by `answer_extraction.pl` first computes the probability of 
<img src="https://render.githubusercontent.com/render/math?math=\color{tan}Pr(a|q,P) = \sum_{i} Pr(a|q,p_i) Pr(p_i|q,P)">, where <img src="https://render.githubusercontent.com/render/math?math=\color{tan}Pr(a|q,p_i)"> and <img src="https://render.githubusercontent.com/render/math?math=\color{tan}Pr(p_i|q,P)">  are the probabilities given by a passage selector and an answer span selector, respectively. Then it extracts answer *a* for question *q* from a set of passages *P* that maximizes the probability <img src="https://render.githubusercontent.com/render/math?math=\color{tan}Pr(a|q,P)">. Finally, `evaluate.py` evaluates the resulting extracted answers by comparing them to the gold standard answers. 



The extracted answers and evaluation results are stored into the following files:

* extracted answers: `extracted_answers/{dev,test}/{quasart,searchqa}/$PSM.$ASSM.out`
* evaluation results: `extracted_answers/{dev,test}/{quasart,searchqa}/$PSM.$ASSM.eval`

