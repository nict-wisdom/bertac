#BASE=/path/to/base/directory
#QA_DSET=QA data set (quasart or searchqa)
#TSET=dataset for evaluation (dev or test)
#ASSM=/model/name/of/answer/span/selector (e.g., albert-xxlarge-v2.${QA_DSET}.1.2.3.4.100.TIER3.1e-05.e2)
#PSM=/model/name/of/passage/selector (e.g., albert-xxlarge-v2.${QA_DSET}.1.2.3.4.100.TIER3.1e-05.e2)
#MODEL=/path/to/trained/model's/directory 
#GT_DIR=/path/to/directory/for/ground/truth/file (e.g., ../../dataset/openqa_preprocess/download)
#SCRIPT=/path/to/directory/for/evaluation/scripts/


QA_DSET=$1
PSM=$2
ASSM=$3
TSET=$4

BASE=../../
GT_DIR=$BASE/dataset/openqa_preprocess/download
MODEL=model
SCRIPT=eval_scripts


## mkdir for output directory
cd extracted_answers
sh mkdir.sh
cd ..

## JSON to TXT conversion for Answer span selector  
echo "python $SCRIPT/nbestjson2txt.py  --data_dir $MODEL/reader/$ASSM --input_file nbest_predictions_test.json --output_file 3best_predictions_test.txt"
python $SCRIPT/nbestjson2txt.py  \
   --data_dir $MODEL/reader/$ASSM \
   --input_file nbest_predictions_test.json  \
   --output_file 3best_predictions_test.txt \


## Answer extraction by aggregating results of passage selector ($PSM) and answer span selector ($ASSM)
perl $SCRIPT/answer_extraction.pl $QA_DSET $MODEL $PSM $ASSM test

## Evaluating the results
GT_FILE=$GT_DIR/$QA_DSET/$TSET.txt
EXTRACTED_ANSWERS=extracted_answers/$TSET/$QA_DSET/$PSM.$ASSM.out
EVAL_OUT=extracted_answers/$TSET/$QA_DSET/$PSM.$ASSM.eval
python $SCRIPT/evaluate.py --gt $GT_FILE --pred $EXTRACTED_ANSWERS > $EVAL_OUT
