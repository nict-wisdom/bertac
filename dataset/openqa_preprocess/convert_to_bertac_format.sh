export PYTHONIOENCODING=utf8

base_dir=`pwd`
cd ..
tgt_dir=`pwd`
cd $base_dir

script="src"
ncpu=36
TMP_DIR=tmp

echo $base_dir
cd NPMasking

## preparing output directories
if [ ! -d "$tgt_dir/openqa" ]; then
	mkdir $tgt_dir/openqa
fi;

for i in searchqa quasart tmp
do
	if [ ! -d "$i" ]; then
		mkdir $i
	fi;
done

## preparing working directories
for i in cand  masked  np  sents
do
	if [ ! -d "tmp/$i" ]; then
		mkdir tmp/$i
	fi;
done

## Convert JSON to TXT
for i in searchqa quasart 
do
   echo "python3 json2txt.py  --dataset $i"
   python3 $script/json2txt.py  --dataset $i --datatype qd --base_dir $base_dir &
   python3 $script/json2txt.py  --dataset $i --datatype qa --base_dir $base_dir &
done
wait

## Split data into smaller ones with 100,000 lines.
for i in searchqa quasart 
do
	for j in train dev test
	do
		echo "perl $script/split_data.pl $i $j $TMP_DIR/sents "
		perl $script/split_data.pl $i $j $TMP_DIR/sents &
	done
done
wait

## NP extraction using the spaCy NP chunker.
for i in searchqa quasart 
do
	echo "perl $script/np_extractor.pl $i $script $TMP_DIR $ncpu"
	perl $script/np_extractor.pl $i $script $TMP_DIR $ncpu 
done
wait

## Finding a candidate for NP-masking
for i in quasart searchqa 
do
	echo "perl $script/run_findcand.pl $i $script $TMP_DIR $ncpu"
	perl $script/run_findcand.pl $i $script $TMP_DIR $ncpu 
done
wait

## Augmenting the original sentences with NP-maksed ones
for i in quasart searchqa 
do
	echo "perl $script/run_mergemaksed.pl $i $script $TMP_DIR $ncpu"
	perl $script/run_mergemasked.pl $i $script $TMP_DIR $ncpu 
done
wait

## Merge all the results in $TMP_DIR/masked to *.*.all.masked.tsv
for i in quasart searchqa 
do
	for j in train dev test
	do
		echo "cat $TMP_DIR/masked/$i.$j.[0-9]*.masked > $TMP_DIR/$i.$j.all.masked.tsv"
		cat $TMP_DIR/masked/$i.$j.[0-9]*.masked > $TMP_DIR/$i.$j.all.masked.tsv &
	done
done
wait

## Arange the results
for i in quasart searchqa 
do
	for j in train dev test
	do
		echo "perl $script/make_npm_data.pl $i $j $TMP_DIR"
		perl $script/make_npm_data.pl $i $j $TMP_DIR &
	done
done
wait


## Convert the *npm files to JSON. 
for i in quasart searchqa 
do
	echo "python $script/txt2json.py --dataset $i --base_dir $base_dir"
	python $script/txt2json.py --dataset $i --base_dir $base_dir &
done
wait

## Finally, apply the same preprocessing step in the OpenQA system (https://github.com/thunlp/OpenQA) to the JSON file.
for i in quasart searchqa 
do
	echo "python src/preprocess_npm.py --dataset $i --base_dir $base_dir"
	python $script/preprocess_npm.py --dataset $i --base_dir $base_dir &
done
wait

## Copy the resulting file to dataset/openqa
for i in quasart searchqa 
do
	cp $i/*.npm.json $tgt_dir/openqa
done
cd $base_dir
