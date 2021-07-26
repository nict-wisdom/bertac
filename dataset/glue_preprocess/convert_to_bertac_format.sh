cur_dir=`pwd`
cd ..
base_dir=`pwd`
echo $base_dir
cd $cur_dir


## preparing output directories
if [ ! -d "$base_dir/GLUE" ]; then
	mkdir $base_dir/GLUE
fi;

for i in CoLA  MNLI  MRPC  QNLI  QQP  RTE  SST-2  STS-B
do
	if [ ! -d "$base_dir/GLUE/$i" ]; then
		mkdir $base_dir/GLUE/$i
	fi;
done

cd  NPMasking

## preparing working directories
for i in np masked
do
	if [ ! -d "$i" ]; then
		mkdir $i
	fi;
done

# NP extracting using the spaCy NP chunker.
echo "perl np_extractor.pl $base_dir"
perl np_extractor.pl $base_dir/glue_preprocess

# Single-sentence tasks: Augmenting a np-maksed sentence to the original data 
for i in CoLA SST-2
do
	echo "perl find_cand4ssent.pl $i.train"
	perl find_cand4ssent.pl $i.train
	perl find_cand4ssent.pl $i.dev
	perl merge_masked_ssent.pl $i train $base_dir
	perl merge_masked_ssent.pl $i dev $base_dir
done

# Sentence-pair tasks: Augmenting np-maksed sentences to the original data 
for i in MNLI  MRPC  QNLI  QQP   RTE  STS-B
do
	echo "perl find_cand4spair.pl $i.train"
	perl find_cand4spair.pl $i.train
	perl merge_masked_spair.pl $i train $base_dir
	if [ $i == 'MNLI' ]; then
		echo "perl find_cand4spair.pl $i.dev_matched"
		perl find_cand4spair.pl $i.dev_matched
		perl find_cand4spair.pl $i.dev_mismatched
		perl merge_masked_spair.pl $i dev_matched $base_dir
		perl merge_masked_spair.pl $i dev_mismatched $base_dir
	else
		echo "perl find_cand4spair.pl $i.dev"
	    perl find_cand4spair.pl $i.dev
		perl merge_masked_spair.pl $i dev $base_dir
	fi
done

