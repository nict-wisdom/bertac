
datadir=download

if [ ! -d "$datadir" ]; then
	mkdir $datadir
fi;

# Downloading data
for i in CoLA SST-2 STS-B QQP-clean MNLI QNLIv2 RTE
do
    data=$datadir/$i.zip
    if [ ! -f "$data" ]; then
		wget https://dl.fbaipublicfiles.com/glue/data/$i.zip -P $datadir
		unzip -o -d $datadir/ $datadir/$i.zip
	fi;
done

echo "ALL the data except for the MRPC task have been downloaded."
echo "Please download MRPC data at https://gluebenchmark.com/tasks and store them into ./$datadir/MRPC/{train,dev}.tsv"

 
