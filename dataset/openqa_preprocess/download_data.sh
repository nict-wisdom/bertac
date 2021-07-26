#
corenlp="stanford-corenlp-full-2017-06-09.zip"
OpenQA="OpenQA_data.tar.gz"
QUASART="quasart/train.json"
Searchqa="searchqa/train.json"
datadir=download

if [ ! -d "$datadir" ]; then
	mkdir $datadir
fi;

cd $datadir

if [ ! -f "$corenlp" ]; then
	wget http://nlp.stanford.edu/software/stanford-corenlp-full-2017-06-09.zip
	unzip stanford-corenlp-full-2017-06-09.zip
fi;

if [ ! -f "$OpenQA" ]; then
	wget https://thunlp.oss-cn-qingdao.aliyuncs.com/OpenQA_data.tar.gz
fi;

if [ ! -f "$QUASART" ]; then
    tar xvfz OpenQA_data.tar.gz data/datasets/quasart --strip-components=2
fi;
if [ ! -f "$Searchqa" ]; then
    tar xvfz OpenQA_data.tar.gz data/datasets/searchqa --strip-components=2
fi;

echo "All the data for OpenQA experiments have been downloaded."
