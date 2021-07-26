fastText=wiki.en.vec
fastText_alias=fastText.enwiki.300d.txt

## download fastText vectors
if [ ! -f "$fastText" ]; then
	wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec
fi;

## make aliases
if [ ! -f "$fastText_alias" ]; then
	ln -s wiki.en.vec fastText.enwiki.300d.txt
fi;
