## mkdir for directories of cached files 
for i in cola  mnli  mnli-mm  mrpc  qnli  qqp  rte  sst-2  sts-b
do
	if [ ! -d "$i" ]; then
		mkdir $i
	fi;
done
