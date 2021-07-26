## mkdir for directories of log files (not mandatory)
for i in glue  make_vocab  preprocess
do
	if [ ! -d "$i" ]; then
		mkdir $i
	fi;
done
