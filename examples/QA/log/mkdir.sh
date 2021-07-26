## mkdir for directories of log files (not mandatory)
for i in make_vocab  preprocess  reader  selector
do
	if [ ! -d "$i" ]; then
		mkdir $i
	fi;
done
