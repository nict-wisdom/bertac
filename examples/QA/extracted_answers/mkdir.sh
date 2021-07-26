## mkdir for output directories of extracting answers
for i in dev test
do
	if [ ! -d "$i" ]; then
		mkdir $i
	fi;
	
	for j in quasart searchqa 
	do
		if [ ! -d "$i/$j" ]; then
			mkdir $i/$j
		fi;
	done
done
