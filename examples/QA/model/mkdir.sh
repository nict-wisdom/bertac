## mkdir for directories of trained models
for i in reader selector
do
	if [ ! -d "$i" ]; then
		mkdir $i
	fi;
done
