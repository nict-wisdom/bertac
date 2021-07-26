## mkdir for directories of cached files 
for i in dset  exset  feat  pdset
do
	if [ ! -d "$i" ]; then
		mkdir $i
	fi;
done
