## mkdir for directories of trained models
for i in CoLA  MNLI  MRPC  QNLI  QQP  RTE  SST-2  STS-B
do
	if [ ! -d "$i" ]; then
		mkdir $i
	fi;
done
