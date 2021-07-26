echo "download pretrained CNNs"

for i in 100 200 300
do
	
	# dir: cnn_1.2.3.4_100
	cnn_1234_dir=cnn_1.2.3.4.$i
	cnn_123_dir=cnn_1.2.3.$i
	cnn_234_dir=cnn_2.3.4.$i
	cnn_1234=cnn_1.2.3.4.$i.pt
	cnn_123=cnn_1.2.3.$i.pt
	cnn_234=cnn_2.3.4.$i.pt
	if [ ! -f "$cnn_1234" ]; then
		echo "$cnn_1234"
		wget https://github.com/nict-wisdom/bertac/releases/download/$cnn_1234_dir/$cnn_1234
	fi;
	if [ ! -f "$cnn_123" ]; then
		echo "$cnn_123"
		wget https://github.com/nict-wisdom/bertac/releases/download/$cnn_123_dir/$cnn_123
	fi;
	if [ ! -f "$cnn_234" ]; then
		echo "$cnn_234"
		wget https://github.com/nict-wisdom/bertac/releases/download/$cnn_234_dir/$cnn_234
	fi;
done
