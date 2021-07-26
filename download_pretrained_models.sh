cd cnn_models
echo "downloading pretrained CNNs"
sh download_CNN_models.sh 
cd ..

cd fastText 
echo "downloading fastText word embedding vectors"
sh download_fastText.sh
cd ..

cd pretrained_models/albert-xxlarge-v2/
echo "downloading ALBERT-xxlarge-v2"
sh download.sh
cd ../../

cd pretrained_models/roberta-large/
echo "downloading RoBERTa-large"
sh download.sh
cd ../../

