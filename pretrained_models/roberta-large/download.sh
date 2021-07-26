## download the huggingface version of ALBERT-xxlarge-v2
wget https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin
wget https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-config.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-merges.txt

## make aliases
ln -s roberta-large-pytorch_model.bin pytorch_model.bin
ln -s roberta-large-config.json config.json
ln -s roberta-large-vocab.json vocab.json
ln -s roberta-large-merges.txt merges.txt


