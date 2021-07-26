## download the huggingface version of ALBERT-xxlarge-v2
albert_model=albert-xxlarge-v2-pytorch_model.bin
albert_model_alias=pytorch_model.bin

if [ ! -f "$albert_model" ]; then
	wget https://s3.amazonaws.com/models.huggingface.co/bert/albert-xxlarge-v2-pytorch_model.bin
	wget https://s3.amazonaws.com/models.huggingface.co/bert/albert-xxlarge-v2-spiece.model
	wget https://s3.amazonaws.com/models.huggingface.co/bert/albert-xxlarge-v2-config.json
fi;

## make aliases
if [ ! -f "$albert_model_alias" ]; then
	ln -s albert-xxlarge-v2-pytorch_model.bin pytorch_model.bin
	ln -s albert-xxlarge-v2-spiece.model spiece.model
	ln -s albert-xxlarge-v2-config.json config.json
fi;


