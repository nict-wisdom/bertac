BERTAC (BERT-style transformer-based language model with Adversarially pretrained Convolutional neural network)
======

**BERTAC** is a framework that combines a Transformer-based Language Model (TLM) such as BERT with an adversarially pretrained CNN (Convolutional Neural Network). It was proposed in our ACL-IJCNLP paper:

* [BERTAC: Enhancing Transformer-based Language Models with Adversarially Pretrained Convolutional Neural Networks](https://aclanthology.org/2021.acl-long.164/). 

We showed in our experiments that BERTAC can improve the performance of TLMs on GLUE and open-domain QA tasks when using ALBERT or RoBERTa as the base TLM. 

This repository provides the source code for *BERTAC* and [*adversarially pretrained CNN models*](cnn_models/README.md) described in the ACL-IJCNLP 2021 paper.

You can download the code and CNN models by following the procedure described in the "[Try BERTAC](#try_bertac) section." The procedure includes downloading the BERTAC code, installing libraries required to run the code, and downloading pretrained models of the [fastText word embedding vectors](https://fasttext.cc/), the [ALBERT xxlarge model](https://github.com/google-research/albert), and [our adversarially pretrained CNNs](cnn_models/README.md). The CNNs provided here were pretrained using the settings described in our ACL-IJCNLP 2021 paper. They can be downloaded automatically by running the script `download_pretrained_model.sh` as described in the "[Try BERTAC](#try_bertac) section" or manually from the following page: [cnn_models/README.md](cnn_models/README.md).

After this is done, you can run the GLUE and Open-domain QA experiments in the ACL-IJCNLP 2021 paper by following the procedure described in these pages, [examples/GLUE/README.md](examples/GLUE/README.md) and [examples/QA/README.md](examples/QA/README.md). The procedure for the experiments starts from downloading GLUE and open-domain QA datasets (Quasar-T and SearchQA datasets for open-domain QA) and includes preprocessing the dataset and training/evaluating BERTAC models. 
 
## Overview of BERTAC
BERTAC is designed to improve Transformer-based Language Models such as ALBERT and BERT by integrating a simple CNN to them. The CNN is pretrained in a GAN (Generative Adversarial Network) style using Wikipedia data. By using as training data sentences in which an entity was masked in a cloze-test style, the CNN can generate alternative entity representations from sentences. BERTAC aims to improve TLMs for a variety of downstream tasks by using multiple text representations computed from different perspectives, i.e., those of TLMs trained by masked language modeling and those of CNNs trained in a GAN style to generate entity representations.

For a technical description of BERTAC, see our paper:

* Jong-Hoon Oh, Ryu Iida, Julien Kloetzer, Kentaro Torisawa, [BERTAC: Enhancing Transformer-based Language Models with Adversarially Pretrained Convolutional Neural Networks](https://aclanthology.org/2021.acl-long.164/) 


## <a name="try_bertac"></a>Try BERTAC 

### Prerequisites

BERTAC requires the following libraries and tools at runtime.

* CUDA: A CUDA runtime must be available in the runtime environment. Currently, BERTAC has been tested with CUDA 10.1 and 10.2.
* Python and Pytorch: BERTAC has been tested with Python 3.6 and 3.8, and Pytorch 1.5.1 and 1.8.1.
* Perl: BERTAC has been tested with Perl 5.16.1 and 5.26.2.

### Installation
You can install BERTAC by following the procedure described below. 

* Create a new [conda](https://www.anaconda.com/) environment `bertac` using the following command. Set a CUDA version available in your environment.

```bash
conda create -n bertac python=3.8 tqdm requests scikit-learn cudatoolkit cudnn lz4
```
* Install Pytorch into the conda environment

```bash
conda activate bertac
conda install -n bertac pytorch=1.8 -c pytorch
```

* Git clone the BERTAC code and run `pip install -r requirements.txt` in the root directory. 

```bash
# git clone the code
git clone https://github.com/nict-wisdom/bertac
cd bertac

# Install requirements
pip install -r requirements.txt
```

* Download the [spaCy](https://spacy.io/) model `en_core_web_md`.
 
```bash
# Download the spaCy model 'en_core_web_md' 
python -m spacy download en_core_web_md
```

* Install Perl and its JSON module into the conda environment. 

```bash 
# Install Perl and its JSON module
conda install -c anaconda perl -n bertac38
cpan install JSON
```

* Download the [fastText word embedding vectors](https://fasttext.cc/), the [ALBERT xxlarge model](https://github.com/google-research/albert), and our adversarially pretrained CNNs. The CNNs were pretrained using the settings described in our ACL-IJCNLP 2021 paper (See [cnn_models/README.md](cnn_models/README.md) for the details on our pretrained CNNs).

```
# Download pretrained CNN models, the fastText word embedding vectors, and
# the ALBERT xxlarge model (albert-xxlarge-v2) 
sh download_pretrained_model.sh
```

**Note**: the BERTAC code was built on the [HuggingFace Transformers v2.4.1](https://github.com/huggingface/transformers/tree/v2.4.1) and requires the [NVIDIA apex](https://github.com/NVIDIA/apex) as in the HuggingFace Transformers. Please install the NVIDIA apex following the procedure described in the [NVIDIA apex page](https://github.com/NVIDIA/apex). 

You can enter `examples/GLUE` or `examples/QA` folders and try the bash commands under these folders to run GLUE or open-domain QA experiments (see [examples/GLUE/README.md](examples/GLUE/README.md) and [examples/QA/README.md](examples/QA/README.md) for details on the procedures of the experiments).

## GLUE experiments
You can run GLUE experiments by following the procedure described in [examples/GLUE/README.md](examples/GLUE/README.md). 

### Results
The performances of BERTAC and other baseline models on the GLUE development set are shown below. 

| Models            | MNLI         | QNLI     | QQP      | RTE      | SST    | MRPC     | CoLA     | STS      | Avg.    | 
|-------------------|--------------|----------|----------|----------|----------|----------|----------|----------|---------|
| RoBERTa-large     | 90.2/90.2    | 94.7     | 92.2     | 86.6     | 96.4     | 90.9     | 68.0     | 92.4     | 88.9    |
| ELECTRA-large     | 90.9/-       | 95.0     | **92.4** | 88.0     | 96.9     | 90.8     | 69.1     | 92.6     | 89.5    |
| ALBERT-xxlarge    | 90.8/-       | 95.3     | 92.2     | 89.2     | 96.9     | 90.9     | 71.4     | 93.0     | 90.0    |
| DeBERTa-large     | 91.1/**91.1**| 95.3     | 92.3     | 88.3     | 96.8     | 91.9     | 70.5     | 92.8     | 90.0    |
| BERTAC<br>(ALBERT-xxlarge)    | **91.3/91.1**| **95.7** | 92.3     | **89.9** | **97.2** | **92.4** | **73.7** | **93.1** | **90.7**|

*BERTAC(ALBERT-xxlarge)*, i.e., BERTAC using ALBERT-xxlarge as its base TLM, showed a higher average score (Avg. of the last column in the table) than (1) ALBERT-xxlarge (the base TLM) and (2) DeBERTa-large (the state-of-the-art method for the GLUE development set). 

## Open-domain QA experiments
You can run open-domain QA experiments by following the procedure described in [examples/QA/README.md](examples/QA/README.md). 

### Results
The performances of BERTAC and other baseline methods on *Quasar-T* and *SearchQA* benchmarks are as follows.

|Model	 				     |Quasar-T (EM/F1) |SearchQA (EM/F1) | 
|------------------------|-----------------|-----------------|
|OpenQA 				     |42.2/49.3        |58.8/64.5        | 
|OpenQA+ARG              |43.2/49.7        |59.6/65.3        |
|WKLM(BERT-base) 	     |45.8/52.2        |61.7/66.7        |
|MBERT(BERT-large)      |51.1/59.1        |65.1/70.7        |
|CFormer(RoBERTa-large) |54.0/63.9        |68.0/75.1        |
|BERTAC(RoBERTa-large)  |55.8/63.7        |71.9/77.1        |
|BERTAC(ALBERT-xxlarge) |**58.0/65.8**    |**74.0/79.2**    |

Here, BERTAC(RoBERTa-large) and BERTAC(ALBERT-xxlarge) represent BERTAC using RoBERTa-large and ALBERT-xxlarge as their base TLM, respectively. BERTAC with any of the base TLMs showed better EM (Exact match with the gold standard answers) than the state-of-the-art method, CFormer(RoBERTa-large), for both benchmarks (Quasar-T and SearchQA).

## Citation

If you use this source code, we would appreciate if you cite the following paper:

```
@inproceedings{ohetal2021bertac,
  title={BERTAC: Enhancing Transformer-based Language Models 
         with Adversarially Pretrained Convolutional Neural Networks},
  author={Jong-Hoon Oh and Ryu Iida and 
          Julien Kloetzer and Kentaro Torisawa},
  booktitle={The Joint Conference of the 59th Annual Meeting  
             of the Association for Computational Linguistics  
             and the 11th International Joint Conference 
             on Natural Language Processing (ACL-IJCNLP 2021)},
  year={2021}
}
```
## Acknowledgements

Part of the source codes is borrowed from [HuggingFace Transformers v2.4.1](https://github.com/huggingface/transformers/tree/v2.4.1) licensed under [Apache 2.0](https://github.com/huggingface/transformers/blob/v2.4.1/LICENSE), [DrQA](https://github.com/facebookresearch/DrQA) licensed under [BSD](https://github.com/facebookresearch/DrQA/blob/master/LICENSE), and [Open-QA](https://github.com/thunlp/OpenQA) licensed under [MIT](https://github.com/thunlp/OpenQA/blob/master/LICENSE).