# THExt

**T**ransformer-based **H**ighlights **Ext**raction(**THExt**)

### Examples and demo

All examples provided below have been extracted using the best-performing model reported in the paper. 


## Installation

Run the following to install

```python
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

## Usage

### Pretrained models and datasets on:

You can find pre-trained models and dataset for testing here: 
https://drive.google.com/drive/folders/14MpUG-F03e7m8yUtvXOKJDK3cLBCPLiJ?usp=sharing

If you want to download and use our dataset and pre-trained models, you open the link above and run the following in a python notebook

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Using pretrained models
```python
#to train the model run:
from Thext.utils.train import train

train('Datasets/dataset_task2.csv', "checkpoint", True)

#to test the preformances run:
from Thext.utils.test import test_models

test_models("task1",method="trigram_block")
