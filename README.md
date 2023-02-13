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

If you want to download and use our dataset and pre-trained models, open the link above and create a shortcut in your drive account.

Then, run the following code in a python notebook

```python
from google.colab import drive
drive.mount('/content/drive')
```

At the end, just set your current directory to
```python
%cd /content/drive/MyDrive/Project File
```

### Using pretrained models
```python
from huggingface_hub import notebook_login
notebook_login() # the token is inside the drive folder

# to train the model run:
from Thext.utils.train import train

train('Datasets/dataset_task2.csv', "checkpoint", True)

# to test the preformances run:
from Thext.utils.test import test_models

test_models("task1",method="trigram_block")
```

### Dataset creation
If you want to create the dataset by your own, run:
```python
from Thext import DatasetPlus

dataset_manager = DatasetPlus()

dataset_manager.dataset_task1("dataset.csv")
# you can use also the methods dataset_task2("dataset.csv") and validation_set_task2("dataset.csv") 
```