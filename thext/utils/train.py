from .. import SentenceRankerPlus
from sklearn.model_selection import train_test_split
import pandas as pd

def train(dataset_name, checkpointName = "checkpoint", freezed = False):

    dataset = pd.read_csv(dataset_name)
    rouge_label = 'r2f'
    base_model_name = "morenolq/thext-cs-scibert"
    model_name_or_path = "morenolq/thext-cs-scibert"
    sr = SentenceRankerPlus(base_model_name=base_model_name, model_name_or_path=model_name_or_path, device='cuda')
    sr.load_model(base_model_name=base_model_name, model_name_or_path=model_name_or_path,device='cuda')
    if freezed:
        for param in sr.model.bert.parameters():
                param.requires_grad = False

    X_train, X_test, y_train, y_test = train_test_split(dataset[['sentence','abstract']], dataset[rouge_label], test_size=0.2, random_state=42)

    sr.set_text(X_train['sentence'].values,True)
    sr.set_text(X_test['sentence'].values,False)
    sr.set_abstract(X_train['abstract'].values,True)
    sr.set_abstract(X_test['abstract'].values,False)
    sr.set_labels(y_train.values,True)
    sr.set_labels(y_test.values,False)

    sr.prepare_for_training()
    sr.continue_fit(checkpointName, last_epoch=0)




if __name__ == "__main__":
    dataset = pd.read_csv('Datasets/dataset_task1.csv')
    train(dataset, "checkpoint")

