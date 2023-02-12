from sklearn.cluster import AgglomerativeClustering
from evaluate import load
import pandas as pd
import numpy as np
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import string

class RedundancyManager():
  def __init__(self, metric = "bertscore"):
    self._bertscore = load(metric) 
    
  def get_highlights_trigram(self,sentences_ranked, n_highlights=3):
    highlights = []
    current_trigrams = set()
    for sentence in sentences_ranked:
      if len(highlights)==n_highlights:
        break
      s = word_tokenize(self.remove_punctuation(sentence))
      check,tmp_current_trigrams = self.check_trigram(current_trigrams, s)
      if check:
        highlights.append(sentence)
        current_trigrams = tmp_current_trigrams
      else:
        continue
    return highlights
    

  def get_highlights_iterative(self,sentences_ranked, n_highlights=3):

    highlights = []
    highlights.append(sentences_ranked[0])
    index_from = 1
    index_taken = [0]
    while len(highlights)<n_highlights:
      worst_candidate_index = self.get_worstcandidate(sentences_ranked, index_from, highlights)
      for ind in range(index_from,len(sentences_ranked)):
        if ind not in worst_candidate_index:
          highlights.append(sentences_ranked[ind])
          index_taken.append(ind)
          index_from = ind+1
          break
    #[print(ind) for ind in index_taken]
    return highlights
      
  def get_highlights_cluster(self,sentences_ranked, nclusters=3): 
    distances = self.make_distance_metric_hugging_opt(sentences_ranked, False) 
    highlights = self.extract_set(distances, nclusters, sentences_ranked)
    return highlights



#------ clustering functions


  def extract_set(self,distances, n_clusters, sentences):
    highlights = []
    
    l={}
    #index = []
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(distances) #create an AgglomerativeClustering object with n_clusters=3
    clusters = clustering.labels_ # create a ndarray of shape (n_samples) with the cluster labels for each point. EX: array([1, 0, 0, 1, 0, 0, 0, 2, 1, 1, 1, 1, 1]) meaning that element 0 of the matrix of distances is clusterend in the second cluster while the element 1 is assigned to the fist cluster
    for i in range(n_clusters):#for each cluster we created 
        l[i] = np.argwhere(clusters==i)[0] #we take the index of the first element of each cluster and save it the dictionary of arrays EX: {0: array([1]), 1: array([0]), 2: array([7])} meaning that the first element clustered as 0 is the second sentence and so on

        #for element in range(len(l[cluster])): #for each element of the cluster in the dictionary(that is composed only from 1 element each)
        sent = sentences[l[i][0].item()] #take the sentence with the index from the dictionary
        highlights.append(sent) #store the sentence in the array

    
    return highlights


  def make_distance_metric_hugging_opt(self, sentences, idf): #create the BertScore distance matrix by creating first the upperdiagonal matrix(since it is a symmetric matrix) and filling the diagonal by 1.
    results = []
    n = len(sentences)
    reference = np.array(sentences)
    reference = np.delete(reference,0,None)
    for i in range(n):
      
      zeros = np.zeros((i+1,))
      predictions = np.array([sentences[i]] * (n-len(zeros)))
      if len(predictions)>0:
        predictions = np.array(self._bertscore.compute(predictions=predictions, references=reference, model_type="distilbert-base-uncased", idf=idf)["f1"])
        final = np.concatenate([zeros,predictions])
        reference = np.delete(reference, 0, None)
      else:
        final = zeros
      results = np.append(results,final)
      
    results = np.array(results).reshape(-1,n)
    final =results+results.T
    np.fill_diagonal(final, 1)  
    return final

#-------- iterative functions

  def get_worstcandidate(self, sentences, index_from, highlights,n_topelements=1):
    concat= np.empty((len(sentences)-index_from,1))
    for highlight in highlights:
      predictions = sentences[index_from:]
      reference = [highlight]*(len(predictions))
      sent_bert = np.array(self._bertscore.compute(predictions=predictions, references=reference, model_type="distilbert-base-uncased", idf=False)["precision"]).reshape((len(sentences)-index_from,1))
      concat = np.concatenate((concat, sent_bert), axis=1)
    max_score = concat.max(axis=1)
    top_elem_index = np.argsort(max_score)[-n_topelements:]
    return top_elem_index+index_from
    
#-------- trigram blocking
  def remove_punctuation(self,text):
      punctuationfree="".join([i for i in text if i not in string.punctuation])
      return punctuationfree


  def check_trigram(self,current_trigrams, units):
    new_trigrams = set(ngrams(units,3))
    return len(current_trigrams.intersection(new_trigrams))==0, current_trigrams.union(new_trigrams)