from Thext import SentenceRankerPlus
from Thext import Highlighter
from Thext import RedundancyManager
import pandas as pd
import numpy as np
import rouge
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from datasets import load_dataset
import nltk
nltk.download('punkt')




def test_models(task = "task1",
		fixed_weights = True,
				method = "default",
				num_highlights = 3,
				base_model_name = "morenolq/thext-cs-scibert"):
	
	"""
	
	task:
		"task1" ---> use as context the first part of an article
		"task2" ---> use as context the abstractive summary of the article given by BART model
	method:
		"default" 	 	---> first 3 highlights ranked 
		"oracle"  	 	---> oracle highlights given the true labels
		"clustering" 	---> best candidate of clustering method based on BertScore
		"iterative"  	---> best candidate ranked at each iteration that don't have the higher BertScore with respect to the already taken highlights
		"trigram_block" ---> iterative approach based on ranked sentences whith filter on trigram of already taken highlights

	"""

	if task =="task1":
		data = load_dataset("cnn_dailymail" ,"3.0.0", split="validation")
		
		if fixed_weights:
			model_name_or_path='checkpoint3_morenolq-thext-cs-scibert_1'
		else :
			model_name_or_path = 'checkpoint_morenolq-thext-cs-scibert_0'
		data = pd.DataFrame(data)
		data = data.iloc[:1600]
	else:
		data = pd.read_csv('Datasets/val_task2.csv')
		if fixed_weights:
			model_name_or_path = 'checkpoint4_morenolq-thext-cs-scibert_2'
		else :
			model_name_or_path = 'checkpoint2_morenolq-thext-cs-scibert_3'
	
	sr = SentenceRankerPlus(device='cuda')
	sr.base_model_name = base_model_name
	sr.load_model(base_model_name=base_model_name, model_name_or_path=model_name_or_path,device='cuda')

	rm = RedundancyManager()
	h = Highlighter(sr, redundancy_manager = rm)


	rougue1_f = np.array([])
	rougue2_f = np.array([])
	rouguel_f = np.array([])

	for i in range(len(data)):

		if task == "task1":
			sentences = sent_tokenize(data.iloc[i]['article'])
			highlights = data.iloc[i]['highlights']

			max_length = max([ len(word_tokenize(s)) for s in sentences])

			size = 0
			max_size = 384 -2 - max_length

			abs = ''
			for s in sentences :
				to_add = len(word_tokenize(s))
				if size + to_add < max_size :
					abs += s
					size += to_add
				else :
					break
		else:
			sentences = sent_tokenize(data.iloc[i]['article'])
			highlights = data.iloc[i]['highlights']
			abs = data.iloc[i]['abstract']

		if method=="default":
			sentences = h.get_highlights_simple(sentences, abs,
                rel_w=1.0, 
                pos_w=0.0, 
                red_w=0.0, 

                prefilter=False, 
                NH = num_highlights)
		elif method=="oracle":
			sentences = h.get_highlights_oracle(sentences, highlights, num_highlights)
		elif method=="clustering" or method == "iterative" or method=="trigram_block":
			sentences = h.get_highlights_redundancy(sentences,abs,method=method)
		else:
			print("Invalid method")
			break

		r1f,r2f,rlf = compute_rogue_single_doc(sentences, highlights)

		rougue1_f = np.append(rougue1_f,r1f)
		rougue2_f = np.append(rougue2_f,r2f)
		rouguel_f = np.append(rouguel_f,rlf)

	print(f"Avarage Rougue-1 f1 score {method}: { np.average(rougue1_f) }")
	print(f"Avarage Rougue-2 f1 score {method}: { np.average(rougue2_f) }")
	print(f"Avarage Rougue-l f1 score {method}: { np.average(rouguel_f) }")


def compute_rogue_single_doc(sentences, highlights):

	predicted_highlights_concat = ' '.join(map(str, sentences))
	real_highlights_concat =  highlights

	r_computer = rouge.Rouge(metrics=['rouge-n', 'rouge-l'], limit_length=False, max_n=2, alpha=0.5, stemming=False)
	score = r_computer.get_scores(predicted_highlights_concat,real_highlights_concat) 

	return score['rouge-1']['f'],score['rouge-2']['f'], score['rouge-l']['f']
