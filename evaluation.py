import numpy as np
import pandas as pd
from sklearn.metrics import precision_score
from search_space_utils import *

class Evaluate():
    def __init__(self,query,truth_ranks,model,top_k):
        """
        Object Oriented implementation of a FIASS optimized document retrieval system. 
        Parameters
        ----------
        :param query: user query for which intend to make a search.
        :param model: The word2vec model used to train embedding representation.
        :param truth_ranks: A dataframe containing the actual truth (ranked documents and their order (target)). 
        :param top_k: maximum number of relevant documents expected to be retrieved in a search. 
        """
        self.query = query
        self.truth_ranks = truth_ranks
        self.top_k = top_k
        self.w2v_model = model


    def get_truth_mapping(self,retrieved_results,qid):
        """
        Get the true mapping, All retrieved document are termed relevant, but the ground truth value for its relevance must be determined. 
        Parameters
        ----------
        :param retrieved_results: search results {a dictionary of document ids retrieved during seearch}.
        :param qid: The query id.
        """
        retrieved_df = pd.DataFrame(retrieved_results)
        retrieved_df['relevance']=1
        result_df=retrieved_df[['docid','relevance']]
        actual_labels=self.truth_ranks[self.truth_ranks['qid']==qid][['docid','relevance','qid']].reset_index(drop=True)
        prediction_map, ground_map={},{}
        #create a dictionary mapping of document ids to relevance for retrieved documents.
        for idx in range(len(result_df.values[:,0])):
            prediction_map[result_df.values[:,0][idx]] = result_df.values[:,1][idx]
        #create a dictionary mapping of document ids to relevance for ground truth top documents.
        for idx in range(len(actual_labels.values[:,0])):
            ground_map[actual_labels.values[:,0][idx]] = int(actual_labels.values[:,1][idx])
        
        ground_truth={}
        for doc_id in prediction_map.keys():
            if doc_id in ground_map.keys():
                ground_truth[doc_id]= prediction_map[doc_id]
            #in the absence of the document's id in the ranked truth, indicate irrelevance.
            else:
                ground_truth[doc_id]= 0
        return ground_truth, prediction_map

    def average_precision(self,ground_truth,predicted_docs):
        """
        Get the precision until desired search number k and average the values.
        -----------------------------------------------------------------------
        :param ground_truth: ground truth for search results {a dictionary of document ids retrieved and ground truth}.
        :param predicted_docs: retrieved documents.
        Both parameters are the outputs of the `get_truth_mapping` function.
        """
        truth = list(ground_truth.values())
        pred = list(predicted_docs.values())
        precision=[]
        for idx in range(len(truth)):
            precision_at_k = precision_score(truth[:idx+1],pred[:idx+1])
            precision.append(precision_at_k)
        return np.mean(precision)

    def mean_average_precision(self,corpus,doc_embeddings):
        """
        Get mean average precision
        :param corpus: dataset from which we intend to make a search.
        :param doc_embeddings: embeddings of all documents to be searched through,
        """
        q= self.query[['qid','query']].values
        qids,queries = q[:,0],q[:,1]
        average_precision =[]
        for qid, query in list(zip(qids,queries)):
            optimizer = Optimized_search(query,doc_embeddings,self.w2v_model,corpus,self.top_k)
            search_results = optimizer.search()
            ground_truth,predicted_docs = self.get_truth_mapping(search_results,qid)
            average_precision.append(self.average_precision(ground_truth,predicted_docs))
        print(f"==========> The Mean Average Precision (MAP@{self.top_k}) of your search: {np.mean(average_precision)}")
        return np.mean(average_precision)