#import Relevant packages
import time
import faiss
import numpy as np
import pandas as pd
import gensim
from preprocessing_utils import *


class InvalidFormatException(Exception):
    """Model Format does not conform with the required format."""
    pass


def test_model(model):
    error_message = "Model Format does not conform with the required format. \nThe required model format is gensim.models.word2vec.Word2Vec"
    if type(model) != gensim.models.word2vec.Word2Vec:
        raise InvalidFormatException(error_message)


class Optimized_search():
    def __init__(self,query,docs_embeddings,w2v_model,docs_df,top_k):
        """
        Object Oriented implementation of a FIASS optimized document retrieval system. 
        Parameters
        ----------
        :param query: user query for which intend to make a search.
        :param docs_embeddings: set of all document embeddings for which we intend to retrieve a document from.
        :param w2v_model: The word2vec model used to train embedding representation.
        :param docs_df: A dataframe containing all the documents from which we intend to make a search. 
        :param top_k: maximum number of relevant documents expected to be retrieved in a search. 
        """
        self.query = query
        self.docs_embeddings = docs_embeddings
        self.docs_df = docs_df
        self.top_k = top_k
        #Verify the model format;
        if not test_model(w2v_model):
            self.model = w2v_model
    
   
    def optimized_similarity_search(self,query_embedding,embedding_dim):
        """
        Perform fast query-document pair normalized cosine similarity search. 
        Parameters
        ----------
        :param query_embedding: The embedding representation of user defined query.
        :param embedding_dim: The embedding dimension (should be the same for both documents and queries).
        """
        index = faiss.index_factory(embedding_dim, "Flat", faiss.METRIC_INNER_PRODUCT)
        index.ntotal
        faiss.normalize_L2(self.docs_embeddings)
        index.add(self.docs_embeddings)
        faiss.normalize_L2(query_embedding)
        #begin search
        distance, sorted_index = index.search(query_embedding, self.top_k)
        #Get ranked results
        sorted_index=np.asarray(sorted_index[0])
        #To tally the results; you can compare the cosine similarity results here with those obtained from Scipy:
        # from scipy import spatial
        # for document in docs:
        #     result = 1 - spatial.distance.cosine(query, document)
        #     print('Distance by scipy:{}'.format(result))
        return sorted_index

    def fetch_document_info(self,dataframe_idx):
        """
        Document Retrieval: Fetch metadata of the top k relevant documents using the indices returned in  `optimized_similarity_search`.
        Parameters
        ----------
        :param dataframe_idx: indices for top k relevant documents. The value is the result of `optimized_similarity_search`.
        """
        info = self.docs_df.iloc[dataframe_idx]
        meta_dict = dict()
        meta_dict['docid'] = info['docid']
        meta_dict['document'] = info['body']
        return meta_dict

    def search(self):
        """
        Perform document retrieval and search based on given query.
        """
        start_time=time.time()
        query_embedding = np.asarray(process_query(self.query,self.model))
        if query_embedding.ndim == 1:
            query_embedding = np.asarray([query_embeddings])
        elif query_embedding.ndim == 3:
            query_embedding = query_embedding.squeeze(1)
        else:
            pass
        embedding_dim = query_embedding.shape[1]
        ranked_indices = self.optimized_similarity_search(query_embedding,embedding_dim)
        print('>>>> Total time taken to perform search: {}'.format(time.time()-start_time))
        results =  [self.fetch_document_info(index) for index in ranked_indices]
        return results