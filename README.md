# Semantic_Search_and_Retrieval
A Query-Document pair ranking system using GloVe embeddings and RankCosine..


### Semantic search
Have you ever wondered how you can create state-of-the-art unsupervised text embeddings and use them in downstream tasks like information retrieval?

In the last 5 years, Natural Language Processing (NLP) has leaped forward with the introduction of the new text embedding architectures, which brought large improvements to NLP applications. One of such application is semantic matching and search. Semantic search applies user intent, context, and conceptual meanings to match a user query to the corresponding content. It uses vector search and machine learning to return results that aim to match a user’s query, even when there are no word matches.
These components work together to retrieve and rank results based on meaningand proximity to a user-defined query.

In this project, I will illustrate how to build a semantic search engine using state-of-the-art vector space models. In implementing this workflow, we will build a system for accessing and retrieving the most appropriate information from text based on a particular query given by the user, with the help of context-based indexing.

![Process Workflow](static/workflow.png)

### Motivation
Whether you want to sift through millions of social media posts, extract information from reports of medical trials and academic research, or simply retrieve relevant texts from thousands of documents, reports, and notes generated by an organization as a part of its daily operation, you would need an information retrieval system with the capability to match a query and its search intent to the relevant documents

The traditional approach for information retrieval, such as `BM25`, relies on word frequencies within indexed documents and on key terms within the query to estimate the relevance of said documents to the query. This approach has two key limitations that affect its accuracy. Documents that do not contain the keywords but include terms that are semantically similar to those keywords may be missed. For a pool of documents containing different languages, and especially languages with different scripts and alphabets, the keyword approach would fail. Hence the need for better architectures or structures.

### Dataset
In this project, we will work with real-world data and I will be making use of the document ranking dataset from [TREC 2019 Deep Learning Track](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019). The dataset consists of 367,013 queries and 3.2 million documents. However, due to the unavailability of resources such as GPU and time, we will work with only a subset of this data.

### Results
Evaluation measures for an information retrieval system are used to assess how well the search results satisfies the user's query intent. Here I employed **The Mean Average Precision (map)** as the evaluation metric.


- Precision @ k: For modern (web-scale) information retrieval, recall is no longer a meaningful metric, as many queries have thousands of relevant documents, and few users will be interested in reading all of them. Precision at k documents (P@k) is still a useful metric (e.g., P@10 or "Precision at 10" corresponds to the number of relevant results among the top 10 retrieved documents).
- Mean average precision (MAP): for a set of queries, it is the mean of the average precision scores for each query. The mean average  precision score ranges from 0 to 1. Below are the map values obtained for the training set during the top@k searches.
  - Map@1: 0.896551
  - Map@2: 0.870182
  - Map@3: 0.845616
  - Map@4: 0.826276
  - Map@5: 0.809175
  - Map@6: 0.793762
  - Map@7: 0.779345
  
![MAP](static/MAP.png)

### Requirements (Libraries & Packages)

![Dask](static/dask.png) ![Spacy](static/spacy.jpg) 
![Pandas](static/pandas(2).png) ![faiss](static/faiss.png) 
![Gensim](static/gensim.png) 


