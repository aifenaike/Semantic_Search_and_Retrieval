import ast
import re
import numpy as np

def construct_contractions_dict():
    # reading the contractions data from text file
    with open('data/contractions.txt') as f:
        contractions = f.read()
        
    # reconstructing the data as a dictionary
    contractions_dict = ast.literal_eval(contractions)
    return contractions_dict

# Function for expanding contractions
def expand_contractions(text,contractions_dict=construct_contractions_dict()):
    # Regular expression for finding contractions
    contractions_re=re.compile('(%s)' % '|'.join(contractions_dict.keys()))
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, text)

# Function for cleaning contractions
def clean_text(text):
    text=re.sub('\w*\d\w*','', text)
    text=re.sub('\n',' ',text)
    text=re.sub(r"http\S+", "", text)
    text=re.sub('[^a-z]',' ',text)
    return text

 # Vectorization Function for averaging sentence level embeddings.
def aggregate_embedding_w2v(w2v_model,doc_tokens):
    """
    Creates a vector representation (word embeddings) for each token in a piece of text and aggregates the mean.
    params:
    -------
    doc_tokens: A list of tokens associated with a given document.
    """
    average_embedding =w2v_model.wv.get_mean_vector(doc_tokens,pre_normalize=False,ignore_missing=True)
    return average_embedding.reshape(1,-1)

def process_query(query,model):
    query=query.lower()
    text = expand_contractions(query,contractions_dict=construct_contractions_dict())
    text = clean_text(text)
    text=re.sub(' +',' ',text)
    text_tokens = text.split()
    query_embeddings = aggregate_embedding_w2v(model,text_tokens)
    return query_embeddings


