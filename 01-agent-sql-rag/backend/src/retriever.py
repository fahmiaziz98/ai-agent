from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings 
from constant import COHERE_API_KEY

embeddings = CohereEmbeddings(model="embed-multilingual-v3.0", cohere_api_key=COHERE_API_KEY)

def retriever(docs, k=5, search_type='mmr', lambda_mult=None):
    """
    Creates a document retriever using FAISS (Facebook AI Similarity Search).

    Parameters:
    -----------
    docs : List[Document]
        A list of documents to be indexed for similarity search.

    k : int, optional, default=5
        The number of top-k results to return for a query.

    search_type : str, optional, default='mmr'
        The type of search to perform. Options include 'mmr' and 'similarity'.

    lambda_mult : float, optional, default=None
        Lambda multiplier for Maximal Marginal Relevance (MMR) search.

    Returns:
    --------
    retriever : Retriever
        A retriever object for querying relevant documents.
    """
    # Create FAISS index from documents
    vector_store = FAISS.from_documents(docs, embedding=embeddings)

    # Prepare search kwargs with optional lambda_mult
    search_kwargs = {'k': k}
    if lambda_mult is not None:
        search_kwargs['lambda_mult'] = lambda_mult

    # Return the retriever
    return vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
