import requests
import feedparser
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from sciexplorer.tools.tool_utils import get_arxiv_vec_store

# Tool to query the official arXiv API, can get Metadata and Abstracts
MAX_AUTHORS = 5
def arxiv_search(query:str, sortBy:str, top_k:int=3) -> dict:
    """
    ArXiv search tool.
    Arguments:
        query: The search query string for official arXiv API.
        sortBy: The sorting criteria, e.g., "relevance", "lastUpdated", "submittedDate".
        top_k: Number of results to return (default is 3). Do not exceed 10.
    Returns a dict of the form:
        search_results: A string containing the top_k search results from arXiv including title, authors, published date, and abstract.
    """
    if sortBy not in ["relevance", "lastUpdated", "submittedDate"]:
        sortBy = "relevance"
    if top_k > 10:
        top_k = 10

    params = {
    "search_query": query,
    "sortBy": sortBy,
    "sortOrder": "descending",
    "max_results": top_k,
}

    response = requests.get("https://export.arxiv.org/api/query", params=params)
    feed = feedparser.parse(response.text)

    result = ''
    for entry in feed.entries:
        try:
            title = entry.title
            authors = [author.name for author in entry.authors]
            if len(authors) > MAX_AUTHORS:
                authors = authors[:MAX_AUTHORS] + ['et al.']
            try:
                published = entry.published
            except Exception as e:
                published = 'unknown'
                print(f"HUMAN DEBUG INFO: Could not get published date due to error: {e.__class__.__name__}: {e}")
            summary = entry.summary.replace('\n', ' ').strip()
            result += f"Title: {title}\nAuthors: {', '.join(authors)}\nPublished: {published}\nAbstract: {summary}\n\n"
        except Exception as e:
            result += f"Could not parse entry due to error: {e}\n\n"


    return {'search_results': result.strip()}



def get_arxiv_pdf_qa_tool(model, embeddings, save_dir='paper_dbs', max_top_k=4, chunk_size=1000, chunk_overlap=200) -> callable:
    """Returns a tool function that can be used to answer questions based on arXiv paper PDFs using RAG.
    Arguments:
        model: An instance of a BaseChatModel to be used for RAG.
        embeddings: An instance of Embeddings to be used for vector store.
                    E.g. OpenAIEmbeddings(model="text-embedding-3-large") (from langchain_openai import OpenAIEmbeddings)
        max_top_k: maximum number of similar chunks to retrieve from the vector store.
        chunk_size: size of text chunks to split the PDF into.
        chunk_overlap: overlap between text chunks.
        save_dir: directory where to store the paper.
    Returns:
        A callable tool function that takes arxiv_id and question as arguments.
    """
    # Tool to download arXiv paper PDF, split into chunks, and answer a question using RAG
    def arxiv_pdf_qa(arxiv_id: str, question: str, top_k: int = max_top_k) -> dict:
        f"""
        Downloads an arXiv paper PDF, and calls an external LLM (with RAG) to answer a question based on the paper's content.
        Make sure to phrase the question clearly in a way that the external LLM model can understand.
        Keep in mind that there is a chance of the external LLM hallucinating answers not present in the paper.
        Arguments:
            arxiv_id: The arXiv identifier of the paper.
            question: The question to answer based on the paper's content.
            top_k: Number of similar chunks to retrieve from the vector store. Maximum is {max_top_k}.
        Returns a dict of the form:
            answer: A string containing the answer to the question based on the paper's content.
        """
        if model is None:
            return "Error: No chat model available in state."
        assert top_k <= max_top_k, f"top_k cannot be greater than {max_top_k}"
        #check if save_dir/arxiv_id exists, if not create it
        arxiv_id = arxiv_id.strip()
        vectorstore = get_arxiv_vec_store(
            arxiv_id,
            embeddings,
            save_dir=save_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
        qa_agent = RetrievalQA.from_chain_type(
            llm=model,
            retriever=retriever,
            chain_type="stuff"
        )
        answer = qa_agent.invoke({'query': question})
        return {'answer': answer.get('result', '')}
    
    return arxiv_pdf_qa


def get_arxiv_pdf_relevant_chunks_tool(embeddings, save_dir='paper_dbs', max_top_k=4, chunk_size=1000, chunk_overlap=200) -> callable:
    """Returns a tool function that can be used to recieve text chunks based on arXiv paper PDFs using a vector embedding.
    Arguments:
        embeddings: An instance of Embeddings to be used for vector store.
                    E.g. OpenAIEmbeddings(model="text-embedding-3-large") (from langchain_openai import OpenAIEmbeddings)
        max_top_k: maximum number of similar chunks to retrieve from the vector store.
        chunk_size: size of text chunks to split the PDF into.
        chunk_overlap: overlap between text chunks.
        save_dir: directory where to store the paper.
    Returns:
        A callable tool function that takes arxiv_id and question as arguments which returns a list of relevant text chunks.
    """
    # Tool to download arXiv paper PDF, split into chunks, and answer a question using RAG
    def arxiv_pdf_relevant_chunks(arxiv_id: str, phrase: str, top_k: int = max_top_k) -> dict:
        f"""
        Downloads an arXiv paper PDF, embeds the content into a vector store, and retrieves relevant text chunks similar to the phrase using similarity search.
        Arguments:
            arxiv_id: arXiv identifier of the paper.
            phrase: The phrase to find similar text chunks for.
            top_k: Number of similar chunks (each of size {chunk_size}) to retrieve from the vector store. Maximum is {max_top_k}.
        Returns a dict of the form:
            relevant_chunks: A list of strings containing the relevant text chunks based on the paper's content.
        """
        assert top_k <= max_top_k, f"top_k cannot be greater than {max_top_k}"
        arxiv_id = arxiv_id.strip()
        #check if save_dir/arxiv_id exists, if not create it
        vectorstore = get_arxiv_vec_store(
            arxiv_id,
            embeddings,
            save_dir=save_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = vectorstore.similarity_search(phrase, k=top_k)
        chunk_texts = [chunk.page_content for chunk in chunks]
        return {'relevant_chunks': chunk_texts}

        
    return arxiv_pdf_relevant_chunks

