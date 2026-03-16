
import os
import time
import arxiv
import signal
from langchain.chat_models import BaseChatModel
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import traceback
import linecache
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout


TIMEOUT = 300


def exec_with_error_line(code, globals, filename="<exec_block>"):
    # Register the code with linecache so traceback can look it up
    linecache.cache[filename] = (
        len(code),              # size
        None,                   # mtime (not important)
        code.splitlines(True),  # list of lines WITH newlines
        filename
    )

    try:
        compiled = compile(code, filename, "exec")
    except SyntaxError as se:
        line_text = linecache.getline(filename, se.lineno).strip()
        error_message = f"{se.__class__.__name__}: {se.msg}"
        error_message += f"\n Error in line: {line_text}"
        raise SyntaxError(error_message)
    
    try:
        exec(compiled, globals)
        return globals
    except Exception as e:
        error_message = ""
        tb = traceback.extract_tb(e.__traceback__)
        tb = [frame for frame in tb if frame.filename == filename]
        if not tb or len(tb) == 0:
            error_message += "Unknown error location"
        frame = tb[-1]
        line_no = frame.lineno
        line_text = frame.line.strip() if frame.line else "(still unavailable)"

        error_message += f"\nError in code block at line {line_no}:"
        error_message += f"\n>>> {line_text}\n{e.__class__.__name__}: {e}"
        raise Exception(error_message)



def download_pdf_from_arxiv(arxiv_id: str, save_dir="papers", wait_time=3) -> str:
    """Download a PDF from arXiv given its identifier."""
    os.makedirs(save_dir, exist_ok=True)
    search = arxiv.Search(id_list=[arxiv_id])
    paper = next(search.results())
    path = paper.download_pdf(dirpath=save_dir, filename=f"{arxiv_id}.pdf")
    time.sleep(wait_time)  # wait for a few seconds to avoid rate limiting
    return path


def extract_text_from_pdf(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """Extract text from a PDF and split it into chunks."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


def get_arxiv_vec_store(arxiv_id: str, embeddings, save_dir='paper_dbs', chunk_size=1000, chunk_overlap=200) -> FAISS:
    """Creates or loads a FAISS vector store for the given arXiv paper.
    Arguments:
        arxiv_id: The arXiv identifier of the paper.
        embeddings: An instance of Embeddings to be used for vector store.
        save_dir: directory where to store the paper.
        chunk_size: size of text chunks to split the PDF into.
        chunk_overlap: overlap between text chunks.
    Returns:
        A FAISS vector store instance.
    """
    #check if save_dir/arxiv_id exists, if not create it
    main_path = os.path.join(save_dir, arxiv_id)
    if not os.path.exists(main_path):
        os.makedirs(main_path, exist_ok=True)
    # check if pdf exists if not download it
    pdf_path = os.path.join(main_path, f"{arxiv_id}.pdf")
    if not os.path.exists(pdf_path):
        pdf_path = download_pdf_from_arxiv(arxiv_id, save_dir=main_path)
    # check if vectorestore already exist if not create it
    if os.path.exists(os.path.join(main_path, "index.faiss")):
        # load FAISS from disk
        vectorstore = FAISS.load_local(main_path, embeddings, allow_dangerous_deserialization=True)
    else:
        chunks = extract_text_from_pdf(pdf_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        vectorstore = FAISS.from_texts(chunks, embeddings)
        vectorstore.save_local(main_path)
    
    return vectorstore


class ImageData():
    """
    A class to represent image data in a format compatible with LLM APIs.
    Attributes:
        image (str): The base64 encoded image in data URL format.
    """
    def __init__(self, image: str):
        self.image = image 


def get_image() -> ImageData:
    """
    Turn the current matplotlib plot into an LLM-compatible image data object.
    Returns:
       ImageData (object with image converted to base64 encoding)
    """
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO
    
    # Save to BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    
    # Convert to base64 and create data URL
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    
    # Format as data URL for OpenAI tool response
    data_url = img_base64

    return ImageData(image=data_url)




def run_with_timeout(func, input, timeout=TIMEOUT):
    """Run a function with a timeout. If the function does not complete within the timeout, raise TimeoutError.
    This only works on Unix-based systems.
    Args:
        func: The function to run.
        args: The positional arguments to pass to the function.
        kwargs: The keyword arguments to pass to the function.
        timeout: The timeout in seconds.
    Returns:
        The result of the function.
    Raises:
        TimeoutError: If the function does not complete within the timeout."""

    # Define a timeout handler
    def handler(signum, frame):
        raise TimeoutError(f"Function {func.__name__} timed out after {timeout} seconds.")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)  # Set the timeout to the specified value
    result = func(input)
    signal.alarm(0)  # Cancel the alarm if finished in time
    return result


def run_with_timeout_threads(func, input, timeout):
    """Run a function with a timeout. If the function does not complete within the timeout, raise TimeoutError.
    This is based on threads and works on all platforms, however it may cause problems with multi-threaded functions.
    Args:
        func: The function to run.
        args: The positional arguments to pass to the function.
        kwargs: The keyword arguments to pass to the function.
        timeout: The timeout in seconds.
    Returns:
        The result of the function.
    Raises:
        TimeoutError: If the function does not complete within the timeout."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, input)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeout:
            raise TimeoutError(
                f"Function {func.__name__} timed out after {timeout} seconds."
            )






