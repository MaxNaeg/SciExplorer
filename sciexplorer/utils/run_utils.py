import time
import numpy as np
import jax
from sciexplorer.tools.tool_utils import ImageData
from langchain_core.language_models import BaseChatModel

SHOW_PICTURE = False
PRINTALL = False
MAXPRINTARRAYELEMENTS = 10
MAX_STR_LENGTH = 10000
MAX_ARRAY_ELEMENTS_IF_PRINT_ALL = 100


def invoke_with_backoff(model:BaseChatModel, messages:list, invoke_kwargs:dict=None,
                        max_retries:int=0, backoff_factor:int=2, initial_delay:int=10):
    """Invoke a model with retries and exponential backoff."""
    if invoke_kwargs is None:
        invoke_kwargs = {}
    delay = initial_delay
    for attempt in range(max_retries + 1):
        try:
            response = model.invoke(messages, **invoke_kwargs)
            return response
        except Exception as e:
            print(f"<<Invocation of model failed with error: {e.__class__.__name__}: {e}.\nRetrying in {delay} seconds...>>", flush=True)
            if attempt == max_retries:
                raise Exception(f"Invocation of model failed after {max_retries} retries.") from e
            time.sleep(delay)
            delay *= backoff_factor


def replace_arrays_with_lists(obj):
    if isinstance(obj, dict):
        return {key: replace_arrays_with_lists(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(replace_arrays_with_lists(item) for item in obj)
    elif isinstance(obj, (np.ndarray, jax.Array)):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj
    


import numpy as np

def downsample_uniform_fraction(arr, max_total_elements):
    arr = np.asarray(arr)
    shape = np.array(arr.shape, dtype=float)
    ndim = arr.ndim
    total = arr.size

    if total <= max_total_elements:
        return arr

    # Check feasibility of minimum 2 per axis
    min_per_axis = 2
    min_total = min_per_axis ** ndim

    enforce_two = max_total_elements >= min_total

    # Uniform fractional scaling
    scale = (max_total_elements / total) ** (1 / ndim)
    new_shape = np.floor(shape * scale).astype(int)

    if enforce_two:
        new_shape = np.maximum(min_per_axis, new_shape)
    else:
        new_shape = np.maximum(1, new_shape)

    # Reduce largest axes until constraint satisfied
    while np.prod(new_shape) > max_total_elements:
        largest = np.argmax(new_shape)
        if enforce_two:
            if new_shape[largest] > min_per_axis:
                new_shape[largest] -= 1
            else:
                break
        else:
            if new_shape[largest] > 1:
                new_shape[largest] -= 1
            else:
                break

    # Build indices
    indices = [
        np.linspace(0, dim - 1, new_dim, dtype=int)
        for dim, new_dim in zip(arr.shape, new_shape)
    ]

    return arr[np.ix_(*indices)]

    

def get_description(field, print_all=PRINTALL, max_str_length=MAX_STR_LENGTH, max_print_array_elements=MAXPRINTARRAYELEMENTS) -> str:
    """Get a string description of a field, handling large arrays and collections.
    Args:
        field: The field to describe.
        print_all: Whether to print all numbers in arrays or just return the shape and dtype for large arrays.
        max_str_length: Maximum length of the resulting description string. If the description exceeds this length, it will be truncated.
        max_print_array_elements: Maximum number of elements in arrays, lists, sets, or tuples to print before summarizing with length and type.
    Returns:        
        A string description of the field.
    """
    if print_all:
      
            # Use numpy printoptions to nicely print arrays
        with np.printoptions(precision=2, suppress=True, threshold=int(1e4)):
            if isinstance(field, ImageData):
                description = f"Image data."
            elif isinstance(field, jax.Array) or isinstance(field, np.ndarray):
                description = f'Array of shape {field.shape} with dtype {field.dtype}. Here are some values uniformly downsapled across each axis shown only up to two decimal places: {str(downsample_uniform_fraction(field, MAX_ARRAY_ELEMENTS_IF_PRINT_ALL))}'
            elif isinstance(field, list):
                description = f"List of length {len(field)} with values {[get_description(f, print_all, max_str_length, max_print_array_elements) for f in field]}"
            elif isinstance(field, set):
                description = f"Set of length {len(field)} with values {[get_description(f, print_all, max_str_length, max_print_array_elements) for f in field]}"
            elif isinstance(field, tuple):
                description = f"Tuple of length {len(field)} with values {[get_description(f, print_all, max_str_length, max_print_array_elements) for f in field]}"
            elif isinstance(field, dict):
                description = f"Dictionary with keys with values:\n"
                for key, value in field.items():
                    description += f"{key}: {get_description(value, print_all, max_str_length, max_print_array_elements)}\n"
            else:
                description = str(field)
    else:
        np.set_printoptions(threshold=10)
        if isinstance(field, jax.Array) or isinstance(field, np.ndarray):
            if np.sum(field.shape) > max_print_array_elements:
                description = f"Array of shape {field.shape} with dtype {field.dtype}"
            else:
                description = f"Array of shape {field.shape} with dtype {field.dtype} and with values {str(field)}"
        elif isinstance(field, list):
            if len(field) > max_print_array_elements:
                description = f"List of length {len(field)}"
            else:
                description = f"List of length {len(field)} with values {[get_description(f, print_all, max_str_length, max_print_array_elements) for f in field]}"
        elif isinstance(field, set):
            if len(field) > max_print_array_elements:
                description = f"Set of length {len(field)}"
            else:
                description = f"Set of length {len(field)} with values {[get_description(f, print_all, max_str_length, max_print_array_elements) for f in field]}"
        elif isinstance(field, tuple):
            if len(field) > max_print_array_elements:
                description = f"Tuple of length {len(field)}"
            else:
                description = f"Tuple of length {len(field)} with values {[get_description(f, print_all, max_str_length, max_print_array_elements) for f in field]}"
        elif isinstance(field, dict):
            description = f"Dictionary with keys with values:\n"
            for key, value in field.items():
                description += f"{key}: {get_description(value, print_all, max_str_length, max_print_array_elements)}\n"
        elif isinstance(field, ImageData):
            description = f"Image data."
        else:
            description = str(field)
    if max_str_length is not None and len(description) > max_str_length:
        description = description[:max_str_length] + f"... (truncated. Max length is {max_str_length} characters.)"
    return description

def add_text_to_message(message: str|list, text: str) -> list|str:
    """Add text to a message, which can be either a string or a list of message parts."""
    if isinstance(message, str):
        message = message + text
    # handle case where  message is a list, e.g. with images
    elif isinstance(message, list):
        text_message_idx = None
        for idx, content_part in enumerate(message):
            if isinstance(content_part, dict) and content_part.get("type", None) == "text":
                text_message_idx = idx
                break
        if text_message_idx is not None:
            message[text_message_idx]["text"] +=  text
        else:
            message.append({"type": "text", "text": text})
    else:
        raise ValueError("Unexpected content type in last HumanMessage.")
    return message