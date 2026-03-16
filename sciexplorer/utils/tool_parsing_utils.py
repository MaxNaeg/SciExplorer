import inspect
import functools
from typing import Any
import typing

def custom_partial(func:callable, /, *args, **kwargs)->callable:
    """Like functools.partial but returns a normal function with updated signature.
    This makes the resulting function compatible, e.g. with the @tool decorator of LangChain."""
    
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    
    # Remove positional arguments already supplied
    new_params = params[len(args):]
    
    # Remove keyword arguments already supplied
    new_params = [
        p for p in new_params
        if p.name not in kwargs
    ]
    
    # Create new signature
    new_sig = sig.replace(parameters=new_params)
    
    @functools.wraps(func)
    def wrapper(*more_args, **more_kwargs):
        return func(*args, *more_args, **{**kwargs, **more_kwargs})
    
    wrapper.__signature__ = new_sig
    return wrapper


def replace_argument(func:callable, arg_name:str, arg_value:Any)->callable:
    """
    Returns a new function like `func` where `arg_name` is replaced by `arg_value` if present.
    If `arg_name` is not a parameter of `func`, returns `func` unchanged.
    Keeps the original function's signature (without the replaced argument) for compatibility.
    Args:
        func: The original function.
        arg_name: The name of the argument to replace.
        arg_value: The value to use for the argument.
    Returns:
        A new function with the argument replaced, or the original function.
    """
    sig = inspect.signature(func)
    if arg_name in sig.parameters:
        # Use custom_partial to pre-fill the argument
        return custom_partial(func, **{arg_name: arg_value})
    else:
        # Argument not present; return original function
        return func
    
    
def function_to_result_label_tool(func:callable)->callable:
    """
    Wraps a function to return its result as a labeled dictionary.
    The label is added as a prefix to each key in the result dictionary and specified as an argument to the function.
    Args:
        func: The original function.
    Returns:
        A new function that returns the result as a labeled dictionary.
    """

    doc = func.__doc__ or ""
    doc = doc.rstrip()  # clean trailing whitespace
    
    # Choose format: we’ll just append a clear line
    new_doc_line = '\nThe parameter `result_label` will be prefixed to the keys of the dictionary returned by this function, e.g., ' \
    'if `result_label=""` the function would return {"<key0>": value0, ...} and if `result_label="test_"`, the function would return {"test_<key0>": value0, ...}. ' \
    'Make sure to use result_labels which are valid starts of python variable names.'
    #f"\n'result_label' is used to save the result in the format <result_label><returned_result_name>."
    new_doc = doc + new_doc_line

    @functools.wraps(func)
    def wrapper(*args ,result_label:str="", **kwargs) -> dict:
        if result_label != "" and not result_label.isidentifier():
            raise ValueError(f"result_label must be a valid identifier (i.e., a valid variable name in Python). Got '{result_label}'.")
        result = func(*args, **kwargs)
        assert isinstance(result, dict), "Function wrapped by function_to_result_label_tool must return a dictionary."
        labeled_result = {}
        for key, value in result.items():
            labeled_key = f"{result_label}{key}"
            labeled_result[labeled_key] = value
        return labeled_result
    # Update the signature to include result_label
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    new_params = params + [inspect.Parameter("result_label", kind=inspect.Parameter.POSITIONAL_OR_KEYWORD, 
                                    annotation=str, default="")]
    new_sig = sig.replace(parameters=new_params)


    wrapper.__signature__ = new_sig

    hints = typing.get_type_hints(func).copy()
    hints["result_label"] = str
    wrapper.__annotations__ = hints

    wrapper.__doc__ = new_doc
    return wrapper