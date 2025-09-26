
"""
A module for simple tool use based on the OpenAI API.

parse_tools : create tool descriptions for a list of functions

ask_question_with_tool_steps: run a conversation with tools

get_image: obtain ImageData for the current plot
"""

__all__ = ['parse_tools', 'ask_question_with_tool_steps', 'get_image']

import inspect
import re

import jax
import numpy as np

from functools import wraps
from typing import get_type_hints, get_origin, get_args, Union, Callable



# class ImageData(TypedDict):
#     type: Literal["image"]
#     image: str  # The data URL with base64 content

class ImageData():
    """
    A class to represent image data in a format compatible with OpenAI tools.
    
    Attributes:
        type (str): The type of the data, should be "image".
        image (str): The base64 encoded image in data URL format.
    """
    def __init__(self, type: str, image: str):
        self.type = type
        self.image = image



def get_method(obj, method_name):
    method = getattr(obj, method_name, None)
    if callable(method):
        return method
    else:
        raise AttributeError(f"'{type(obj).__name__}' object has no method '{method_name}'")

def get_image() -> ImageData:
    """
    Turn the current matplotlib plot into an 
    OpenAI-style dictionary, for use with an LLM.
    Return this from your tool (and give the type hint
    "def my_tool(...) -> ImageData").
    
    Note: This does not close the image, so you have a chance
    of showing it using plt.show()! If you do not want to show
    it, anyway use plt.close() to free the memory.
    
    Returns:
       ImageData (dict with image converted to base64 encoding)
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
    data_url = f"data:image/png;base64,{img_base64}"

    return ImageData(type="image", image=data_url)
    
    # Return in format expected by OpenAI
    # return {
    #     "type": "image",
    #     "image": data_url
    # }
  

def method_to_openai_tool(method, include_docstring=True, include_return=True, response_api=False):
    """
    Convert a Python method to an OpenAI tool description format.
    
    Args:
        method: The method to convert
        include_docstring: Whether to parse and include docstring info
        include_return: Whether to include return type in the tool description
    
    Returns:
        dict: OpenAI tool description
    """
    # Get method signature
    signature = inspect.signature(method)
    params = signature.parameters
    
    # Get return type annotation if available
    return_type = signature.return_annotation
    return_type_str = str(return_type) if return_type is not inspect.Signature.empty else None
    
    # Get type hints for better type information
    type_hints = get_type_hints(method)
    
    # Create parameter properties for JSON Schema
    properties = {}
    required = []
    
    # Process each parameter (skip 'self' for class methods)
    for name, param in params.items():
        if name == 'self':
            continue
            
        # Parameter has no default value, so it's required
        if param.default is inspect.Parameter.empty:
            required.append(name)
        
        # Create parameter schema
        param_schema = {"type": "string"}  # Default type
        
        # Try to get type from type hints
        if name in type_hints:
            hint = type_hints[name]
            param_schema.update(python_type_to_json_schema(hint))
            
        # Add to properties
        properties[name] = param_schema
    
    # Parse docstring if requested
    description = ""
    returns_doc = None
    if include_docstring:
        docstring_info = parse_method_docstring(method)
        description = docstring_info.get("description", "")
        returns_doc = docstring_info.get("returns")
        
        # Update parameter descriptions and types from docstring
        for param_name, param_info in docstring_info.get("args", {}).items():
            if param_name in properties:
                properties[param_name]["description"] = param_info["description"]
    if response_api:
        tool = {
            "type": "function",
            "name": method.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
        if returns_doc:
            tool["description"] += f"\n\nReturns: {returns_doc}"
    else:
        # Create the OpenAI tool description
        tool = {
            "type": "function",
            "function": {
                "name": method.__name__,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
    
        if returns_doc:
            tool["function"]["description"] += f"\n\nReturns: {returns_doc}"
    
    if signature.return_annotation is ImageData and include_return:
        tool['returns'] = "A base64 encoded image in data URL format (data:image/png;base64,...)"

    return tool

def parse_method_docstring(method):
    """Parse a method's docstring to extract argument descriptions."""
    
    # Get the docstring
    doc = inspect.getdoc(method)
    if not doc:
        return {"args": {}, "returns": None, "description": None}
    
    # Get argument names from method signature
    signature = inspect.signature(method)
    param_names = list(signature.parameters.keys())
    
    # Remove 'self' if it's a class method
    if param_names and param_names[0] == 'self':
        param_names = param_names[1:]
    
    result = {
        "args": {},
        "returns": None,
        "description": None
    }
    
    # Extract the description (everything before Args/Parameters section)
    description = doc
    
    # Find the start of any sections (Args:, Parameters:, Returns:, etc.)
    section_pattern = r'\n\s*(?:Args|Parameters|Returns|Raises|Examples|Notes|Attributes):\s*\n'
    section_match = re.search(section_pattern, doc)
    
    if section_match:
        description = doc[:section_match.start()].strip()
    else:
        # If no sections found, the entire docstring is the description
        description = doc.strip()
    
    result["description"] = description
    
    # Check for different docstring styles
    
    # Google style: "Args:" or "Parameters:" section
    if 'Args:' in doc or 'Parameters:' in doc:
        # Find section start
        args_section = None
        if 'Args:' in doc:
            args_section = doc.split('Args:')[1]
        elif 'Parameters:' in doc:
            args_section = doc.split('Parameters:')[1]
        
        if args_section:
            # Extract until the next section (Returns, Raises, etc.)
            next_section = re.search(r'\n\s*[A-Z][a-z]+:\s*\n', args_section)
            if next_section:
                args_section = args_section[:next_section.start()]
            
            # Parse args section
            arg_pattern = re.compile(r'(\w+)(?:\s+\(([^)]+)\))?\s*:\s*(.*?)(?=\n\s*\w+\s*(?:\([^)]*\))?\s*:|$)', re.DOTALL)
            matches = arg_pattern.findall(args_section)
            
            for arg_name, arg_type, arg_desc in matches:
                result["args"][arg_name] = {
                    "type": arg_type.strip() if arg_type else None,
                    "description": arg_desc.strip()
                }
    
    # NumPy/reStructuredText style: "Parameters" section
    elif re.search(r'Parameters\s*\n\s*-+', doc):
        # Find parameters section
        params_match = re.search(r'Parameters\s*\n\s*-+\s*\n(.*?)(?:\n\s*[A-Za-z]+\s*\n\s*-+|$)', doc, re.DOTALL)
        if params_match:
            params_section = params_match.group(1)
            
            # Parse parameters
            param_pattern = re.compile(r'(\w+)\s*:\s*([^\n]*?)(?:\n\s+(.+?))?(?=\n\w+\s*:|$)', re.DOTALL)
            matches = param_pattern.findall(params_section)
            
            for arg_name, arg_type, arg_desc in matches:
                result["args"][arg_name] = {
                    "type": arg_type.strip(),
                    "description": arg_desc.strip()
                }
    
    # Returns section
    returns_match = re.search(r'Returns:\s*\n\s*(.*?)(?=\n\s*[A-Z]|$)', doc, re.DOTALL)
    if returns_match:
        result["returns"] = returns_match.group(1).strip()
    
    return result

def python_type_to_json_schema(py_type):
    """Convert Python type hints to JSON Schema types."""
    schema = {}
    
    # Handle primitive types
    if py_type is str:
        schema["type"] = "string"
    elif py_type is int:
        schema["type"] = "integer"
    elif py_type is float:
        schema["type"] = "number"
    elif py_type is bool:
        schema["type"] = "boolean"
    elif py_type is list or py_type is tuple:
        schema["type"] = "array"
    elif py_type is dict:
        schema["type"] = "object"
    
    # Handle typing module types (List, Dict, etc.)
    elif get_origin(py_type) is not None:
        origin = get_origin(py_type)
        args = get_args(py_type)
        
        if origin is list or origin is tuple:
            schema["type"] = "array"
            if args:
                schema["items"] = python_type_to_json_schema(args[0])
        elif origin is dict:
            schema["type"] = "object"
            if len(args) >= 2:
                # If keys are strings, we can define additionalProperties
                if args[0] is str:
                    schema["additionalProperties"] = python_type_to_json_schema(args[1])
        elif origin is Union:  # Now Union is properly imported
            # Handle Optional (Union with None)
            non_none_types = [t for t in args if t is not type(None)]
            if len(non_none_types) == 1:
                schema.update(python_type_to_json_schema(non_none_types[0]))
            else:
                # For complex unions, just use 'string' as a fallback
                schema["type"] = "string"
    
    # Default to string for complex types
    else:
        schema["type"] = "string"
    
    return schema


def parse_tools(functions: list[callable], response_api=False, include_return=True) -> tuple[list[dict], dict]:
    """
    Parse the docstrings of a list of functions to turn them
    into OpenAI-compatible tool descriptions.
    
    The functions (or methods) are supposed to have docstrings
    in either Google or NumPy style, with definitions of
    the arguments ('Args' section in Google style) entering
    the tool description. Use type annotations to indicate
    the type of the function arguments, e.g. 
    "def tool(a: int, b: float, c: str) -> str:. The return
    value must be either a string (the default) or else
    of type ImageData, for images. At the moment, only one
    image can be returned at a time.
    
    Args:
        functions (list[callable]): A list of functions
        These could also be methods of an object.
        include_return (bool): Whether to include the return type
            in the tool description.
        
    Returns:
        tools (list[dict]): A list of dictionaries, each of which
            represents a tool use description.
        available_functions (dict): A mapping from function names to functions.
    """
    tools=[]
    available_functions={}
    for function in functions:
        tools.append(method_to_openai_tool(function, response_api=response_api, include_return=include_return))
        available_functions[function.__name__]=function
    return tools, available_functions



def array_tool_wrapper(func: Callable):
    sig = inspect.signature(func)
    if '_fields' in sig.parameters:
        func_requires_fields = True
    else:
        func_requires_fields = False
    type_hints = get_type_hints(func)

    def is_iterable_type(tp):
        iterable_types = (jax.Array, np.array)
        if tp in iterable_types:
            return True
        else:
            return False

    @wraps(func)
    def wrapper(*args, _fields: dict, result_label: str, **kwargs):
        # insert _fields if required
        if func_requires_fields:
            kwargs['_fields'] = _fields

        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()

        new_arguments = {}
        for name, value in bound.arguments.items():
            expected_type = type_hints.get(name)
            if expected_type and is_iterable_type(expected_type):
                if not isinstance(value, str):
                    raise TypeError(f"Expected a string for parameter '{name}', got {type(value)}")
                if value not in _fields:
                    raise ValueError(f"Key '{value}' not found in fields dictionary.")
                new_arguments[name] = _fields[value]
            else:
                new_arguments[name] = value
        result = func(**new_arguments)
        
        if not isinstance(result, dict):
            raise TypeError(f"Expected a dictionary for the result, got {type(result)}")
        adapted_result = {}
        for key, value in result.items():
            adapted_result[result_label + "_" + key] = value
        return adapted_result
    

    # Modify docstring
    original_doc = inspect.getdoc(func) or ""
    doc_lines = original_doc.splitlines()
    new_doc_lines = []
    inside_args = False

    for line in doc_lines:
        stripped = line.strip()

        if stripped.startswith("Args:"):
            inside_args = True
            new_doc_lines.append(line)
            # insert result_label
            new_doc_lines.append("    result_label: The label under which the results will be saved in the format <result_label>_<key>.")
            continue

        if inside_args and (stripped.startswith("Returns:") or stripped.endswith(":") and not stripped.startswith("    ")):
            inside_args = False

        if inside_args and ":" in line:
            name_part, desc_part = line.split(":", 1)
            name = name_part.strip()
            desc = desc_part.strip()

            if type_hints.get(name) == jax.Array:
                new_desc = f"{desc}. Specified as the result_label it was saved under previously"
                new_doc_lines.append(f"    {name}: {new_desc}")
            elif name == "_fields":
                print('--Info: _fields should not be described in the docstring. Ignoring it.--')
                continue
            else:
                new_doc_lines.append(line)
        else:
            new_doc_lines.append(line)
            if stripped.startswith("Returns:"):
                new_doc_lines.append("Description and result_labels of the following variables:")

    # Adjust signature: jax.Array -> str (and keep self)
    new_params = []
    for name, param in sig.parameters.items():
        if name == "_fields":
            continue  # hide this
        if type_hints.get(name) == jax.Array:
            new_params.append(param.replace(annotation=str))
        else:
            new_params.append(param)
    # Add result_label: str
    new_param = inspect.Parameter(
        name="result_label",
        kind=inspect.Parameter.KEYWORD_ONLY,
        default=inspect.Parameter.empty,
        annotation=str,
    )
    new_params.append(new_param)

    new_sig = sig.replace(parameters=new_params, return_annotation=str)
    wrapper.__signature__ = new_sig
    wrapper.__doc__ = "\n".join(new_doc_lines)

    return wrapper

