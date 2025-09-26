from openai import OpenAI
from openai.types.responses import ResponseOutputMessage, ResponseReasoningItem, ResponseFunctionToolCall
import json
import re
from dataclasses import dataclass
import time
from .tool_utils import parse_tools
from.tool_utils import array_tool_wrapper
import jax
import numpy as np
from .tool_utils import ImageData
import multiprocessing
from sciexplorer.tool_utils import get_image
import concurrent.futures
import copy
import signal

# Define a timeout handler
def handler(signum, frame):
    raise TimeoutError("Function call timed out")


TIMEOUT = 30 #30 for all others -> increase for other hioddens?
def run_with_timeout(func, args=(), kwargs={}, timeout=TIMEOUT):
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)  # Set the timeout to the specified value
    try:
        # Call your long-running SciPy function here
        result = func(*args, **kwargs)
        signal.alarm(0)  # Cancel the alarm if finished in time
        return result
    except TimeoutError:
        print("--Function timed out!--")
        return {"error": f"Function {func.__name__} timed out after {timeout} seconds."}

def get_description(field, print_all=False) -> str:
        if print_all:
            if isinstance(field, ImageData):
                description = f"Image data."
            else:
                with np.printoptions(precision=2, suppress=True, threshold=int(1e4)):
                    description = str(field)
        else:
            np.set_printoptions(threshold=10)
            max_print_array_elemnts = 10
            if isinstance(field, jax.Array) or isinstance(field, np.ndarray):
                if np.sum(field.shape) > max_print_array_elemnts:
                    description = f"Array of shape {field.shape} with dtype {field.dtype}"
                else:
                    description = f"Array of shape {field.shape} with dtype {field.dtype} and with values {str(field)}"
            elif isinstance(field, list):
                if len(field) > max_print_array_elemnts:
                    description = f"List of length {len(field)}"
                else:
                    description = f"List of length {len(field)} with values {str(field)}"
            elif isinstance(field, set):
                if len(field) > max_print_array_elemnts:
                    description = f"Set of length {len(field)}"
                else:
                    description = f"Set of length {len(field)} with values {str(field)}"
            elif isinstance(field, tuple):
                if len(field) > max_print_array_elemnts:
                    description = f"Tuple of length {len(field)}"
                else:
                    description = f"Tuple of length {len(field)} with values {str(field)}"
            elif isinstance(field, dict):
                description = f"Dictionary with keys with values:\n"
                for key, value in field.items():
                    description += f"{key}: {get_description(value, print_all)}\n"
            elif isinstance(field, ImageData):
                description = f"Image data."
            else:
                description = str(field)
        return description

@dataclass
class Model:
    client: OpenAI
    model_name: str
    tools: list[dict]
    available_functions: dict
    
    """
    An object representing an LLM model. This allows you to
    so simple_query without tools and also query with tools.
    """
    
    def __init__(self, api_key, base_url, model_name):
        """
        Initialize the LLM model.
        
        Args:
          api_key (str): the API key from the provider
          base_url (str): the base URL
          model_name (str): the model name
          
        Returns:
          A Model object
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = self.client.models.list().data[0].id

        response = self.client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Please say only one word. The word is 'OK'."}
            ],
        )
        

    def set_tools(self, functions, response_api=False, **kwargs):
        """
        Set the tools by parsing docstrings of functions.
        
        Args:
            functions (list[callable]): a list of functions or object methods.
        """
        wrapped_functions = []
        for func in functions:
            wrapped_functions.append(array_tool_wrapper(func))
        tools, available_functions = parse_tools(wrapped_functions, response_api=response_api, **kwargs)
        self.tools=tools
        self.available_functions=available_functions

        
    def simple_query(self,*args):
        """
        Run a simple query on the LLM, without tools.
        
        Args:
           An arbitrary number of strings and/or images (ImageData).
           
        Returns:
           A string with the model answer.
        """
        content=[]
        for arg in args:
            entry={}
            if type(arg) is dict:
                entry['type']="image_url"
                entry['image_url']={"url": arg['image']}
            else:
                entry['type']="text"
                entry['text']=str(arg)
            content.append(entry)
        messages = [
          {"role": "user", "content": content}
        ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
        return response.choices[0].message.content
    
    def query(self,question, 
                          random_exp_func,
                            plot_function,
                            n_exps_to_run:int=0,
                            system_prompt=None, intermediate_prompts=None, 
                            result_tool=None, run_kwargs={}, human_input=False,
                            use_tools=True,
                            summarize_at_end=False,
                            replace_message_function=None,
                            separate_tool_calls=True,
                            response_api=False,
                            reasoning_level=None,
                            verbosity=None,
                            reasoning_summary=None,
                            timeout=None,
                          ):
        """Query the model with a question, using tools if specified.
        Args:
            question (str): The question to ask.
            random_exp_func (callable): A function to run experiments, only needed if n_exps_to_run > 0.
            plot_function (callable): A function to plot the results of the intial experiments.
            n_exps_to_run (int): Number of experiments to run before asking the question.
            system_prompt (str): Optional system prompt to set the context.
            intermediate_prompts (dict): Optional dictionary with intermediate prompts to use.
            result_tool (str): Optional name of the tool to call for the final answer.
            run_kwargs (dict): Additional keyword arguments for the run.
            human_input (bool): If True, take intermediate prompts from human user.
            use_tools (bool): If False, only reason without using tools.
            summarize_at_end (bool): If True, ask the LLM to summarize the conversation at the end.
            replace_message_function: (function) if set, this function will be called to replace messages of the assistant in the conversation..
                It should have the signature replace_message_function(last_message:str, time_step:int, tool_calls:list,) -> tuple(str, bool)
                and return the modified message of the assistant and a boolean indicating whether the LLM should keep reasoning after the replaced message.
            separate_tool_calls (bool): If True, the LLM will be asked to first only reason and then call only a tool.
            response_api (bool): If True, use the response API instead of chat completions.
            reasoning_level (str): amount of thingking in reasoning process. 'low', 'medium', 'high' or None (for default reasoning level).
            verbosity (str): amount of verbosity in answer. 'low', 'medium', 'high' or None (for default verbosity level).
            reasoning_summary (str): if set, the LLM will be asked to summarize its reasoning at each step.
            timeout (int): if set, each tool call will be limited to this many seconds (default None, uses TIMEOUT variable)
        Returns:
            messages (list): List of messages returned by the model.
            _fields (dict): Dictionary with fields extracted from the messages.
            """
        if use_tools:
            messages,past_history, _fields = ask_question_with_tool_iterative(
                client = self.client,
                model = self.model_name,
                question = question,
                tools = self.tools,
                available_functions= self.available_functions,
                system_prompt = system_prompt,
                intermediate_prompts = intermediate_prompts,
                result_tool = result_tool,
                random_exp_function = random_exp_func,
                plot_function=plot_function,
                n_exps_to_run=n_exps_to_run,
                human_input=human_input,
                summarize_at_end=summarize_at_end,
                replace_message_function=replace_message_function,
                separate_tool_calls=separate_tool_calls,
                response_api=response_api,
                reasoning_level=reasoning_level,
                verbosity=verbosity,
                reasoning_summary=reasoning_summary,
                timeout=timeout,
                **run_kwargs
            )
        else:
           raise NotImplementedError("Query without tools is not implemented yet. ")

        return messages, past_history, _fields

    
def get_repsonse_and_tool_calls(client:OpenAI, model_name, messages, tools, tool_choice, retry_no_choice=3, retry_api_error=5,
                    require_answer=False, repsonse_api=False, past_history=None, reasoning_level=None, last_reponse_id=None, allow_tools=True,
                    allow_multiple_tool_calls=True, verbosity=None, reasoning_summary=None) -> tuple:
    if repsonse_api:
        error_try = 0
        last_error = ""
        while error_try < retry_api_error:
            error_try += 1
            try:
                create_kwargs = dict(
                    model=model_name,
                    input=past_history,
                    tools=tools,
                    tool_choice=tool_choice
                )
                if reasoning_level is not None:
                    create_kwargs['reasoning'] = {"effort": reasoning_level}
                if reasoning_summary is not None:
                    reasoning_dict = create_kwargs.get('reasoning', {})
                    reasoning_dict['summary'] = reasoning_summary
                    create_kwargs['reasoning'] = reasoning_dict
                if verbosity is not None:
                    create_kwargs['text'] = {'verbosity': verbosity}
                # if last_reponse_id is not None:
                #     create_kwargs['previous_response_id'] = last_reponse_id
                response = client.responses.create(
                    **create_kwargs
                )
                curr_try = 0
                while not response.output and curr_try < retry_no_choice:
                    curr_try += 1
                    # wait for a second and try again
                    print("--No choices in the response. Retrying...--")
                    time.sleep(1)
                    response = client.responses.create(
                        **create_kwargs
                    )
                if not response.output:
                    raise ValueError("No output in the response. Please check the model and messages.")
                # response_message = response.choices[0].message
                # find the response message in the response
                response_message = None
                tool_calls_list = []
                for out in response.output:
                    if isinstance(out, ResponseOutputMessage):
                        response_message = out.content[0].text
                    elif isinstance(out, ResponseFunctionToolCall) and allow_tools:
                        tool_dict = out.to_dict()
                        #bring to same format as chat api
                        arguments = tool_dict.pop('arguments', None)
                        name = tool_dict.pop('name', None)
                        tool_dict['function'] = dict()
                        tool_dict['function']['arguments'] = arguments
                        tool_dict['function']['name'] = name
                        tool_calls_list.append(tool_dict)
                    elif isinstance(out, ResponseReasoningItem):
                        if out.summary:
                            print(f"ASSISTANT REASONING: {out.summary[0].text}")
                if tool_calls_list:
                    tool_calls = tool_calls_list #[toolcall.to_dict() for toolcall in tool_calls_list]
                else:
                    tool_calls = None
                print(f"ASSISTANT RESPONSE: {response_message}")
                if not allow_multiple_tool_calls and tool_calls:
                    tool_calls = tool_calls[:1]
                return response_message, tool_calls, response, True #response, True
            
            except Exception as e:
                last_error = e
                match=re.search("rate_limit_error",str(e))
                if match:
                    print("Got rate limit error...waiting for 10 seconds")
                    time.sleep(10 * error_try)
                    print("...done waiting")
                    # now try again:
                    try:
                        response = client.responses.create(
                        **create_kwargs
                    )
                    except Exception as e:
                        print(f"❌ Error during API call: {e}")
                else:
                    print(f"❌ Error during API call: {e}--")
    else:
        error_try = 0
        last_error = ""
        while error_try < retry_api_error:
            error_try += 1
            try:
                create_kwargs = dict(
                    model=model_name,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice
                )
                if reasoning_level is not None:
                    create_kwargs['reasoning_effort'] = reasoning_level
                if verbosity is not None:
                    create_kwargs['verbosity'] = verbosity

                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice
                )
                curr_try = 0
                while not response.choices and curr_try < retry_no_choice:
                    curr_try += 1
                    # wait for a second and try again
                    print("--No choices in the response. Retrying...--")
                    time.sleep(1)
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        tools=tools,
                        tool_choice='auto'
                    )
                if not response.choices:
                    raise ValueError("No choices in the response. Please check the model and messages.")
                response_message = response.choices[0].message
                
                if require_answer and not response_message.content:
                    tool_calls = response_message.tool_calls
                    if tool_calls:
                        response.choices[0].message.content = 'I want to call a tool next.'
                        response_message.content = 'I want to call a tool next.'
                    else:
                        raise ValueError("No content in the response message. Please check the model and messages.")
                # get tool_calls:
                if response_message.tool_calls and allow_tools:
                    tool_calls = [toolcall.to_dict() for toolcall in response_message.tool_calls]
                else:
                    tool_calls = None

                response_message_str = response_message.content
                print(f"ASSISTANT RESPONSE: {response_message_str}")
                if not allow_multiple_tool_calls and tool_calls:
                    tool_calls = tool_calls[:1]
                return response_message_str, tool_calls, response, True #response, True
            
            except Exception as e:
                last_error = e
                match=re.search("rate_limit_error",str(e))
                if match:
                    print("Got rate limit error...waiting for 10 seconds")
                    time.sleep(10)
                    print("...done waiting")
                    # now try again:
                    try:
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            tools=tools,
                            tool_choice='auto'
                        )
                    except Exception as e:
                        print(f"❌ Error during API call: {e}")
                else:
                    print(f"❌ Error during API call: {e}--")
    raise ValueError(f"An error occurred during API call: {last_error}")

def save_completion(client, model_name, messages, tools, tool_choice, retry_no_choice=3, retry_api_error=5,
                    require_answer=False) -> tuple:
    """
    Get a single completion from the model.
    
    Args:
        client: An initialized OpenAI client instance.
        model_name (str): The name of the model to use.
        messages (list): The messages to send to the model.
        tools (list): A set of OpenAI tool descriptions.
        tool_choice (str): The tool choice.
        retry_no_choice (int): Number of retries if no choices are found in the response.
        replace_message_function (callable, optional): A function to replace the message of the assistant in the conversation.
        
    Returns:
        message (str): The response message from the model.
        tool_calls (list): A list of tool calls made by the model (each tool_call is a dict)
        response: The full API response object.
        success (bool): True if the call was successful, False otherwise.
    """

    error_try = 0
    last_error = ""
    while error_try < retry_api_error:
        error_try += 1
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice
            )
            curr_try = 0
            while not response.choices and curr_try < retry_no_choice:
                curr_try += 1
                # wait for a second and try again
                print("--No choices in the response. Retrying...--")
                time.sleep(1)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=tools,
                    tool_choice='auto'
                )
            if not response.choices:
                raise ValueError("No choices in the response. Please check the model and messages.")
            response_message = response.choices[0].message
            if require_answer and not response_message.content:
                tool_calls = response_message.tool_calls
                if tool_calls:
                    response.choices[0].message.content = 'I want to call a tool next.'
                else:
                    raise ValueError("No content in the response message. Please check the model and messages.")
            return response, True
        
        except Exception as e:
            last_error = e
            match=re.search("rate_limit_error",str(e))
            if match:
                print("Got rate limit error...waiting for 10 seconds")
                time.sleep(10)
                print("...done waiting")
                # now try again:
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        tools=tools,
                        tool_choice='auto'
                    )
                except Exception as e:
                    print(f"❌ Error during API call: {e}")
            else:
                print(f"❌ Error during API call: {e}--")
    return f"An error occurred during API call: {last_error}", False
    
    


def run_tool(tool_call, function_to_call, messages:list, function_name:str, _fields:dict, separate_image_result=True, timeout=TIMEOUT) -> list:
    try:
        function_args = json.loads(tool_call.function.arguments)
        print(f"--Running tool '{function_name}' with arguments {function_args}--")
        tool_result = run_with_timeout(function_to_call, args=(), kwargs=dict(_fields=_fields, **function_args), timeout=timeout)
                                             #function_to_call(_fields=_fields, **function_args)
        if not isinstance(tool_result, dict):
            tool_result = {"last_error":f"""Tool '{function_name}' did not return a dictionary. 
                                Tool does not follow the expected format."""}
    except json.JSONDecodeError:
        tool_result = {"last_error": f"Error: Invalid JSON format in arguments for tool '{function_name}'."}
    except Exception as e:
        tool_result = {"last_error": f"Error executing tool '{function_name}': {e}"}

    # save results
    message_to_append = {
        "tool_call_id": tool_call.id,
        "role": "tool",
        "name": function_name,
        'content': [],
    }

    
    saved_message = "Saved the following results in the format <result_label>: <result>:"
    for key, value in tool_result.items():
        _fields[key] = value
        if isinstance(value, ImageData) and separate_image_result:
            # Handle image data separately
            saved_message += f"\n{key}: Image data is provided later by the user."
        else:
            if isinstance(value, ImageData):
                base64_data = value.image
                if base64_data:
                    message_to_append["content"].append({"type": "image_url", "image_url": {"url": base64_data}})
            else:    
                saved_message += f"\n{key}: {get_description(value)}"
    message_to_append["content"].append({"type": "text", "text": saved_message})
    messages.append(message_to_append)

    # messages.append({
    #     "tool_call_id": tool_call.id,
    #     "role": "tool",
    #     "name": function_name,
    #     "content": saved_message,
    # })
    print(f"TOOL {tool_call.id}: {saved_message}")
    # create image messages:
    for key, value in tool_result.items():
        if isinstance(value, ImageData) and separate_image_result:
            base64_data = value.image
            if base64_data:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Here's the image {key} generated by the tool:"},
                        {"type": "image_url", "image_url": {"url": base64_data}}
                    ]
                })
                print(f'USER: {messages[-1]["content"][0]["text"]}')
                print('-- Image sent--')
            else:
                print('--Warning: No image data found in tool result.--')
    return messages, _fields


def run_tool_dict(tool_call, function_to_call, messages:list, function_name:str, _fields:dict, separate_image_result=True, past_history=None,
                  user_text_key = 'input_text', image_url_key = 'input_image',
                  function_id_key="",
                function_result_key="",
                function_content_key= "",
                type_key="",
                f_call_key="",
                response_api=True,
                timeout=TIMEOUT) -> list:
    try:
        function_args = json.loads(tool_call['function']['arguments'])
        print(f"--Running tool '{function_name}' with arguments {function_args}--")
        tool_result = run_with_timeout(function_to_call, args=(), kwargs=dict(_fields=_fields, **function_args), timeout=timeout)
                                             #function_to_call(_fields=_fields, **function_args)
        if not isinstance(tool_result, dict):
            tool_result = {"last_error":f"""Tool '{function_name}' did not return a dictionary. 
                                Tool does not follow the expected format."""}
    except json.JSONDecodeError:
        tool_result = {"last_error": f"Error: Invalid JSON format in arguments for tool '{function_name}'."}
    except Exception as e:
        tool_result = {"last_error": f"Error executing tool '{function_name}': {e}"}

    # save results
    message_to_append = {
        function_id_key: tool_call[f_call_key],
        type_key: function_result_key, # 'role': 'tool',
        #"name": function_name,
        function_content_key: "", # content
    }
    if not response_api:
        message_to_append["name"] = function_name


    saved_message = "Saved the following results in the format <result_label>: <result>:"
    for key, value in tool_result.items():
        _fields[key] = value
        if isinstance(value, ImageData) and separate_image_result:
            # Handle image data separately
            saved_message += f"\n{key}: Image data is provided later by the user."
        else:
            if isinstance(value, ImageData):
                base64_data = value.image
                if base64_data:
                    message_to_append[function_content_key].append({"type": image_url_key, "image_url": {"url": base64_data}})
            else:    
                saved_message += f"\n{key}: {get_description(value)}"
    message_to_append[function_content_key] = saved_message
    #message_to_append['content'] = saved_message
    messages.append(message_to_append)
    if past_history is not None:
        past_history.append(message_to_append)

    # messages.append({
    #     "tool_call_id": tool_call.id,
    #     "role": "tool",
    #     "name": function_name,
    #     "content": saved_message,
    # })
    print(f"TOOL {tool_call['id']}: {saved_message}")
    # create image messages:
    for key, value in tool_result.items():
        if isinstance(value, ImageData) and separate_image_result:
            base64_data = value.image
            if response_api:
                    img_res = base64_data
            else:
                img_res = {"url": base64_data}
            if base64_data:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": user_text_key, "text": f"Here's the image {key} generated by the tool:"},
                        {"type": image_url_key, "image_url": img_res} # {"url": base64_data}}
                    ]
                })
                past_history.append(messages[-1])
                print(f'USER: {messages[-1]["content"][0]["text"]}')
                print('-- Image sent--')
            else:
                print('--Warning: No image data found in tool result.--')
    return messages, past_history, _fields




def ask_question_with_tool_iterative(
    client: OpenAI,
    model: str,
    question: str,
    tools: list, # list of tools to be used
    available_functions: dict, # mapping of function names to actual functions
    random_exp_function: callable,
    plot_function: callable,
    n_exps_to_run:int,
    max_steps: int = 5,
    max_tools: int = 5, 
    final_response_key: str = '', #"STOP EXPLORATION", # key to look for in the final response
    allow_multiple_tool_calls: bool = True, 
    keep_only_one_image: bool = False,
    max_tries_for_correct_tool_call: int = 1,
    system_prompt: str = None,
    intermediate_prompts: dict = None,
    result_tool: str = None, 
    human_input: bool = False,
    summarize_at_end: bool = False,
    replace_message_function=None, # function to replace the message of the assistant
    separate_tool_calls=False,
    response_api = True, # if True, use the response API instead of chat completions
    reasoning_level: str = "low", # if not None, use the reasoning level in the
    verbosity=None,
    reasoning_summary=None,
    timeout=None,
):
    if timeout is None:
        timeout = TIMEOUT
    print(f"Timeout for function calls set to: {timeout}")

    #multiprocessing.set_start_method('fork', force=True)

    # previously saved fields

    if response_api:
        user_text_key = 'input_text'
        image_url_key = 'input_image'
        function_result_key = 'function_call_output' #"function_call_output" # tool
        function_id_key = "call_id" # tool_call_id
        f_call_key = "call_id" # "id"
        function_content_key = 'output' # "content"
        type_key = 'type'
    else:
        user_text_key = 'text'
        image_url_key = 'image_url'
        function_result_key = 'tool'
        function_id_key = "tool_call_id" # tool_call_id
        f_call_key = "id" # "id"
        function_content_key = 'content' # "content"
        type_key = 'role' # 'role'
    
    _fields = {}

    past_tool_calls = []

    if intermediate_prompts is None:

            intermediate_prompts = {'ask_for_reasoning': {'content': f"""If already present, what do the past tool results teach you about the question?
    Describe which tool you want to call next and why you want to call it in plain text.
    Don't yet call the tool, just describe what you want to do.
    If you don't need to call anymore tools, you can instead provide the final answer."""},
                                    'ask_for_tool_use':{'content': f"""Now call the tools.
    Don't include any text but the tool request in the format defined in your system prompt."""},
                                    'wrong_format_tool_call': {'content': f"""Badly formatted tool call, please try again and remember the proper format defined in your system prompt.
    Especially remeber to use required brackets."""},
                                    'ask_final_answer': {'content': f"""Please provide a final answer without using any tools."""}
                                    }
            
    assert 'ask_for_reasoning' in intermediate_prompts, "intermediate prompt 'ask_for_reasoning' not found"
    assert 'ask_for_tool_use' in intermediate_prompts, "intermediate prompt 'ask_for_tool_use' not found"
    assert 'wrong_format_tool_call' in intermediate_prompts, "intermediate prompt 'wrong_format_tool_call' not found"
    assert 'ask_final_answer' in intermediate_prompts, "intermediate prompt 'ask_final_answer' not found"


    if system_prompt:
        system_message = {
            "role": "system",
            "content": system_prompt
        }
        messages = [system_message,]
        print(f'SYSTEM: {messages[-1]["content"]}')
    else:
        messages = []
    first_message = question 

    if n_exps_to_run > 0:
        # add observations:
        first_message = question + '\nTo help you answer the question, I will run some experiments.'
        messages.append({
                        "role": "user",
                        "content": first_message})
        print(f'USER: {messages[-1]["content"]}')
        # run exps and communicate results:
        for i in range(n_exps_to_run):
            exp_result = random_exp_function()

            #exp_result = random_exp_function()
            # _fields[f'exp_result_{i}'] = exp_result
            # plot_function(exp_result)
            # ecoded_image = get_image()

            saved_message = f"For experiment {i}, saved the following results in the format <result_label>: <result>:"
            for key, value in exp_result.items():
                adapted_key = f'exp_{i}_' + key
                _fields[adapted_key] = value
                saved_message += f"\n{adapted_key}: {get_description(value)}"
            messages.append({
                        "role": "user",
                        "content": saved_message})            
            print(f'USER: {messages[-1]["content"]}')

            if plot_function:
                plot_function(exp_result)
                encoded_image = get_image()
                if response_api:
                    img_res = encoded_image.image
                else:
                    img_res = {"url": encoded_image.image}
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": user_text_key, "text": f"Experiment {i} plot."},
                        {"type": image_url_key, "image_url": img_res}#{"url": encoded_image.image}}
                    ]
                })
            print(f'USER: {messages[-1]["content"][0]["text"]}')

        

        first_message = ""
        if not plot_function:
            first_message += 'Now plot all the experiments before doing anything else!'
    if result_tool:
        first_message += f"""\nYou can reason for up to {max_steps} steps and call up to {max_tools} tools to help you find the answer.
After your exploration please call the tool {result_tool} to provide your final answer."""
    if final_response_key:
        first_message += f"""\nEnd your final message with '{final_response_key}'. 
Never use this key before answering all questions of the user and providing your final answer."""
    messages.append({
                    "role": "user",
                     "content": first_message})
    print(f'USER: {messages[-1]["content"]}')


    current_step = 0
    total_tool_calls = 0
    past_history = copy.deepcopy(messages) # save the past history of messages
    last_reponse_id = None
    break_loop = False
    while current_step < max_steps and not break_loop:
        #print(f"\n🔄 Step {current_step + 1}/{max_steps} - Sending request to AI...")
        # first reason without tools
        user_prompt = intermediate_prompts['ask_for_reasoning']['content']
        if human_input:
            print(f"Message would be: {user_prompt} - Please enter the message:")
            user_prompt = input()
        messages.append({
                    "role": "user",
                    "content":f"\n You potentially have up to {max_steps - current_step} steps left and can call up to {max_tools - total_tool_calls} tools. " +
                      user_prompt})
        past_history.append(messages[-1]) # save the last user message to the past history
        print(f'USER: {messages[-1]["content"]}')
        #messages = potentially_remove_image_from_messages(messages)
        message, tool_calls, response, success = get_repsonse_and_tool_calls(client, model, messages, tools, 'auto', require_answer=separate_tool_calls,
                                                        repsonse_api=response_api, past_history=past_history, reasoning_level=reasoning_level, last_reponse_id=last_reponse_id,
                                                        allow_tools = not separate_tool_calls, allow_multiple_tool_calls=allow_multiple_tool_calls,
                                                        verbosity=verbosity, reasoning_summary=reasoning_summary)            
        tools_not_to_run = []
        if response_api:
            for r in response.output:
                if not isinstance(r, ResponseFunctionToolCall) or not separate_tool_calls:
                    # if not a tool call or not separate tool calls, add to past history
                    past_history.append(r.to_dict())
                else:
                    past_history.append(r.to_dict())
                    tools_not_to_run.append(r.to_dict())

            # append result that tool cannot be run to the messages
            for tool_not_to_run in tools_not_to_run:
                    messages.append({
                        function_id_key: tool_not_to_run[f_call_key],
                        type_key: function_result_key, # 'role': 'tool',
                        function_content_key: "Tools can only be run in the next message."
                    })
                    past_history.append(messages[-1]) # save the last message to the past history
                    print(f"TOOL {tool_not_to_run[f_call_key]}: {messages[-1][function_content_key]}")
                

            #past_history.extend([r.to_dict() for r in response.output])
        last_reponse_id = response.id


        if separate_tool_calls:
            messages.append({
                    "role": "assistant",
                    "content": message,
                    })
            user_prompt = intermediate_prompts['ask_for_tool_use']['content']
            messages.append({
                        "role": "user",
                        "content":  user_prompt})
            print(f'USER: {messages[-1]["content"]}')
            past_history.append(messages[-1]) # save the last user message to the past history
            message, tool_calls, response, success = get_repsonse_and_tool_calls(client, model, messages, tools, 'auto', require_answer=False,
                                                        repsonse_api=response_api, past_history=past_history, reasoning_level=reasoning_level, last_reponse_id=last_reponse_id,
                                                        allow_tools=True, allow_multiple_tool_calls=allow_multiple_tool_calls,
                                                        verbosity=verbosity, reasoning_summary=reasoning_summary,)  
            if response_api:    
                past_history.extend([r.to_dict() for r in response.output])
            last_reponse_id = response.id

        tool_call_message= {
                "role": "assistant",
                "content": message,
                }
        if tool_calls:
            tool_call_message["tool_calls"] = tool_calls

        messages.append(tool_call_message)
        #print(f'ASSISTANT: {messages[-1]["content"]}')

        current_step += 1 
        if tool_calls:
            initial_messages = len(messages)
            initial_past_history = len(past_history)
            for tool_call in tool_calls:
                #function_name = tool_call.function.name

                function_name = tool_call['function']['name']
                
                if total_tool_calls >=max_tools and function_name != result_tool:
                    messages.append({
                                function_id_key: tool_call[f_call_key],
                                type_key: function_result_key, # 'role': 'tool',
                                #"name": function_name,
                                function_content_key: f"Maximum amount of tool calls reached.",
                                })
                    if not response_api:
                            messages[-1]["name"] = function_name
                    past_history.append(messages[-1]) # save the last message to the past history
                    print(f"TOOL {tool_call['id']}: {messages[-1][function_content_key]}")
                    current_step = max_steps
                else:
                    function_to_call = available_functions.get(function_name)
                    if not function_to_call:
                        messages.append({
                            function_id_key: tool_call[f_call_key],
                            type_key: function_result_key, # 'role': 'tool',
                            #"name": function_name,
                            function_content_key: f"Error: Tool '{function_name}' not found.",
                            })
                        if not response_api:
                            messages[-1]["name"] = function_name
                        past_history.append(messages[-1])
                        print(f"TOOL {tool_call['id']}: {messages[-1][function_content_key]}")
                    else:
                        messages, past_history, _fields = run_tool_dict(tool_call, function_to_call, 
                                                                        messages, function_name, _fields, 
                                                                        past_history=past_history,
                                                                        user_text_key=user_text_key,
                                                                        image_url_key=image_url_key,
                                                                        function_id_key=function_id_key,
                                                                        function_result_key=function_result_key,
                                                                        f_call_key=f_call_key,
                                                                        function_content_key=function_content_key,
                                                                        type_key=type_key,
                                                                        response_api=response_api,
                                                                        timeout=timeout)
                        past_tool_calls.append(function_name)
                        if function_name  == result_tool:
                            print(f"--Result tool {function_name} used. Returning messages--")
                            break_loop=True
                            break
                total_tool_calls += 1
            # move user messages including the image data to end of the messages
            message_copy = copy.deepcopy(messages)
            n_removed = 0
            for i in range(initial_messages, len(messages)):
                if message_copy[i].get("role", None) == "user":
                    # move the user message to the end of the messages
                    messages.pop(i - n_removed)
                    messages.append(message_copy[i])
                    n_removed += 1
            # do the same for the past history
            past_history_copy = copy.deepcopy(past_history)
            n_removed = 0
            for i in range(initial_past_history, len(past_history)):
                if past_history_copy[i].get("role", None) == "user":
                    # move the user message to the end of the past history
                    past_history.pop(i - n_removed)
                    past_history.append(past_history_copy[i])
                    n_removed += 1

    if not break_loop:
        messages.append({
            "role": "user",
            "content": f"""Please call the tool {result_tool} to provide your final answer. Do not use any other tools!"""})
        past_history.append(messages[-1])
        print(f'USER: {messages[-1]["content"]}')
        final_message, tool_calls, response, success = get_repsonse_and_tool_calls(client, model, messages, tools, 'auto', require_answer=separate_tool_calls,
                                                            repsonse_api=response_api, past_history=past_history, reasoning_level=reasoning_level, last_reponse_id=last_reponse_id,
                                                            allow_multiple_tool_calls=False,
                                                            verbosity=verbosity, reasoning_summary=reasoning_summary,)            
        if response_api:
            past_history.extend([r.to_dict() for r in response.output])
                    
        last_reponse_id = response.id
        messages.append({
            "role": "assistant",
            "content": final_message,
        })
        # only one tool call is allowed
        tool_call = tool_calls[0] if tool_calls else None
        if tool_call:
            messages[-1]["tool_calls"] = [tool_call, ]
            function_name = tool_call['function']['name']

            function_to_call = available_functions.get(function_name)
            messages, past_history, _fields = run_tool_dict(tool_call, function_to_call, messages, 
                                                            function_name, _fields, past_history=past_history,
                                                            user_text_key=user_text_key,
                                                            image_url_key=image_url_key,
                                                            function_id_key=function_id_key,
                                                            function_result_key=function_result_key,
                                                            f_call_key=f_call_key,
                                                            function_content_key=function_content_key,
                                                            type_key=type_key,
                                                            response_api=response_api,
                                                            timeout=timeout)   
        
        else:
            print("--No tool calls found in the final response. Returning messages--")
            raise ValueError("--No tool calls found in the final response. Returning messages--")

    
            
    if summarize_at_end:
        messages.append({
            "role": "user",
            "content": "Please summarize the previous conversation."})
        past_history.append(messages[-1])
        print(f'USER: {messages[-1]["content"]}')
        summary_message, tool_calls, response, success = get_repsonse_and_tool_calls(client, model, messages, tools, 'auto', require_answer=separate_tool_calls,
                                repsonse_api=response_api, past_history=past_history, reasoning_level=reasoning_level, last_reponse_id=last_reponse_id,
                                allow_multiple_tool_calls=allow_multiple_tool_calls, verbosity=verbosity, reasoning_summary=reasoning_summary,)            
        if response_api:
            for r in response.output:
                if not isinstance(r, ResponseFunctionToolCall):
                    past_history.append(r.to_dict())
        last_reponse_id = response.id
        messages.append({
            "role": "assistant",
            "content": summary_message,
        })

    return messages, past_history, _fields