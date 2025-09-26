
import datetime
import json

from sciexplorer.model_utils import Model, get_repsonse_and_tool_calls, run_tool_dict
from sciexplorer.runner import get_dict_hash, uniquify

import base64
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
import os




system_prompt = 'You are an expert scientist.'

first_message = '''<<In the following, you are going to hold multiple conversations.
In each conversation, you will try to solve the same task.
During a specific conversation, you will not be able to remeber the other conversations.
After the conversations, you need to decide in which conversation you solved the task more accurately.>>'''

middle_message = '''<<Here is the conversation number <i> of <n> conversations.>>'''

final_message = '<<You completed the conversations. ' \
'Please explain in which of the conversations you best solved the task and why.' \
'Remember that you tried to solve the same task in all conversations.' \
'Critically evaluate your past conclusions given the new information you have from all conversations.' \
'Fully trust the additional data from the conversations but do not fully trust the conclusions you drew from it in the past.'
#'Also call the final_tool to save your judgement.>>' \

final_tool_message = '<<Now call the final_tool to save your judgement.>>'


def final_tool(_fields:dict, best_conversation: int):
    """Tool to save the best conversation.
    Args:
        best_conversation: The index of the best conversation. Starting from 0.
    Returns:
        saved_result (dict): A message indicating the result of the save operation.
    """
    res = {'best_conversation': best_conversation}
    return res

def compare_conversations(api_key:str, base_url:str, model_name:str, 
              exp_path_list:list, save_to_file:str='', 
              system_prompt:str=system_prompt, final_tool:callable=final_tool,
              response_api:bool=False, reasoning_level:str='high', verbosity:str='high', reasoning_summary:str='auto',
              quality_measure:str=None):
    """Let the model decide which experiment is better.
    Args:
        api_key (str): The API key for the model.
        base_url (str): The base URL for the model API.
        model_name (str): The name of the model to use.
        model (Model): The model to use for comparison.
        exp_path_list (list): List of paths to the experiment files.
        save_to_file (str): Path to save the comparison results.
        system_prompt (str): The system prompt to use for the model.
        final_tool (callable): A function to call at the end to save the best conversation.
        response_api (bool): Whether to use the response API format.
    Returns:
        dict: A dictionary containing the comparison results, including the best conversation.
    """
    model = Model(api_key=api_key, base_url=base_url, model_name=model_name)

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

    conversations = []
    past_histories = []
    conversations_qualities = []
    for path in exp_path_list:
        # Load the experiment data
        with open(f'{path}', 'r') as f:
            data = json.load(f)
        if response_api:
            past_histories.append(data['past_history'])
        else:
            past_histories.append([])
        if quality_measure:
            conversations_qualities.append(data["result"].get(quality_measure, None))
        conversations.append(data['messages'])
    quality_str = ""
    if quality_measure:
        for i, q in enumerate(conversations_qualities):
            quality_str += f"Conversation {i} has {quality_measure} of {q}.\n"
    
    print(quality_str)
  
    past_history = []
    if system_prompt:
        system_message = {
            "role": "system",
            "content": system_prompt
        }
        messages = [system_message,]
        if response_api: past_history.append(system_message)
    else:
        messages = []
    messages.append({"role": "user",
                            "content": first_message})
    if response_api: past_history.append(messages[-1])

    for i, (conversation, ph) in enumerate(zip(conversations, past_histories)):
        mid_mes_adapted = middle_message.replace("<i>", str(i)).replace("<n>", str(len(conversations)))
        messages.append({"role": "user",
                                "content":  mid_mes_adapted})
        if response_api: past_history.append(messages[-1])
        messages.extend(conversation)
        if response_api: past_history.extend(ph)
            
    messages.append({"role": "user",
                        "content": final_message})
    if response_api: past_history.append(messages[-1])

    model.set_tools([final_tool], response_api=response_api)
    
    
    message, tool_calls, response, success = get_repsonse_and_tool_calls(model.client, model.model_name, messages, model.tools, 'auto', require_answer=True,
                                                        repsonse_api=response_api, past_history=past_history, reasoning_level=reasoning_level, last_reponse_id=None,
                                                        allow_tools = True, allow_multiple_tool_calls=False,
                                                        verbosity=verbosity, reasoning_summary=reasoning_summary)
    if response_api:    
        past_history.extend([r.to_dict() for r in response.output])

    tool_call_message= {
                "role": "assistant",
                "content": message,
                }

    messages.append(tool_call_message)


    # model.set_tools([final_tool], response_api=response_api)
    if not tool_calls:
        messages.append({'role': 'user', 
                        'content': final_tool_message})
        if response_api: past_history.append(messages[-1])

        tool_message, tool_calls, response, success = get_repsonse_and_tool_calls(model.client, model.model_name, messages, model.tools, 'auto', require_answer=True,
                                                            repsonse_api=response_api, past_history=past_history, reasoning_level=reasoning_level, last_reponse_id=None,
                                                            allow_tools = True, allow_multiple_tool_calls=False,
                                                            verbosity=verbosity, reasoning_summary=reasoning_summary)
        if response_api:    
            past_history.extend([r.to_dict() for r in response.output])
    
    best_conversation = None
    if tool_calls:
        tool_call = tool_calls[0]
        function_name = tool_call['function']['name']
        assert function_name == final_tool.__name__, f"Expected final tool to be called, but got {function_name}."
        function_to_call = model.available_functions.get(function_name)
        messages, past_history, _ = run_tool_dict(tool_call, function_to_call, 
                                                        messages, function_name, {}, 
                                                        past_history=past_history,
                                                        user_text_key=user_text_key,
                                                        image_url_key=image_url_key,
                                                        function_id_key=function_id_key,
                                                        function_result_key=function_result_key,
                                                        f_call_key=f_call_key,
                                                        function_content_key=function_content_key,
                                                        type_key=type_key,
                                                        response_api=response_api)
        best_conversation = json.loads(tool_call['function']['arguments'])


    everything=dict(
        messages=messages,
        model=model.model_name,
    )
    input_hash = get_dict_hash(everything)
    everything['messages'] = messages
    everything['exp_path_list'] = exp_path_list
    everything['conversations_qualities'] = conversations_qualities
    everything['best_conversation'] = best_conversation
    everything['past_history'] = past_history
    everything['input_hash'] = input_hash
    everything['timestamp']=str(datetime.datetime.now())
    everything['exp_path_list'] = exp_path_list

    if save_to_file:

        with open(uniquify(save_to_file + "_" + input_hash ), "w") as file:
            json.dump(everything, file)

    return everything, message, quality_str





def summarize_conversation(api_key:str, base_url:str, model_name:str, 
              exp_path:str, save_path:str='', 
              system_prompt:str=system_prompt,
              figure_name=None,
              response_api:bool=False, reasoning_level:str='high', verbosity:str='high', reasoning_summary:str='auto',
              first_message:str="", final_message:str=""):
    """Let the model summarize a conversation.
    Args:
        api_key (str): The API key for the model.
        base_url (str): The base URL for the model API.
        model_name (str): The name of the model to use.
        model (Model): The model to use for comparison.
        exp_path (str): Path to the experiment file.
        save_path (str): Path to folder where to save figures and result.
        figure_name (str): Base name of the figures to save.
        system_prompt (str): The system prompt to use for the model.
        final_tool (callable): A function to call at the end to save the best conversation.
        response_api (bool): Whether to use the response API format.
    """
    os.makedirs(save_path, exist_ok=True)

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
    model = Model(api_key=api_key, base_url=base_url, model_name=model_name)

    if not figure_name:
        figure_name = 'figure'

    if not first_message:
        first_message = '''<<In the following, you are going to hold a conversation.
In this conversation, you will try to solve a specific task.>>'''
    if not final_message:
        final_message = '<<You completed the conversation. '\
    'Please create a two page summary of the conversation paying special attention to the following aspects: '\
    '1. State the goal of the conversation.\n'\
    '2. List the steps you took to reach the goal and why you took them. '\
    'State what you learned from each step. '\
    'Pay special attention to visualizations and analysis routines you created.\n'\
    '3. State your final conclusion and why you think it is correct.\n'\
    'Make sure to format your summary in Latex so that it can be directly used in a scientific publication. '\
    'However, do not use any \section or \subsection commands. You can use e.g. \bf instead. ' \
    'Make sure not to use unicode characters in the Latex code. ' \
    'Include also key figures. You can include them with the \includegraphics command. Use labels for the figures and reference them in your text.' \
    f'The figures of the conversation are saved as {figure_name}_<i>.jpg, where i indicates the number of the figure in the conversation starting at 1.'
    #'Include key figures you created during the conversation, by first saving them with the save_figure tool and then including them with \includegraphics commands.>>'


    # Load the experiment data
    with open(f'{exp_path}', 'r') as f:
        data = json.load(f)
    if response_api:
        history_conv=data['past_history']
    else:
        history_conv = []
    message_conv = data['messages']


    def save_figure(figure_save_name:str, image):
        """Saves the ith figure of the conversation to file, where i is the figure_idx.
        The name of the figure will be <figure_save_name>.pdf"""
        dpi = 300  # screen DPI for display
        figsize = (image.width / dpi, image.height / dpi)
        plt.figure(figsize=figsize, dpi=dpi)
        plt.imshow(image)
        plt.axis('off')  # Hide axes
        save_name = f"{figure_save_name}.jpg"
        plt.savefig(save_name)
        return save_name
    figure_list = []
    figure_idx = 1
    for message in message_conv:
        role = message.get('role', 'unknown')
        if role == 'unknown':
            role = message.get('type', 'unknown')
        content = message.get('content', [])
        tools = message.get('tool_calls', [])
        if not isinstance(content, list):
             content = [content, ]
        for c in content: 
            if (isinstance(c, dict) and c['type'] == image_url_key):
                if isinstance(c.get('image_url'), str):
                    # If the image_url is a string, it is likely a base64 encoded image
                    image_data = c['image_url']
                elif isinstance(c.get('image_url'), dict):
                    image_data = c['image_url']['url']

                base64_str = image_data.split(',')[1] if ',' in image_data else image_data
                # Decode the base64 string
                image_data = base64.b64decode(base64_str)
                # Open the image
                image = Image.open(BytesIO(image_data))
                figure_list.append(image)
                save_figure(os.path.join(save_path, f"{figure_name}_{figure_idx}"), image)
                figure_idx += 1

    past_history = []
    if system_prompt:
        system_message = {
            "role": "system",
            "content": system_prompt
        }
        messages = [system_message,]
        if response_api: past_history.append(system_message)
    else:
        messages = []
    messages.append({"role": "user",
                            "content": first_message})
    if response_api: past_history.append(messages[-1])

    messages.extend(message_conv)
    if response_api: past_history.extend(history_conv)

    messages.append({"role": "user",
                        "content": final_message})
    if response_api: past_history.append(messages[-1])




    model.set_tools([], response_api=response_api)
    
    message, tool_calls, response, success = get_repsonse_and_tool_calls(model.client, model.model_name, messages, model.tools, 'auto', require_answer=True,
                                                        repsonse_api=response_api, past_history=past_history, reasoning_level=reasoning_level, last_reponse_id=None,
                                                        allow_tools = True, allow_multiple_tool_calls=False,
                                                        verbosity=verbosity, reasoning_summary=reasoning_summary)
    if response_api:    
        past_history.extend([r.to_dict() for r in response.output])
    messages.append({
                "role": "assistant",
                "content": message,
                })


    everything=dict(
        messages=messages,
        model=model.model_name,
    )
    input_hash = get_dict_hash(everything)
    everything['messages'] = messages
    everything['exp_path'] = exp_path
    everything['past_history'] = past_history
    everything['input_hash'] = input_hash
    everything['timestamp']=str(datetime.datetime.now())
    
    if save_path:
        with open(uniquify(os.path.join(save_path, f"all_{input_hash}")), "w") as file:
            json.dump(everything, file)

    return everything, message
