import json
from IPython.display import display, HTML
from html import escape

import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

def count_seen_experiments(file_path, exp_funcs:dict):
    """
    Count the number of seen experiments in a JSON file.
    Args:
        file_path: Path to the JSON file containing experiment data.
        exp_funcs: Dictionary with keys being tool names and values beiong amount of initializations per tool."""
    with open(f'{file_path}', 'r') as f:
        data = json.load(f)
    messages = data.get('messages', [])
    total_counts = 0
    for message in messages:
        role = message.get('role', 'unknown')
        if role == 'tool':
            tool_name = message.get('name', '')
            for sim_tool_name, count in exp_funcs.items():
                if tool_name == sim_tool_name:
                    total_counts +=  count
        elif role == 'assistant':
            tool_calls = message.get('tool_calls', [])
            for tool in tool_calls:
                function_name = tool.get('function', {}).get('name', '')
                for sim_tool_name, count in exp_funcs.items():
                    if function_name == sim_tool_name:
                        total_counts +=  count
    return total_counts


def render_json_to_html(json_string):
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError:
        return "<p style='color:red;'>Invalid JSON</p>"

    def render(data):
        if isinstance(data, dict):
            html = "<ul>"
            html += "<strong>ARGUMENTS:</strong><br>"
            for key, value in data.items():
                html += "<li><strong>{}</strong>: ".format(escape(str(key)))
                if (key == "code" or key=="rhs") and isinstance(value, str):
                    html += "<pre><code>{}</code></pre>".format(escape(value))
                else:
                    html += render(value)
                html += "</li>"
            html += "</ul>"
            return html
        elif isinstance(data, list):
            return "<ul>" + "".join(f"<li>{render(item)}</li>" for item in data) + "</ul>"
        else:
            return escape(str(data))

    return f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            ul {{ list-style-type: none; padding-left: 20px; }}
            pre {{
                background-color: #000000;
                color: #f8f8f2;
                padding: 10px;
                border-radius: 5px;
                overflow-x: auto;
                white-space: pre-wrap;
            }}
            code {{
                font-family: Consolas, monospace;
                color: inherit;
                display: block;
            }}
        </style>
    </head>
    <body>
        {render(data)}
    </body>
    </html>
    """

def print_tool_calls(file_path, count_call=None):
    """Prints only the tool calls in a JSON saved by the runner script.
    """
    
    with open(f'{file_path}', 'r') as f:
        data = json.load(f)
    messages = data.get('messages', [])

    if count_call is not None:
        counter=0
        
    for message in messages:
        tools = message.get('tool_calls', [])

        for tool in tools:
            tool_function = tool.get('function', {})
            if tool_function:
                function_name = tool_function.get('name', 'unknown')
                function_args = tool_function.get('arguments', {})
                if count_call is not None:
                    if function_name==count_call:
                        counter+=1
                else:
                    print(f"{function_name}")

    if count_call is not None:
        return counter

def new_extract_tool_calls(file_path, tool_names):
    """Extract the tool calls in a JSON saved by the runner script.
    """

    calls=[]
    
    with open(f'{file_path}', 'r') as f:
        data = json.load(f)
    messages = data.get('messages', [])

    for message in messages:
        tools = message.get('tool_calls', [])

        for tool in tools:
            tool_function = tool.get('function', {})
            if tool_function:
                function_name = tool_function.get('name', 'unknown')
                function_args = tool_function.get('arguments', {})
                if function_name in tool_names:
                    calls.append(tool)

    return calls
        
def print_messages(file_path, roles_to_exclude=None, show_function_args=True, print_metadata=True,
                   response_api=True):
    """Prints messages from a JSON saved by the runner.py script.
    Args:
        file_path (str): Path to the JSON file.
        roles_to_exclude (list, optional): List of roles to exclude from printing. Defaults to None.
        show_function_args (bool, optional): Whether to show function arguments in tool calls. Defaults to True.
        print_metadata (bool, optional): Whether to print metadata. Defaults to True."""
    
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

    if roles_to_exclude is None:
        roles_to_exclude = []

    with open(f'{file_path}', 'r') as f:
        data = json.load(f)
    messages = data.get('messages', [])
    metadata_keys = ['task', 'simulator', 'simulator_init', 'simulator_tools', 'analyzer_tools',
                     'system_prompt', 'base_url', 'model_name']
    runner_args = data.get('runner_args', {})
    if print_metadata:
        for key in metadata_keys:
            key_data = runner_args.get(key, None)
            if key_data is not None:
                display(HTML(f"<div style='white-space: pre-wrap'><b>{key.upper()}:</b> {escape(str(key_data))}</div>"))
        print(data.get('result', None))




    for message in messages:
        role = message.get('role', 'unknown')
        if role == 'unknown':
            role = message.get('type', 'unknown')
        content = message.get('content', [])
        tools = message.get('tool_calls', [])
        if not isinstance(content, list):
             content = [content, ]
        for c in content: 
            if (isinstance(c, dict) and c['type'] == user_text_key and role not in roles_to_exclude):
                display(HTML(f"<div style='white-space: pre-wrap'><b>{role.upper()}:</b> {c['text']}</div>"))
            elif isinstance(c, str) and role not in roles_to_exclude:
                if not c.strip() == '':
                    display(HTML(f"<div style='white-space: pre-wrap'><b>{role.upper()}:</b> {c}</div>"))
            elif (isinstance(c, dict) and c['type'] == image_url_key):
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
                # Plot the image
                dpi = 100  # screen DPI for display
                figsize = (image.width / dpi, image.height / dpi)
                plt.figure(figsize=figsize, dpi=dpi)
                plt.imshow(image)
                plt.axis('off')  # Hide axes
                plt.show()
        for tool in tools:
            tool_function = tool.get('function', {})
            # if not tool_function:
            #     tool_function = tool
            function_name = tool_function.get('name', 'unknown')
            function_args = tool_function.get('arguments', {})
            display(HTML(f"<div style='white-space: pre-wrap'><b>ASSISTANT TOOL CALL:</b> {function_name}</div>"))
            if show_function_args:
                display(HTML(render_json_to_html(function_args)))
        if role == 'function_call_output':
            display(HTML(f"<div style='white-space: pre-wrap'><b>{role.upper()}:</b> {message.get('output', None)}</div>"))

        #print(f"{role}: {content}")


def print_summary(file_path, print_metadata=True, experiment_tools=['run_experiment','run_field_evolution_experiment',
                                                                   'run_wave_experiment','observe_evolution']):
    """Prints the final (summary) message from a JSON saved by the runner.py script,
    together with the relevant metadata describing the experiment. This should
    be ready to include, for example, in a supplementary file for a publication.
    
    Args:
        file_path (str): Path to the JSON file.    
    """

    with open(f'{file_path}', 'r') as f:
        data = json.load(f)
        messages = data.get('messages', [])

        # print file
        split=file_path.rsplit('/',2)
        if len(split)>=2:
            filename=split[-2]+"/"+split[-1]
        else:
            filename=split[-1]

        print(f"Run: {filename}")
        runtime=data.get("runtime")
        if runtime is not None:
            print(f"runtime: {int(runtime)} secs")
        timestamp=data.get("timestamp")
        if timestamp is not None:
            print(f"timestamp: {timestamp}")
            
        # print important metadata
        metadata_keys = ['task', 'simulators_file', 'simulator', 'simulator_init', 'simulator_tools', 'analyzer_file', 'analyzer_tools',
                         'system_prompt', 'intermediate_prompt', 'model_name']
        runner_args = data.get('runner_args', {})
        if print_metadata:
            for key in metadata_keys:
                key_data = runner_args.get(key, None)
                if key_data is not None:
                    print(f"{key}: {key_data}")

        # count tool calls:
        tool_calls=0
        experiment_calls=0
        # and images:
        num_images=0
        
        for message in messages:
            content = message.get('content', [])
            if not isinstance(content, list):
                 content = [content, ]
            for c in content: 
                if (isinstance(c, dict) and c['type'] == 'image_url'):
                    num_images+=1
                
            tools = message.get('tool_calls', [])
            for tool in tools:
                tool_function = tool.get('function', {})
                if tool_function:
                    tool_calls+=1
                    function_name = tool_function.get('name', 'unknown')
                    if function_name.rsplit('.',1)[-1] in experiment_tools: # rsplit will turn Wave.run_experiment into run_experiment
                        experiment_calls+=1
        print(f"tool calls (total): {tool_calls}\nexperimental tool calls: {experiment_calls}\nimages: {num_images}")

        print("")
        print("="*80)
        
        # print summary message, which is assumed to be the last:
        print(messages[-1].get('content',""))
