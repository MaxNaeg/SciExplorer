import sys
import os
import importlib
import hashlib
import importlib.util
import time
import datetime
import json
import glob

import numpy as np
import jax

from .model_utils import Model
from .parsers import parse_markdown_to_dict
from .run_exp import run_experiment

from .tool_utils import get_method

def sha256sum(filename):
    h  = hashlib.sha256()
    b  = bytearray(128*1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        while n := f.readinto(mv):
            h.update(mv[:n])
    return h.hexdigest()

def get_dict_hash(dictionary):
    # Convert dictionary to a JSON string
    # made by Claude
    # Using sort_keys=True ensures consistent ordering of keys
    json_string = json.dumps(dictionary, sort_keys=True)
    
    # Create a hash using SHA-256 (you can choose other algorithms too)
    hash_object = hashlib.sha256(json_string.encode())
    
    # Get the hexadecimal digest of the hash
    hash_value = hash_object.hexdigest()
    
    return hash_value
    
def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    path = filename + "_" + str(counter) + extension

    while os.path.exists(path):
        path = filename + "_" + str(counter) + extension
        counter += 1

    return path

def import_from_path(module_path, class_name=None):
    """
    Import a module from a relative or absolute path.
    Supports both "subdir/my_module" and "/abs/path/to/my_module.py".
    
    Optionally extracts a specific class or function from it.
    """
    if os.path.isabs(module_path) or module_path.endswith('.py'):
        # Absolute or direct path to .py file
        if not module_path.endswith('.py'):
            module_path += '.py'

        if not os.path.exists(module_path):
            raise ImportError(f"Module file not found: {module_path}")

        module_name = f"dynamic_import_{os.path.splitext(os.path.basename(module_path))[0]}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None:
            raise ImportError(f"Failed to create module spec from {module_path}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    else:
        # Relative import using dot notation
        module_name = module_path.replace('/', '.').replace('\\', '.')
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            # Try to resolve relative file path from the current script location
            parts = module_path.replace('\\', '/').split('/')
            module_filename = parts[-1] + ".py"
            module_dir = os.path.join(*parts[:-1]) if len(parts) > 1 else "."
            base_dir = os.path.dirname(os.path.abspath(__file__))
            full_path = os.path.join(base_dir, module_dir, module_filename)

            if not os.path.exists(full_path):
                raise ImportError(f"Module file not found: {full_path}")

            module_name = f"dynamic_import_{parts[-1]}"
            spec = importlib.util.spec_from_file_location(module_name, full_path)
            if spec is None:
                raise ImportError(f"Failed to create module spec from {full_path}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

    return getattr(module, class_name) if class_name else module



def replace_arrays_with_lists(obj):
    if isinstance(obj, dict):
        return {key: replace_arrays_with_lists(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(replace_arrays_with_lists(item) for item in obj)
    elif isinstance(obj, (np.ndarray, jax.Array)):
        return obj.tolist()
    else:
        return obj
    



def run(base_dir: str, tasks_file: str, task: str, simulators_file: str, 
        simulator: str, simulator_init: dict, system_prompts_file: str, 
        system_prompt: str, save_to_file: str, api_key, base_url, model_name,
        intermediate_prompts_file: str = '', 
        intermediate_prompt: str = '',
        use_tools: bool = True,
        analyzer_file: str = "", analyzer_tools: str = "",
        simulator_tools: str = "",
        random_exp_func: str = None, 
        plot_tool:str = None, 
        plot_tool_file:str = "",
        n_exps_to_run:int = 0,
        run_kwargs=dict(
                final_response_key = "",
                allow_multiple_tool_calls = False,
                keep_only_one_image = False,
                max_tries_for_correct_tool_call = 2,
                max_steps = 20, max_tools = 8),
        human_input=False,
        summarize_at_end=False,
        replace_message_function=None,
        separate_tool_calls=True,
        response_api = True, # if True, use the response API instead of chat completions
        reasoning_level = None,
        verbosity = None,
        reasoning_summary=None,
        timeout=None,
):
    """
    Run a single agentic experiment.

    This is based on a combination of
        an analyzer
        a simulator
        a task
        a system prompt
        an intermediate prompt

    The analyzer does not need to know which simulator it
    will be used for. It may offer one or several toolboxes,
    i.e. collections of its methods.

    The simulator does not need to know which analyzer it
    will be subjected to. It may offer one or several toolboxes.
    It will be initialized according to a given set of
    parameters, which modifies the simulator scenario.
    It must contain one or several result tool methods that
    can be called by the LLM to announce the result and which
    must make sense for the chosen task.
    
    The task does not need to know which analyzer is used,
    and the coupling to the simulator is weak: the task must
    make sense for the given simulator, and the task specifies
    a particular result_tool, which is a method that must
    be present inside the simulator and which will be called
    by the LLM to announce the final result and which checks
    whether (or how well) this result matches the expected
    result for the given task and the given simulator scenario.

    The system prompt is generic and tries to guide the
    LLM towards intelligent analysis.
    base_dir: str, tasks_file: str, task: str, simulators_file: str, 
        simulator: str, simulator_init: dict, system_prompts_file: str, 
        intermediate_prompts_file: str, intermediate_prompt: str,
        system_prompt: str, save_to_file: str, api_key, base_url, model_name,
        use_tools: bool = False,
        analyzer_file: str = "", analyzer_tools: str = "",
        simulator_tools: str = "",
        random_exp_func: str = None, 
        plot_tool:str = None, 
        n_exps_to_run:int = 0,
        run_kwargs=dict(
                final_response_key = "",
                allow_multiple_tool_calls = False,
                keep_only_one_image = False,
                max_tries_for_correct_tool_call = 2,
                max_steps = 20, max_tools = 8),
        human_input=False
    Args:
        base_dir: will be prepended to all filenames and modules (default "", otherwise dir without trailing "/")
        tasks_file: name of .md file containing tasks (including .md)
        task: name of task (a # headline inside the .md file)
        simulators_file: name of module containing simulator (without .py)
        simulator: name of simulator class (inside that .py)
        simulator_init (dict): parameters passed to init method of simulator
        system_prompts_file: name of .md file containing system prompts (including .md)
        intermediate_prompts_file: name of .md file containing intermediate prompts (including .md)
        intermediate_prompt: which intermediate prompt to use (# headline)
        system_prompt: which system prompt to use (# headline)
        save_to_file: file to save input and results as json (None to omit, adds counter if file exists)
        api_key: api key for the LLM
        base_url: base url for the LLM
        model_name: model_name for the LLM
        use_tools (bool): if True, use tools from analyzer and simulator, otherwise only reason
        analyzer_file: name of module containing analyzer (without .py)
        analyzer_tools: name of analysis toolbox
        simulator_tools: name of simulator toolbox to use
        random_exp_func: name of method in simulator to call for random experiments provided at beginning if n_exps_to_run > 0
        plot_tool: name of method in analyzer to call for plotting results of initial experiments (if None, no plotting)
        n_exps_to_run (int): number of random experiments to run before starting the actual task (0 to skip)
        run_kwargs (dict, optional): extra parameters (e.g. max_steps)
        human_input (boolean, optional): if True, get human input for intermediate prompts
        summarize_at_end: Ask the LLM to summarize the conversation at the end (default False).
        replace_message_function: (function) if set, this function will be called to replace messages of the assistant in the conversation..
            It should have the signature replace_message_function(last_message:str, time_step:int, tool_calls:list,) -> tuple(str, bool)
            and return the modified message of the assistant and a boolean indicating whether the LLM should keep reasoning after the replaced message.
        separate_tool_calls: (bool) if True, the LLM will be asked to first only reason and then call only a tool.
        response_api: (bool) if True, use the response API instead of chat completions
        reasoning_level: (str) amount of thingking in reasoning process. 'low', 'medium', 'high' or None (for default reasoning level).
        verbosity: (str) amount of verbosity in answer. 'low', 'medium', 'high' or None (for default verbosity level).
        reasoning_summary: (str) if set, the LLM will be asked to summarize its reasoning at each step.
        timeout: (int) if set, each tool call will be limited to this many seconds (default None, uses TIMEOUT variable)
    Returns:
        result: (dict) final result announced by LLM to simulator
        messages: (list) all messages
        model: the LLM model object
        simulator_object: the simulator object
        _fields: (dict) all the internal memory stored during conversation for analysis
        
    """

    if n_exps_to_run > 0: 
        assert random_exp_func is not None, "n_exps_to_run > 0 requires random_exp_func to be set"

    if use_tools:
        assert analyzer_file != "", "use_tools=True requires analyzer_file to be set"
        assert analyzer_tools != "", "use_tools=True requires analyzer_tools to be set"
        assert simulator_tools != "", "use_tools=True requires simulator_tools to be set"

    start_time = time.time()
    
    if base_dir != "":
        base_dir = base_dir + "/"
    
    SimulatorClass = import_from_path(base_dir + simulators_file, class_name = simulator)
    simulator_object = SimulatorClass(**simulator_init)

    if analyzer_file and analyzer_tools:
        analyzer = import_from_path(base_dir + analyzer_file)
        analyzer_tool_list = analyzer.toolboxes[analyzer_tools]
    else:
        analyzer_tool_list = []



    
    model=Model(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name
    )

    with open(base_dir + tasks_file) as file:
        tasks_text=file.read()
    tasks_dict = parse_markdown_to_dict(tasks_text)
    
    with open(base_dir + system_prompts_file) as file:
        system_prompts_text=file.read()
    system_prompts_dict = parse_markdown_to_dict(system_prompts_text)

    system_prompt_text = system_prompts_dict[system_prompt]['content']
    task_prompt = tasks_dict[task]['children']['description']['content']
    result_tool = tasks_dict[task]['children'].get('result_tool', {}).get('content', None)    

    if intermediate_prompt and intermediate_prompts_file:
        with open(base_dir + intermediate_prompts_file) as file:
            intermediate_prompts_text=file.read()
        intermediate_prompts_dict_all = parse_markdown_to_dict(intermediate_prompts_text)
        intermediate_prompt_dict = intermediate_prompts_dict_all[intermediate_prompt]['children']
    else:
        intermediate_prompt_dict = None

    if plot_tool:
        if not plot_tool_file:
            plot_tool_file = analyzer_file
        plot_tool_module = import_from_path(base_dir + plot_tool_file)
        plot_function = get_method(plot_tool_module, plot_tool)
    else:
        plot_function = None

    if random_exp_func:        
        random_exp_func = get_method(simulator_object, random_exp_func)
    else:
        random_exp_func = None

    result, messages, past_history, _fields = run_experiment(
                model, task_prompt, simulator_object, 
                simulator_tools = simulator_tools,
                analysis_tools = analyzer_tool_list,
                run_kwargs=run_kwargs, 
                result_tool = result_tool,
                system_prompt=system_prompt_text,
                intermediate_prompts=intermediate_prompt_dict,
                random_exp_func=random_exp_func,
                plot_function=plot_function,
                n_exps_to_run=n_exps_to_run,
                human_input=human_input,
                use_tools=use_tools,
                summarize_at_end=summarize_at_end,
                replace_message_function=replace_message_function,
                separate_tool_calls=separate_tool_calls,
                response_api=response_api,
                reasoning_level=reasoning_level,
                verbosity=verbosity,
                reasoning_summary=reasoning_summary,
                timeout=timeout
                )

    runtime=time.time() - start_time
    print(f"-- runtime: {runtime} seconds --")
    
    try:
        if save_to_file is not None:
            everything=dict(
                runner_args=dict(
                base_dir=base_dir,
                tasks_file=tasks_file, task=task, 
                simulators_file=simulators_file, simulator=simulator, 
                simulator_init=replace_arrays_with_lists(simulator_init), 
                simulator_tools=simulator_tools, 
                analyzer_file=analyzer_file, analyzer_tools=analyzer_tools, 
                system_prompts_file=system_prompts_file, system_prompt=system_prompt,
                intermediate_prompts_file=intermediate_prompts_file, intermediate_prompt=intermediate_prompt,
                save_to_file=save_to_file,
                run_kwargs=run_kwargs, base_url=base_url, model_name=model_name
                ),
                hashes=dict(
                    simulators_file=sha256sum(base_dir + simulators_file + ".py"),
                    analyzer_file=sha256sum(base_dir + analyzer_file + ".py"),
                    system_prompts_file=sha256sum(base_dir + system_prompts_file),
                    intermediate_prompts_file=sha256sum(base_dir + intermediate_prompts_file),
                    tasks_file=sha256sum(base_dir + tasks_file)
                )
            )

            hash_input=get_dict_hash(everything)

            everything['messages']=messages
            everything['past_history']=past_history
            everything['result']=replace_arrays_with_lists(result)
            everything['runtime']=runtime
            everything['timestamp']=str(datetime.datetime.now())
            everything['input_hash']=hash_input

            filename=uniquify(save_to_file + "_" + hash_input )
            with open(filename, "w") as file:
                json.dump(everything, file)
            print(f"Saved to file: {filename}")
            
    except Exception as e:
        print(f"Problem writing to json -- {e}")
        pass
        
    return result, messages, model, simulator_object, _fields


   


def summarize_runs(base_dir: str, save_to_file: str):
    if base_dir != "":
        base_dir += "/"
        
    file_list = sorted(glob.glob(base_dir + save_to_file + "_*"))

    # extract argument names from first file
    with open(file_list[0], "r") as file:
        contents=json.load(file)
        args=contents['runner_args']
        for arg in args:
            print(arg, end=",")
        print()
        
    for file_name in file_list:
        with open(file_list[0], "r") as file:
            contents=json.load(file)
            args=contents['runner_args']
            for arg in args:
                print(args[arg], end=",")
            print()
