
from .model_utils import Model
from .tool_utils import get_method




def run_experiment(model:Model, task_prompt:str, simulator, simulator_tools:list,
                   analysis_tools:list, run_kwargs:dict, 
                   random_exp_func, # (should be method of simulator)
                   plot_function, # should be method in analyzer
                   n_exps_to_run,
                   result_tool = None,
                   system_prompt:str = None,  intermediate_prompts:dict = None,
                   include_return_in_tool_parsing=False,
                   human_input=False,
                   use_tools=True,
                   summarize_at_end=False,
                   replace_message_function=None,
                   separate_tool_calls=True,
                   response_api=False,
                   reasoning_level=None,
                   verbosity=None,
                   reasoning_summary=None,
                   timeout=None, # if set, the LLM will be asked to summarize its reasoning at each step.
            
):
    """model: the model to use
    task_prompt: the prompt to use for the task
    simulator: the (already initialized) simulator object to use
    simulator_tools: the tools to use for the simulator, either a list of tools as strings or a string descibing a toolbox
    analysis_tools: a list of analysis tools
    run_kwargs: the kwargs to use for the interative model calls
    random_exp_func: the function to use for generating random experiments, only needed if n_exps_to_run >0.
    result_tool: the tool to use for retrieving the result, if None, no result will be retrieved
    system_prompt (str): Optional system prompt to set the context.
    intermediate_prompts (dict): Optional dictionary of intermediate prompts to use for the experiment.
    include_return_in_tool_parsing (bool)
    human_input (bool): If True, allow the actual user to supply the intermediate prompts.
    use_tools (bool): If True, use the tools specified in simulator_tools and analysis_tools.
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
    """

    question = simulator.description + "\n" + task_prompt
    if not use_tools:
        all_tools = []
    else:
        if isinstance(simulator_tools, str):
            simulator_tools = simulator.toolboxes[simulator_tools]
        elif isinstance(simulator_tools, list):
            simulator_tools = [get_method(simulator, tool) for tool in simulator_tools]


        # TODO: more gracefull handling
        print(f'--Setting simulator tools: {simulator_tools}--')
        print(f'--Setting analysis tools: {analysis_tools}--')
        all_tools = simulator_tools + analysis_tools
    if result_tool is not None:
        try:
            all_tools.append(get_method(simulator, result_tool))
        except Exception as e:
            result_tool = None
            print(f"--Error adding result tool: {e}--")

    model.set_tools(all_tools, include_return=include_return_in_tool_parsing, response_api=response_api,)
    print("--Running experiment--")
    messages, past_history,  _fields = model.query(
            question=question,
            system_prompt=system_prompt,
            intermediate_prompts=intermediate_prompts,
            result_tool=result_tool,
            run_kwargs=run_kwargs,
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
            timeout=timeout,
    )
    if result_tool is not None:
        print(f"--Checking wether result was set with tool {result_tool}--")
        try:
            result = simulator.last_result
            print(f"--Saved result: {result}--")
        except Exception as e:
            print(f"--Retrieving result failed with error: {e}\n Messages where: {messages}--")
    else:
        print("--No result tool specified, skipping result check--")
        result = None
    print("--Experiment finished--")
    return result, messages, past_history, _fields
