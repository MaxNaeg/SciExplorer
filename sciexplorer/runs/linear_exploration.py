
import os
import json
import cloudpickle as pickle
import time
from datetime import datetime

from langchain_core.language_models import BaseChatModel
from langchain.tools import tool as tool_decorator
from langchain.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from langchain_google_genai import ChatGoogleGenerativeAI


from typing_extensions import TypedDict
from typing import Literal, Any, Callable

from sciexplorer.utils.tool_parsing_utils import replace_argument, function_to_result_label_tool
from sciexplorer.tools.tool_utils import ImageData
from sciexplorer.utils.run_utils import replace_arrays_with_lists, get_description, add_text_to_message, invoke_with_backoff
from sciexplorer.utils.model_specifics import get_format_image_function
from sciexplorer.utils.print_utils import print_message



MAX_RESULT_TRIES = 5



class GraphState(TypedDict):
    model: BaseChatModel
    steps: int # llm_steps so far
    max_steps: int # maximum llm steps allowed
    tool_calls: int # tool_calls so far
    format_image_for_model: Callable # function to format images for the current model
    max_tool_uses: int | None # maximum tool uses allowed, if None unlimited
    _fields: dict # stores all tool results
    result_tool_name: str # name of the result tool, required.
    tools_by_name: dict # mapping of tool names to tool instances
    tool_wrapper: Callable | None # wrapper to apply to each tool call, e.g. to add a timeout
    intermediate_message_list: list[str] | None # list of intermediate messages to cycle through
    messages: list[AnyMessage] # stores all intermediate messages
    result: Any | None # final result after exploration is done
    print_long_arrays: bool # if true, prints full arrays in tool results instead of summaries
    final_runtime: float | None # total runtime of the exploration
    print_intermediate: bool # if true, prints all intermediate messages
    ask_for_final_message_str: str | None # if set, asks the model for a final message after exploration is done
    invoke_kwargs: dict # kwargs passed to model.invoke function
    max_retries: int # maximum retries for LLM calls with backoff
    backoff_factor: int  # backoff factor for LLM calls with backoff
    initial_delay: int # initial delay for LLM calls with backoff
    intermediate_human_messages: bool # whether to insert human messages at each step to inform the model about remaining steps and tools. The content of the human message can be controlled via the intermediate_message_list argument.




def llm_call_node(state: dict) -> Command[Literal['tool_node', 'result_tool_node', 'llm_call_node']]:
    steps = state.get('steps', 0)
    tool_calls = state.get('tool_calls', 0)
    max_tool_uses = state.get('max_tool_uses', None)
    max_steps = state.get('max_steps', None)
    messages = state.get('messages', [])
    model = state.get('model', None)
    intermediate_human_messages = state.get('intermediate_human_messages', True)
    result_tool_name = state['result_tool_name']
    intermediate_message_list = state.get('intermediate_message_list', [""])
    
    steps += 1

    if intermediate_human_messages and max_steps is not None and steps > max_steps:
        messages.append(HumanMessage(f'Maximum steps reached. Call the tool {result_tool_name} now.'))
    elif steps == 1 or intermediate_human_messages:
        human_message = ''
        if max_steps is not None:
            human_message += f'You have {max_steps - steps + 1} steps remaining and you can still call {"up to " + str(max_tool_uses - tool_calls) if max_tool_uses is not None else "unlimited"} tools.\n'
        human_message += intermediate_message_list[(steps - 1) % len(intermediate_message_list)]
        if human_message.strip():
            if isinstance(messages[-1], HumanMessage):
                messages[-1] = HumanMessage(add_text_to_message(messages[-1].content, human_message))
            else:
                messages.append(HumanMessage(content=human_message))
        
    if isinstance(messages[-1], HumanMessage) and state.get('print_intermediate', False):
        print_message(messages[-1].model_dump())
    
    model_response = invoke_with_backoff(model, messages, invoke_kwargs=state.get('invoke_kwargs', {}), 
                                         max_retries=state.get('max_retries', 0), 
                                         backoff_factor=state.get('backoff_factor', 2), 
                                         initial_delay=state.get('initial_delay', 10))
    messages.append(model_response)
    if state.get('print_intermediate', False):
        print_message(messages[-1].model_dump())

    if max_steps is not None and steps > max_steps:
        return Command(update={'messages': messages, 'steps': steps},
            goto='result_tool_node')
    elif model_response.tool_calls:
        if result_tool_name in [tool_call["name"] for tool_call in model_response.tool_calls]:
            return Command(update={'messages': messages, 'steps': steps},
            goto='result_tool_node')
        else:
            return Command(update={'messages': messages, 'steps': steps},
            goto='tool_node')
    else:
        return Command(update={'messages': messages, 'steps': steps},
            goto='llm_call_node')
    
def tool_node(state: dict) -> Command[Literal['llm_call_node']]:
    # run all tools add results if result tool was called set to true
    tools_by_name = state['tools_by_name']
    messages = state.get('messages', [])
    tool_wrapper = state.get('tool_wrapper', None)
    print_long_arrays = state.get('print_long_arrays', False)
    tool_calls = state.get('tool_calls', 0)
    # high number to allow unlimited tool uses if no max set
    max_tool_uses = state.get('max_tool_uses', int(1e9))
    _fields = state.get('_fields', {})
    intermediate_human_messages = state.get('intermediate_human_messages', True)
    all_images = []
    for tool_call in messages[-1].tool_calls:
        images = []
        if max_tool_uses is None or tool_calls < max_tool_uses:
            tool_calls += 1
            tool = tools_by_name.get(tool_call["name"], None)
            if tool is None:
                observation = f"Tool {tool_call['name']} not found."
                messages.append(ToolMessage(content=[{"type": "text", "text": observation}], tool_call_id=tool_call["id"]))
                continue
            try:
                if tool_wrapper:
                    tool_result = tool_wrapper(tool.invoke, tool_call["args"])
                else:
                    tool_result = tool.invoke(tool_call["args"])
                assert isinstance(tool_result, dict), "Tool wrapped by tool_node must return a dictionary."

                observation = "Saved the following result under <result_key>: <value>"
                for key, value in tool_result.items():
                    description = get_description(value, print_all=print_long_arrays)
                    observation += f" \n{key}: {description}"
                    if isinstance(value, ImageData):
                        images.append(state['format_image_for_model'](value.image))
                _fields.update(tool_result)
                all_images += images
            except Exception as e:
                observation = f"{e.__class__.__name__}: {e}"
        else:
            observation = f"Maximum tool uses {max_tool_uses} reached. Cannot call tool {tool_call['name']}."
        if intermediate_human_messages:
            messages.append(ToolMessage(content=[{"type": "text", "text":observation},] + images,
                                        tool_call_id=tool_call["id"]))
        else:
            messages.append(ToolMessage(content=[{"type": "text", "text":observation},],
                                        tool_call_id=tool_call["id"]))
        if state.get('print_intermediate', False):
            print_message(messages[-1].model_dump())
    if not intermediate_human_messages and all_images:
        messages.append(AIMessage(content="Please show me the images from the last tool calls."))
        if state.get('print_intermediate', False):
            print_message(messages[-1].model_dump())
        messages.append(HumanMessage(content=[{"type": "text", "text": "Here are the images."}] + all_images))
        
    return Command(update={'messages': messages, '_fields': _fields, 'tool_calls': tool_calls, 'tools_by_name': tools_by_name},
            goto='llm_call_node')
        
    
def result_tool_node(state: dict) -> Command[Literal['final_message_node']]:
    messages = state.get('messages', [])
    result_tool_name = state['result_tool_name']
    result = None
    model_response = messages[-1]
    successful = False
    for i in range(MAX_RESULT_TRIES):
        if model_response.tool_calls:
            for tool_call in model_response.tool_calls:
                if tool_call["name"] == result_tool_name:
                    try:
                        tool = state['tools_by_name'][tool_call["name"]]
                        result = tool.invoke(tool_call["args"]) # for now no wrapper on result tool
                        messages.append(ToolMessage(content=[{"type": "text", "text": 'Result saved successfully'}], tool_call_id=tool_call["id"]))
                        successful = True
                    except Exception as e:
                        error_message= f"{e.__class__.__name__}: {e}"
                        messages.append(ToolMessage(content=[{"type": "text", "text": error_message}], tool_call_id=tool_call["id"]))
                else:
                    messages.append(ToolMessage(content=[{"type": "text", "text": f'Only the tool {result_tool_name} can be called now'}], tool_call_id=tool_call["id"]))
                if state.get('print_intermediate', False):
                    print_message(messages[-1].model_dump())
        if successful or i == MAX_RESULT_TRIES-1: # dont't call model again in last iteration
            break
        if state.get('intermediate_human_messages', True):
            messages.append(HumanMessage(f'Calling tool {result_tool_name} failed. Please try again. You can only call the tool {result_tool_name} now.'))
        if state.get('print_intermediate', False):
            print_message(messages[-1].model_dump())
        model_response = state['model'].invoke(messages, **state.get('invoke_kwargs', {}))
        messages.append(model_response)
        if state.get('print_intermediate', False):
            print_message(messages[-1].model_dump())
    
    return Command(update={'messages': messages, 'result': result},
            goto='final_message_node')





def final_message_node(state: dict) -> Command[Literal[END]]:
    ask_for_final_message_str = state.get('ask_for_final_message_str', None)
    intermediate_human_messages =  state.get('intermediate_human_messages', True)
    messages = state.get('messages', [])
    if ask_for_final_message_str:
        model = state['model']
        if not intermediate_human_messages:
            messages.append(AIMessage(content='Complete.'))
        messages.append(HumanMessage(content=ask_for_final_message_str))
        if state.get('print_intermediate', False):
            print_message(messages[-1].model_dump())

        model_response = invoke_with_backoff(model, messages, invoke_kwargs=state.get('invoke_kwargs', {}), 
                                         max_retries=state.get('max_retries', 0), 
                                         backoff_factor=state.get('backoff_factor', 2), 
                                         initial_delay=state.get('initial_delay', 10))
        messages.append(model_response)
        if state.get('print_intermediate', False):
            print_message(messages[-1].model_dump())

        if model_response.tool_calls:
            for tool_call in model_response.tool_calls:
                messages.append(ToolMessage(content=[{"type": "text", "text": 'No tools can be called at this stage.'}], tool_call_id=tool_call["id"]))
                if state.get('print_intermediate', False):
                    print_message(messages[-1].model_dump())

    return Command(update={'messages': messages},
            goto=END)


def save_exploration(final_state:GraphState, save_path: str, add_to_save_name: str = "",
                     initial_message: str = None,
                     system_message: str = None,
                     max_iterations: int = None,
                     max_tool_uses: int = None,
                     tools_by_name: dict = None,
                     print_long_arrays: bool = False,
                     ask_for_final_message_str:str = None,
                     intermediate_message_list: list[str] = None,
                     add_to_saved_dict: dict = None,
                     add_tools_to_fields: bool = None,):
    '''Saves the exploration result to the specified path.
    Pickles the fields dict separately, and saves all other information in a json file.'''

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(save_path, exist_ok=True)
    # with open(os.path.join(save_path, 'final_state.pkl'), 'wb') as f:
    #     pickle.dump(final_state, f)
    try:
        with open(os.path.join(save_path, f'{add_to_save_name}all_{timestamp}.json'), 'w') as f:
            all_dict = {'messages': [message.model_dump() for message in final_state.get('messages', [])],
                'final_runtime': final_state.get('final_runtime', None),
                'result': replace_arrays_with_lists(final_state.get('result', None)),
                'initial_message': initial_message,
                'system_message': system_message,
                'max_iterations': max_iterations,
                'max_tool_uses': max_tool_uses,
                'print_long_arrays': print_long_arrays,
                'ask_for_final_message_str': ask_for_final_message_str,
                'steps': final_state.get('steps', None),
                'tool_calls': final_state.get('tool_calls', None),
                'tools': list(tools_by_name.keys()),
                'intermediate_message_list': intermediate_message_list,
                'add_tools_to_fields': add_tools_to_fields,
                }
            if add_to_saved_dict:
                all_dict.update(add_to_saved_dict)
            json.dump(all_dict, f, indent=4)
        print(f'Saved results to {os.path.join(save_path, f"{add_to_save_name}all_{timestamp}.json")}')
    except Exception as e:
        print(f"Error saving all.json: {e}")
    try:
        with open(os.path.join(save_path, f'{add_to_save_name}fields_{timestamp}.pkl'), 'wb') as f:
            pickleable_fields = {}
            for key, value in final_state['_fields'].items():
                try:
                    pickle.dumps(value)
                    pickleable_fields[key] = value
                except Exception as e:
                    print(f"Could not pickle field {key}: {e}")
            pickle.dump(pickleable_fields, f)
        print(f'Saved fields to {os.path.join(save_path, f"{add_to_save_name}fields_{timestamp}.pkl")}')
    except Exception as e:
        print(f"Error saving fields.pkl: {e}")

def run_exploration(
        model: BaseChatModel,
        initial_message: str,
        result_tool: callable = None,
        system_message: str = None,
        intermediate_message_list: list[str] = None,
        max_iterations: int = 5,
        max_tool_uses: int = None,
        tools: list[callable] = None,
        initial_fields: dict = None,
        add_tools_to_fields: bool = False,
        save_path : str = None,
        tool_wrapper: callable = None,
        print_intermediate: bool = True,
        ask_for_final_message_str:str = None,
        print_long_arrays: bool = False,
        invoke_kwargs: dict = None,
        add_to_saved_dict: dict = None,
        add_to_save_name: str = "",
        inbuilt_tools: list = None,
        max_retries: int = 0, 
        backoff_factor: int = 2, 
        initial_delay: int = 10,
        intermediate_human_messages: bool = True,
) -> GraphState:
    """Runs an exploration loop using the provided model and tools.
    Args:
        model: The language model to use for exploration.
        initial_message: The initial message to start the exploration.
        result_tool: A tool to process and return the final result. If None, asks for final result as string.
        system_message: An optional system message to set the context for the model.
        intermediate_message_list: An optional list of messages to provide intermediate feedback after each step. The messages are cycled through.
        max_iterations: Maximum number of iterations to run.
        max_tool_uses: Maximum number of tool uses allowed.
        tools:  List of tools available for the model to use. Tools must return dicts.
                If a tool uses the argument '_field', this will be automatically be filled with a dict containing the tool results collected so far.
                For all tool calls, the LLM is additionally asked to provide a 'result_label' under which the tool result will be stored in _fields.
        initial_fields: Initial fields to provide to the tools under the '_field' argument.
        add_tools_to_fields: If True, adds to tool functions to the _fields dict.
        save_path: Optional path to save the exploration.
        tool_wrapper: Wrapper to apply to each tool call, e.g. to add a timeout.
        print_intermediate: If True print conversation.
        ask_for_final_message_str: Message passed to LLM after exploration is complete (e.g. to ask for a summary).
        print_long_arrays: If True, prints full arrays in tool results instead of summaries.
        invoke_kwargs: kwargs passed to model.invoke function
        add_to_saved_dict: additional dict to add to the saved dictionary.
        add_to_save_name: string to add to the save file name.
        inbuilt_tools: list of inbuilt tools to add to the tools list. E.g. [{'type': 'web_search'}, ], for OpenAI web search tool.
        max_retries: int, maximum retries for LLM calls with backoff, if 0 only one attempt is made
        backoff_factor: int, backoff factor for LLM calls with backoff
        initial_delay: int, initial delay for LLM calls with backoff in seconds
        intermediate_human_messages: bool, Insert human messages at each step to inform the model about remaining steps and tools. The content of the human message can be controlled via the intermediate_message_list argument.
                                    If False, this will not interleave human messages and also communicate images in tool results as HumanMessages. This is needed e.g. for Mistral models.
    Returns:
        GraphState: The final state of the exploration. Holding all messages, created fields, the result, and runtime.
    """
    _fields = {} if initial_fields is None else initial_fields
    tools = [] if tools is None else tools
    invoke_kwargs = {} if invoke_kwargs is None else invoke_kwargs
    initial_messages = []


    if add_tools_to_fields:
        add_to_first_message = '\nIf you directly call a tool, the result will be stored (using the result_label) and available later. ' \
        'However, if you call a tool indirectly inside another tool, the result will not be stored automatically and you will not be able to access it later. '
    else:
        add_to_first_message = '\nThe results of all tool calls will be stored (using the result_label) and are available later. '
    
    initial_message = add_text_to_message(initial_message, add_to_first_message)

    if not result_tool:
        def result_tool(result: str) -> dict:
            """Please provide your final answer to the users query. 
            Only call this at the end of your exploration when you cannot improve your answer any further."""
            return result
        
    adapted_tools = [function_to_result_label_tool(replace_argument(replace_argument(tool, '_fields', _fields), '_messages', initial_messages)) for tool in tools]
    if add_tools_to_fields:
        for tool in adapted_tools:
            tool_name = tool.__name__
            _fields[tool_name] = tool
    adapted_tools = [tool_decorator(tool) for tool in adapted_tools]
    adapted_tools += [tool_decorator(replace_argument(result_tool, '_fields', _fields))]


    tools_by_name = {tool.name: tool for tool in adapted_tools}
    print(f'--binding tools {[tool.name for tool in adapted_tools]}--')
    model = model.bind_tools(adapted_tools + (inbuilt_tools if inbuilt_tools is not None else []))

    if isinstance(model.bound, ChatOpenAI):
        format_image_for_model = get_format_image_function()
    elif isinstance(model.bound, ChatGoogleGenerativeAI):
        format_image_for_model = get_format_image_function()
    else:
        raise NotImplementedError(f"Models of type {type(model.bound)} are currently not supported in linear_exploration.py")

    if system_message:
        initial_messages.append(SystemMessage(content=system_message))
        if print_intermediate:
            print_message(initial_messages[-1].model_dump())
    initial_messages.append(HumanMessage(content=initial_message))

    initial_state = GraphState(model=model,
        steps=0,
        max_steps=max_iterations,
        tool_calls=0,
        max_tool_uses=max_tool_uses,
        format_image_for_model=format_image_for_model,
        _fields=_fields,
        result_tool_name=result_tool.__name__,
        tools_by_name=tools_by_name,
        tool_wrapper=tool_wrapper,
        intermediate_message_list=intermediate_message_list,
        messages=initial_messages,
        result=None,
        print_long_arrays=print_long_arrays,
        ask_for_final_message_str=ask_for_final_message_str,
        print_intermediate=print_intermediate,
        final_runtime=None,
        invoke_kwargs=invoke_kwargs,
        max_retries=max_retries,
        backoff_factor=backoff_factor,
        initial_delay=initial_delay,
        intermediate_human_messages=intermediate_human_messages,
    )
    
    agent_builder = StateGraph(GraphState)
    agent_builder.add_node('llm_call_node', llm_call_node)
    agent_builder.add_node('tool_node', tool_node)
    agent_builder.add_node('result_tool_node', result_tool_node)
    agent_builder.add_node('final_message_node', final_message_node)
    agent_builder.add_edge(START, 'llm_call_node')

    agent = agent_builder.compile()

    # uncomment to display the graph
    #display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

    start_time = time.time()
    final_state = agent.invoke(initial_state, {"recursion_limit": int(1e6)})
    end_time = time.time()
    final_state['final_runtime'] = end_time - start_time
    print(f'--Exploration finished in {final_state["final_runtime"]:.2f} seconds.--')
    print(f'--Final result: {final_state["result"]}--')

    if save_path:
        save_exploration(final_state, save_path, add_to_save_name=add_to_save_name,
                        initial_message=initial_message,
                        system_message=system_message,
                        max_iterations=max_iterations,
                        max_tool_uses=max_tool_uses,
                        tools_by_name=tools_by_name,
                        print_long_arrays=print_long_arrays,
                        ask_for_final_message_str=ask_for_final_message_str,
                        intermediate_message_list=intermediate_message_list,
                        add_to_saved_dict=add_to_saved_dict,
                        add_tools_to_fields=add_tools_to_fields,
            )
    return final_state










    
    