

import time

from langchain_core.language_models import BaseChatModel
from langchain.tools import tool as tool_decorator
from langchain.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from langchain_google_genai import ChatGoogleGenerativeAI


from typing_extensions import TypedDict
from typing import Literal, Any, Callable

from sciexplorer.utils.tool_parsing_utils import replace_argument, function_to_result_label_tool
from sciexplorer.utils.run_utils import add_text_to_message, invoke_with_backoff
from sciexplorer.utils.model_specifics import get_format_image_function
from sciexplorer.utils.print_utils import print_message
from sciexplorer.runs.linear_exploration import tool_node, save_exploration


MAX_RESULT_TRIES = 3





class GraphState(TypedDict):
    model: BaseChatModel
    steps: int # llm_steps so far
    tool_calls: int # tool_calls so far
    format_image_for_model: Callable # function to format images for the current model
    _fields: dict # stores all tool results
    tools_by_name: dict # mapping of tool names to tool instances
    tool_wrapper: Callable | None # wrapper to apply to each tool call, e.g. to add a timeout
    messages: list[AnyMessage] # stores all intermediate messages
    print_long_arrays: bool # if true, prints full arrays in tool results instead of summaries
    final_runtime: float | None # total runtime of the exploration
    print_intermediate: bool # if true, prints all intermediate messages
    invoke_kwargs: dict
    result: Any | None
    ask_at_each_step: bool # if true, asks the user for input at each step. Otherwise call only if no tool calls were made.
    max_retries: int # maximum retries for LLM calls with backoff
    backoff_factor: int # backoff factor for LLM calls with backoff
    initial_delay: int # initial delay for LLM calls with backoff

def llm_call_node(state: dict) -> Command[Literal['tool_node', 'llm_call_node', END]]:
    steps = state.get('steps', 0)
    messages = state.get('messages', [])
    model = state.get('model', None)
    ask_at_each_step = state.get('ask_at_each_step', True)
    
    steps += 1
    if not isinstance(messages[-1], ToolMessage) or ask_at_each_step:
        human_message = input('Your message (type "END" to finish conversation): ')

        if len(messages) > 0 and isinstance(messages[-1], HumanMessage):
            messages[-1] = HumanMessage(add_text_to_message(messages[-1].content, human_message))
        else:
            messages.append(HumanMessage(content=human_message))
        if state.get('print_intermediate', False):
                print_message(messages[-1].model_dump())
        
        if human_message.strip().upper() == 'END':
            return Command(update={'messages': messages, 'steps': steps},
                goto=END)
    
    model_response = invoke_with_backoff(model, messages, invoke_kwargs=state.get('invoke_kwargs', {}), 
                                         max_retries=state.get('max_retries', 0), 
                                         backoff_factor=state.get('backoff_factor', 2), 
                                         initial_delay=state.get('initial_delay', 10))
    messages.append(model_response)
    if state.get('print_intermediate', False):
        print_message(messages[-1].model_dump())

    if model_response.tool_calls:
        return Command(update={'messages': messages, 'steps': steps},
        goto='tool_node')
    else:
        return Command(update={'messages': messages, 'steps': steps},
            goto='llm_call_node')


def run_human_in_the_loop(
        model: BaseChatModel,
        system_message: str = None,
        initial_message: str = None,
        tools: list[callable] = None,
        initial_fields: dict = None,
        add_tools_to_fields: bool = False,
        save_path : str = None,
        tool_wrapper: callable = None,
        print_intermediate: bool = True,
        print_long_arrays: bool = False,
        invoke_kwargs: dict = None,
        add_to_saved_dict: dict = None,
        add_to_save_name: str = None,
        ask_at_each_step: bool = True,
        inbuilt_tools: list = None,
        max_retries: int = 0, 
        backoff_factor: int = 2, 
        initial_delay: int = 10, 
) -> GraphState:
    """Runs an exploration loop using the provided model and tools. To end the exploration, the user can type 'END' when prompted for input.
    Args:
        model: The language model to use for exploration.
        system_message: An optional system message to set the context for the model.
        initial_message: An optional initial message to start the conversation.
        tools:  List of tools available for the model to use. Tools must return dicts.
                If a tool uses the argument '_field', this will be automatically be filled with a dict containing the tool results collected so far.
                For all tool calls, the LLM is additionally asked to provide a 'result_label' under which the tool result will be stored in _fields.
        initial_fields: Initial fields to provide to the tools under the '_field' argument.
        add_tools_to_fields: If True, adds to tool functions to the _fields dict.
        save_path: Optional path to save the exploration.
        tool_wrapper: Wrapper to apply to each tool call, e.g. to add a timeout.
        print_intermediate: If True print conversation.
        print_long_arrays: If True, prints full arrays in tool results instead of summaries.
        invoke_kwargs: kwargs passed to model.invoke function
        add_to_saved_dict: additional dict to add to the saved dictionary.
        add_to_save_name: string to add to the save file name.
        ask_at_each_step: If True, asks the user for input at each step. Otherwise call only if no tool calls were made.
        inbuilt_tools: list of inbuilt tools to add to the tools list. E.g. [{ type: "web_search" }, ], for OpenAI web search tool.
        max_retries: int, maximum retries for LLM calls with backoff, if 0 only one attempt is made
        backoff_factor: int, backoff factor for LLM calls with backoff
        initial_delay: int, initial delay for LLM calls with backoff in seconds
    Returns:
        GraphState: The final state of the exploration. Holding all messages, created fields, the result, and runtime.
    """
    _fields = {} if initial_fields is None else initial_fields
    tools = [] if tools is None else tools
    invoke_kwargs = {} if invoke_kwargs is None else invoke_kwargs

    adapted_tools = [function_to_result_label_tool(replace_argument(tool, '_fields', _fields)) for tool in tools]
    if add_tools_to_fields:
        for tool in adapted_tools:
            tool_name = tool.__name__
            _fields[tool_name] = tool
    adapted_tools = [tool_decorator(tool) for tool in adapted_tools]

    tools_by_name = {tool.name: tool for tool in adapted_tools}
    print(f'--binding tools {[tool.name for tool in adapted_tools]}--')
    model = model.bind_tools(adapted_tools + (inbuilt_tools if inbuilt_tools is not None else []))

    if isinstance(model.bound, ChatOpenAI):
        format_image_for_model = get_format_image_function()
    elif isinstance(model.bound, ChatGoogleGenerativeAI):
        format_image_for_model = get_format_image_function()
    else:
        raise NotImplementedError(f"Models of type {type(model.bound)} are currently not supported.")

    initial_messages = []
    if system_message:
        initial_messages.append(SystemMessage(content=system_message))
        if print_intermediate:
            print_message(initial_messages[-1].model_dump())
    if initial_message:
        initial_messages.append(HumanMessage(content=initial_message))

    initial_state = GraphState(model=model,
        steps=0,
        tool_calls=0,
        format_image_for_model=format_image_for_model,
        _fields=_fields,
        tools_by_name=tools_by_name,
        tool_wrapper=tool_wrapper,
        messages=initial_messages,
        print_long_arrays=print_long_arrays,
        print_intermediate=print_intermediate,
        final_runtime=None,
        invoke_kwargs=invoke_kwargs,
        result=None,
        ask_at_each_step=ask_at_each_step,
        max_retries=max_retries,
        backoff_factor=backoff_factor,
        initial_delay=initial_delay,
    )
    
    agent_builder = StateGraph(GraphState)
    agent_builder.add_node('llm_call_node', llm_call_node)
    agent_builder.add_node('tool_node', tool_node)
    agent_builder.add_edge(START, 'llm_call_node')

    agent = agent_builder.compile()

    # uncomment to display the graph
    #display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

    start_time = time.time()
    final_state = agent.invoke(initial_state, {"recursion_limit": int(1e3)})
    end_time = time.time()
    final_state['final_runtime'] = end_time - start_time
    print(f'--Exploration finished in {final_state["final_runtime"]:.2f} seconds.--')
    print(f'--Final result: {final_state["result"]}--')

    if save_path:
        save_exploration(final_state, save_path, add_to_save_name=add_to_save_name,
                        initial_message=initial_message,
                        system_message=system_message,
                        max_iterations=None,
                        max_tool_uses=None,
                        tools_by_name=tools_by_name,
                        print_long_arrays=print_long_arrays,
                        ask_for_final_message_str=None,
                        intermediate_message_list=None,
                        add_to_saved_dict=add_to_saved_dict,
                        add_tools_to_fields=add_tools_to_fields,
            )
    return final_state










    
    