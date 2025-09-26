# default
## ask_for_reasoning
If already present, what do the past tool results teach you about the question?
Then describe which tool you want to call next and why you want to call it in plain text.
Don't yet call the tool, just describe what you want to do.
If you don't need to call anymore tools, you can instead provide the final answer.
While reasoning, remember your system prompt.

## ask_for_tool_use
Now call the tools.
Don't include any text but the tool request in the format defined in your system prompt.


## wrong_format_tool_call
Badly formatted tool call, please try again and remember the proper format defined in your system prompt.
Especially remember to use required brackets.

## ask_final_answer
Please provide a final answer without using any tools.


# very_short
## ask_for_reasoning
Follow your system prompt now.
Do not yet call any tools!
## ask_for_tool_use
Now call the tools. Don't include any text but the tool request in the format defined in your system prompt.
## wrong_format_tool_call
Badly formatted tool call, please try again and remember the proper format defined in your system prompt.
## ask_final_answer
Please provide a final answer without using any tools.


# no_separate_tools
## ask_for_reasoning
Keep solving the problem while following your system prompt.
## ask_for_tool_use
THIS SHOULD NEVER BE CALLED
## wrong_format_tool_call
THIS SHOULD NEVER BE CALLED
## ask_final_answer
Please provide a final answer without using any tools.


# no_separate_tools_ask_answer
## ask_for_reasoning
Keep solving the problem while following your system prompt. Make sure to answer the questions posed in the system prompt in your response.
## ask_for_tool_use
THIS SHOULD NEVER BE CALLED
## wrong_format_tool_call
THIS SHOULD NEVER BE CALLED
## ask_final_answer
Please provide a final answer without using any tools.