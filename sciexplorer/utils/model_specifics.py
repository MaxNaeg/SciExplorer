import json
MODEL_TYPE = "openai"  # or "gemini"



GEMINI_MEDIA_RESOLUTION = 'media_resolution_high'
#format_images for different APIs-------------------------------------------------------------------------------------------
def format_image_for_openai(base64_str: str, mime:str="png"):
    """Return a dictionary representing an image for OpenAI API."""
    return {
        "type": "image", "source_type": "url", 
        "url": f"data:image/{mime};base64,{base64_str}"}


def format_image_for_gemini(base64_str, mime:str="png"):
    # return {
    #     "inline_data": {
    #         "mime_type": mime,
    #         "data": base64_str
    #     }
    # }
    return {"inline_data": {
        "mime_type": mime,
        "data": base64_str,
        },
        "media_resolution": {
        "level": GEMINI_MEDIA_RESOLUTION}
    }
# print message for different APIs-------------------------------------------------------------------------------------------
def return_parsed_message_for_openai(message_dict:dict, image_counter:int=0) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Parse a message dictionary from OpenAI API into a list of message strings and image strings.
    Args:
        message_dict: dictionary representing a message from OpenAI API, created by message.model_dump()
        image_counter: counter for images to number them sequentially.
    Returns:
        A tuple containing:
        - A list of tuples (role, content_type, content) representing the parsed message.
        - A list of base64 strings representing the images in the message.
    """
    role = message_dict.get("type", "unknown").upper()
    message_strings = []
    image_strings = []
    content = message_dict.get("content", "")
    if content is not None:
        if isinstance(content, str):
            message_strings.append((role, "TEXT", content))
        elif isinstance(content, list):
            for content_part in content:
                assert isinstance(content_part, dict), "Expected content parts to be dictionaries."
                content_type = content_part.get("type", "unknown")
                if content_type == "text":
                    message_strings.append((role, "TEXT", content_part.get("text", "")))
                elif content_type == "reasoning":
                    summary = content_part.get('summary', [])
                    if isinstance(summary, str):
                        message_strings.append((role, "REASONING", summary))
                    elif isinstance(summary, list):
                        for item in summary:
                            if isinstance(item, dict):
                                message_strings.append((role, "REASONING", item.get('text', '')))
                            else:
                               raise ValueError("Expected summary items to be dictionaries.")
                    else:
                        raise ValueError("Expected summary to be either a string or a list.")
                elif content_type == 'function_call':
                    #descr = f"{content_part.get('name', '')} with args:"
                    message_strings.append((role, "FUNCTION CALL", content_part.get('name', '')))
                    args = json.loads(content_part.get('arguments', "{}"))
                    for arg_name, arg_value in args.items():
                        #descr += f"\n  {arg_name}: {arg_value}"
                        message_strings.append((role, "FUNCTION ARGUMENT", f"{arg_name}:{arg_value}"))
                    #message_strings.append((role, "FUNCTION CALL", descr))
                elif content_type == 'image':
                    message_strings.append((role, "IMAGE", f"<<Image {image_counter}>>"))
                    image_strings.append(content_part.get("url", "").split(',')[1])  # Extract base64 part
                    image_counter += 1
                elif content_type == 'web_search_call':
                    pass # Skip web search calls for now
                else:
                    raise ValueError(f"Unknown content type of content part: {content_part}")
        else:
            raise ValueError("Expected content to be either a string or a list.")
    else:
        message_strings.append((role, "NO CONTENT", ""))
    return message_strings, image_strings



def return_parsed_message_for_gemini(message_dict:dict, image_counter:int=0) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Parse a message dictionary from OpenAI API into a list of message strings and image strings.
    Args:
        message_dict: dictionary representing a message from OpenAI API, created by message.model_dump()
        image_counter: counter for images to number them sequentially.
    Returns:
        A tuple containing:
        - A list of tuples (role, content_type, content) representing the parsed message.
        - A list of base64 strings representing the images in the message.
    """
    role = message_dict.get("type", "unknown").upper()
    message_strings = []
    image_strings = []
    content = message_dict.get("content", "")
    if content is not None:
        if isinstance(content, str):
            message_strings.append((role, "TEXT", content))
        elif isinstance(content, list):
            for content_part in content:
                #assert isinstance(content_part, dict), "Expected content parts to be dictionaries."
                if isinstance(content_part, str):   
                    message_strings.append((role, "TEXT", content_part))
                    continue     
                elif not isinstance(content_part, dict):
                    message_strings.append((role, "UNKNOWN", str(content_part)))
                    continue
                content_type = content_part.get("type", "unknown")
                if content_type == "text":
                    message_strings.append((role, "TEXT", content_part.get("text", "")))
                    #message_strings.append((role, "FUNCTION CALL", descr))
                # elif content_type == 'image':
                #     message_strings.append((role, "IMAGE", f"<<Image {image_counter}>>"))
                #     image_strings.append(content_part.get("url", "").split(',')[1])  # Extract base64 part
                #     image_counter += 1
                elif content_type == 'thinking':
                    thinking = content_part.get('thinking', '')
                    if isinstance(thinking, str):
                        message_strings.append((role, "REASONING", thinking))
                    else:
                        raise ValueError("Expected thinking to be a string.")
                else:
                    if (len(content_part) == 2 and 
                        "inline_data" in content_part and 
                        content_part["inline_data"].get("mime_type", None) == 'png'):
                        message_strings.append((role, "IMAGE", f"<<Image {image_counter}>>"))
                        image_strings.append(content_part["inline_data"].get("data", ""))
                        image_counter += 1
                    else:
                        raise ValueError(f"Unknown content type: {content_type}")
        else:
            raise ValueError("Expected content to be either a string or a list.")
    else:
        message_strings.append((role, "NO CONTENT", ""))

    tool_calls = message_dict.get("tool_calls", [])
    for call in tool_calls:
        if call.get('type', '') == 'tool_call':
            message_strings.append((role, "FUNCTION CALL", call.get('name', '')))
            args = call.get('args', "{}")
            for arg_name, arg_value in args.items():
                message_strings.append((role, "FUNCTION ARGUMENT", f"{arg_name}:{arg_value}"))
        else:
            raise ValueError(f"Unknown tool call type: {call.get('type', None)}")

    return message_strings, image_strings



# print message for different APIs-------------------------------------------------------------------------------------------
def return_parsed_message_for_chat_openai(message_dict:dict, image_counter:int=0) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Parse a message dictionary from OpenAI API into a list of message strings and image strings.
    Args:
        message_dict: dictionary representing a message from OpenAI API, created by message.model_dump()
        image_counter: counter for images to number them sequentially.
    Returns:
        A tuple containing:
        - A list of tuples (role, content_type, content) representing the parsed message.
        - A list of base64 strings representing the images in the message.
    """
    role = message_dict.get("type", "unknown").upper()
    message_strings = []
    image_strings = []
    content = message_dict.get("content", "")
    if content is not None:
        if isinstance(content, str):
            message_strings.append((role, "TEXT", content))
        elif isinstance(content, list):
            for content_part in content:
                assert isinstance(content_part, dict), "Expected content parts to be dictionaries."
                content_type = content_part.get("type", "unknown")
                if content_type == "text":
                    message_strings.append((role, "TEXT", content_part.get("text", "")))
                # elif content_type == "reasoning":
                #     summary = content_part.get('summary', [])
                #     if isinstance(summary, str):
                #         message_strings.append((role, "REASONING", summary))
                #     elif isinstance(summary, list):
                #         for item in summary:
                #             if isinstance(item, dict):
                #                 message_strings.append((role, "REASONING", item.get('text', '')))
                #             else:
                #                raise ValueError("Expected summary items to be dictionaries.")
                #     else:
                #         raise ValueError("Expected summary to be either a string or a list.")
                # elif content_type == 'function_call':
                #     #descr = f"{content_part.get('name', '')} with args:"
                #     message_strings.append((role, "FUNCTION CALL", content_part.get('name', '')))
                #     args = json.loads(content_part.get('arguments', "{}"))
                #     for arg_name, arg_value in args.items():
                #         #descr += f"\n  {arg_name}: {arg_value}"
                #         message_strings.append((role, "FUNCTION ARGUMENT", f"{arg_name}:{arg_value}"))
                #     #message_strings.append((role, "FUNCTION CALL", descr))
                elif content_type == 'image':
                    message_strings.append((role, "IMAGE", f"<<Image {image_counter}>>"))
                    image_strings.append(content_part.get("url", "").split(',')[1])  # Extract base64 part
                    image_counter += 1
                # elif content_type == 'web_search_call':
                #     pass # Skip web search calls for now
                else:
                    raise ValueError(f"Unknown content type of content part: {content_part}")
        else:
            raise ValueError("Expected content to be either a string or a list.")
    else:
        message_strings.append((role, "NO CONTENT", ""))
        
    tool_calls = message_dict.get("tool_calls", [])
    for call in tool_calls:
        if call.get('type', '') == 'tool_call':
            message_strings.append((role, "FUNCTION CALL", call.get('name', '')))
            args = call.get('args', "{}")
            for arg_name, arg_value in args.items():
                message_strings.append((role, "FUNCTION ARGUMENT", f"{arg_name}:{arg_value}"))
        else:
            raise ValueError(f"Unknown tool call type: {call.get('type', None)}")

    return message_strings, image_strings


# Dictionaries to get functions based on MODEL_TYPE---------------------------------------------------------------------------
format_image_results_dict = {
    "openai": format_image_for_openai,
    "gemini": format_image_for_gemini,
    'chat_openai': format_image_for_openai,
}

return_parsed_messages_dict = {
    "openai": return_parsed_message_for_openai,
    "gemini": return_parsed_message_for_gemini,
    'chat_openai': return_parsed_message_for_chat_openai,
}

def get_format_image_function():
    return format_image_results_dict.get(MODEL_TYPE)
def get_return_parsed_messages():
    return return_parsed_messages_dict.get(MODEL_TYPE)