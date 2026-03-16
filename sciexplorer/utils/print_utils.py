               


import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from sciexplorer.utils.model_specifics import get_return_parsed_messages

def print_message(message_dict:dict) -> None:
    """Print a message dictionary in a human-readable format.
    Make sure to set MODEL_TYPE in model_specifics.py before using this function.
    Args:
        message_dict: dictionary representing a message from OpenAI API, created by message.model_dump()
    """
    return_parsed_message = get_return_parsed_messages()
    messages, images = return_parsed_message(message_dict)
    for role, content_type, content in messages:
        # if function arg indent:
        if content_type == "FUNCTION ARGUMENT":
            has_break = '\n' in content
            arg_name, arg_value = content.split(":", 1)
            print(f"  ARGUMENT {arg_name}:{'\n    ' if has_break else ' '}{arg_value.replace('\n', '\n    ')}")
        else:
            print(f"{role} {content_type}: {content}")

def whole_conversation_to_string(messages_list: list[dict], to_exclude:list[tuple[str,str]]=None) -> tuple[str, list[str]]:
    """Convert a list of message dictionaries into a single string representation.
    Make sure to set MODEL_TYPE in model_specifics.py before using this function.
    Args:
        messages_list: list of dictionaries representing messages, created by message.model_dump()
        to_exclude: list of tuples (role, content_type) to exclude from the output.
    Returns:
        A tuple containing:
        - A string representing the whole conversation.
        - A list of base64 strings representing the images in the conversation.
    """
    return_parsed_message = get_return_parsed_messages()
    if to_exclude is None:
        to_exclude = []
    conversation_str = ""
    images = []
    image_counter = 0
    for message_dict in messages_list:
        parsed_messages, new_images = return_parsed_message(message_dict, image_counter)
        images.extend(new_images)
        image_counter += len(new_images)
        for role, content_type, content in parsed_messages:
            if (role, content_type) in to_exclude:
                continue
            if content_type == "FUNCTION ARGUMENT":
                has_break = '\n' in content
                arg_name, arg_value = content.split(":", 1)
                conversation_str += f"  ARGUMENT {arg_name}:{'\n    ' if has_break else ' '}{arg_value.replace('\n', '\n    ')}\n"
            elif content_type == "IMAGE":
                conversation_str += f"{role} {content_type}: [{content}, provided as separate image data]\n"
            else:
                conversation_str += f"{role} {content_type}: {content}\n"
        conversation_str += "\n"
    return conversation_str, images

def print_whole_conversation(messages_list: list[dict], to_exclude:list[tuple[str,str]]=None, show_images:bool=True) -> None:
    '''Print a list of message dictionaries in a human-readable format.
    Make sure to set MODEL_TYPE in model_specifics.py before using this function.
    Args:
        messages_list: list of dictionaries representing messages, created by message.model_dump()
        to_exclude: list of tuples (role, content_type) to exclude from the output.
        show_images: whether to display images found in the messages.
    '''
    return_parsed_message = get_return_parsed_messages()
    if to_exclude is None:
        to_exclude = []
    image_counter = 0
    for message_dict in messages_list:
        parsed_messages, new_images = return_parsed_message(message_dict, image_counter)
        image_counter += len(new_images)
        for role, content_type, content in parsed_messages:
            if (role, content_type) in to_exclude:
                continue
            if content_type == "FUNCTION ARGUMENT":
                has_break = '\n' in content
                arg_name, arg_value = content.split(":", 1)
                print(f"  ARGUMENT {arg_name}:{'\n    ' if has_break else ' '}{arg_value.replace('\n', '\n    ')}")
            else:
                print(f"{role} {content_type}: {content}")
        if show_images:
            for base64_str in new_images:
                image = Image.open(BytesIO(base64.b64decode(base64_str)))
                dpi = 100  # screen DPI for display
                figsize = (image.width / dpi, image.height / dpi)
                plt.figure(figsize=figsize, dpi=dpi)
                plt.imshow(image)
                plt.axis('off')  # Hide axes
                plt.show()
        print()


def print_result(result, show_images:bool=True, image_key:str="image") -> None:
    """Print the result of a SciExplorer run in a human-readable format.
    Args:
        result: dictionary or string saved by linear_exploration function. Saved as 'result' in all.json files.
        show_images: whether to display images found in the result (only works if result is a dictionary).
        image_key: must be contained in key of result dictionary if and only if the value is a base64 encoded image.
    """
    if isinstance(result, str):
        print(result)
    elif isinstance(result, dict):
        image_idx = 0
        base_64_images = []
        for key, value in result.items():
            if image_key in key and isinstance(value, str):
                if not show_images:
                    print(f"{key.upper()}: [image data {image_idx}]")
                    image_idx += 1
                else:
                    base_64_images.append(value)
            else:
                print(f"{key.upper()}:\n{value}")
        if show_images:
            for value in base_64_images:
                image = Image.open(BytesIO(base64.b64decode(value)))
                dpi = 100  # screen DPI for display
                figsize = (image.width / dpi, image.height / dpi)
                plt.figure(figsize=figsize, dpi=dpi)
                plt.imshow(image)
                plt.axis('off')  # Hide axes
                plt.show()
    else:
        print("Result is neither a string nor a dictionary.")
    
