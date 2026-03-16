from openai import OpenAI
from sciexplorer.runs.linear_exploration import run_exploration
from sciexplorer.utils.print_utils import whole_conversation_to_string
from sciexplorer.utils.model_specifics import get_format_image_function
from typing import Callable
from faiss import IndexFlatL2
import numpy as np


def get_critic_tool(model, tools=None,
                    system_message:str="You are a helpful critic.",
                    first_message_str:str="Please evaluate the following interaction and provide constructive feedback.",
                    critic_tool_doc:str=None, # Documentation string for the critic tool
                    invoke_kwargs:dict=None, # Additional kwargs to pass when calling the model
                    save_critique:Callable=None, # Function to save the final critique
                    max_steps:int=0, # Maximum number of steps for the critic to take before saving the critique
                    max_tool_uses:int=None, # Maximum number of tool uses for the critic
                    print_intermediate:bool=True #print intermediate steps
                    ) -> Callable:

    """Creates a critic tool that can evaluate a previous conversation and provide feedback."""

    if save_critique is None:
        def save_critique(critique:str, performance:int) -> str:
            """Function to save your final critique. Call this function to end your analysis, potentially after using other tools first.
            You can only call this function once as your final action.
            Args:
                critique (str): Your final critique of the previous conversation.
                performance (int): Your assessment of the AI assistant's performance. Rate from 1 (poor) to 10 (excellent).
            """
            return {"critique": critique, "performance": performance}

    def critic(_messages)->dict:
        """Call this tool to get feedback on your progress so far.
        Use it after major steps in the conversation to get an outside perspective."""
        prev_conv_str, prev_conv_images = whole_conversation_to_string([mes.model_dump() for mes in _messages])
        first_message = [{'type': 'text', 'text': f"""{first_message_str}
            <previous_conversation>
            {prev_conv_str}
            </previous_conversation>
            """},]
        if print_intermediate:
            print("<<STARTING CRITIC CONVERSATION>>")
        # add previous images to first message
        format_image = get_format_image_function()
        first_message += [format_image(base64_img) for base64_img in prev_conv_images]

        final_state = run_exploration(
            model = model, # The language model to use
            initial_message = first_message, # The initial message to the agent
            result_tool = save_critique, # The tool to save the final result
            system_message = system_message, # The system prompt to guide the agent's behavior
            # List of messages to send to the agent in between iterations, will be cycled through if more iterations than messages
            intermediate_message_list=['Go on researching for your critique or save your result if you cannot improve it anymore.'],
            max_iterations=max_steps, # Maximum number of iterations to run
            max_tool_uses=max_tool_uses, # Maximum number of tool uses
            tools=tools, # tools to be used by the agent
            initial_fields=None, # Initial fields available eg. in the execute_code tool
            add_tools_to_fields = False, # Whether to add all tools as callable fields to the _fields dictionary, e.g. to call them from execute_code
            save_path=None, # Path to save the results
            print_intermediate=print_intermediate, # Whether to print intermediate messages
            invoke_kwargs=invoke_kwargs, # kwargs passed when calling the model
        )

        if print_intermediate:
            print("<<END CRITIC CONVERSATION>>")
            
        return final_state['result']
    
    if critic_tool_doc is not None:
        critic.__doc__ = critic_tool_doc

    return critic

def get_vec_db_query_tool(embeddings:np.ndarray,
                          index_to_text:dict,
                          client: OpenAI,
                          docstring:str=None,
                          max_k:int=10,
                          model_name:str="text-embedding-3-small") -> Callable:
    
    # create the FAISS index
    embedding_dimension = embeddings.shape[1]
    index = IndexFlatL2(embedding_dimension)
    index.add(embeddings)

    def query_db(query:str, top_k:int=max_k) -> list:
        f"""Query the vector database for similar entries to the given query string.
        Args:
            query (str): The query string to search for.
            top_k (int): The number of top similar entries to retrieve. Must be less than or equal to {max_k}.
        Returns a dict of the form:
            relevant_chunks: A list of the top_k most similar text chunks from the database.
        """
        assert top_k <= max_k, f"top_k must be less than or equal to max_k ({max_k})"

        # embed the query string into a vector
        query_embedding = client.embeddings.create(
            model=model_name,
            input=query
        ).data[0].embedding
        query_embedding = np.array([query_embedding], dtype="float32")
        # search the index for similar vectors

        distances, indices = index.search(query_embedding, top_k)
        # retrieve the corresponding text chunks
        text_chunks = [index_to_text[idx] for idx in indices[0]]
        return {'relevant_chunks': text_chunks}
    
    if docstring is not None:
        query_db.__doc__ = docstring

    return query_db