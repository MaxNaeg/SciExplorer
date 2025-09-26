import jax
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import jax.numpy as jnp
import optax
import flax
from sciexplorer.tool_utils import ImageData, get_image
from sciexplorer import model_utils
from sciexplorer.physics_eom import ode_solve


# General analysis tools-------------------------------------

def execute_code(_fields:dict, code: str) -> dict:
    """ Evaluate python code.
        This code can be e.g. be used to transform the previously saved fields or to calculate or save new fields.
        The following variables are available during evaluation:
            all previosuly saved fields with their previously stated result_labels as variable names (as local variables).
            jax: jax for numerical operations.
            jnp: jax.numpy for numerical operations.
            np: numpy for numerical operations.
            scipy: scipy for numerical operations including optimization and solving differential equations.
            optax: optax for efficient gradient based optimizers like adam, rmsprop, etc.
        IMPORTANT: Your code must set the variable 'result' to a dictionary in the end which should contain the newly generated data, for example: result={'<result_key>': <data>, ...}.
        Args:
            code: python code that sets the result variable to a dictionary containing some newly generated data.
        Returns:
            the result dictionary.
        """
    globals_dict = _fields | {'result':None, 'jax':jax, 'jnp': jnp, 'np':np, 'scipy':scipy, 
                              'optax':optax,}
    exec(code, globals_dict)
    if globals_dict['result'] is None:
        raise ValueError("The code did not set the result variable. It must be set to a dictionary containing the newly generated data. E.g. result={'<result_key>': <data>, ...}")
    if not isinstance(globals_dict['result'], dict):
        raise ValueError("The result variable must be a dictionary. E.g. result={'<result_key>': <data>, ...}")
    return globals_dict['result']


def plot_from_code(_fields:dict, code: str) -> dict:
    """
    Execute python code that produces a plot from one or more previously saved arrays.
    The following variables are available during evaluation:
        all previosuly saved fields with their previously stated result_labels as variable names.
        get_image: function to get the image of the plot.
    You may import the following libraries:
            matplotlib: for plotting.
            matplotlib.pyplot as plt: for plotting.
            jax: for numerical operations.
            jax.numpy: for numerical operations.
            numpy: for numerical operations.
    The end of the code must say "result=get_image()" such that the image of the plot can be returned for visual analysis.

    Args:
        code: python code that produces a plot (without calling plt.show).
    Returns:
        Image
    """
    locals_dict= {'result':None}
    globals_dict = _fields | {'get_image': get_image} | globals()
    exec(code, globals_dict, locals_dict)
    plt.show()
    result = locals_dict['result']
    if not isinstance(result, ImageData):
        raise ValueError(f"The code did not set the result variable to an image, but instead to {result}.")
    return {'image': result}

def get_all_previously_saved_results(_fields:dict) -> dict:
    """
    Returns a description of all previously saved results.
    
    Returns:
        description: A string describing all previously saved results.
    """
    descriptions = []
    for field_label, field in _fields.items():
        description = model_utils.get_description(field)
        descriptions.append(f"{field_label}: {description}")
    
    return {"description": "\n".join(descriptions)}

def only_plot(_fields:dict, use_legend:str, x_label_list:str, y_label_list:str, x_idcs_list:str, y_idcs_list:str) -> dict:
    """
    Plot two previously saved arrays against each other.
    Args:
        use_legend: whether to use a legend in the plot, either "True" or "False"
        x_label_list: list of result_labels of the x arrays to plot, in the form "['label1', 'label2', ...]"
        y_label_list: list of result_labels of the y arrays to plot, in the form "['label1', 'label2', ...]"
        x_idcs_list: list of indices of the x arrays to plot (as a string), in the form "['[idx1_label1, :, ...]', '[idx1_label2, :, ...]', ...]", where idx1 is an integer index and : means all indices.
        y_idcs_list: list of indices of the y arrays to plot (as a string), in the form "['[idx1_label1, :, ...]', '[idx1_label2, :, ...]', ...]", where idx1 is an integer index and : means all indices.
    For example, to plot the first row of two-dimensional array 'array1' against the one-dimensional array 'time1', 
    and the second row of two-dimensional array 'array2', against the one-dimensional array 'time2' ,you would call:
        plot_only("['time1', 'time2']", "['array1', 'array2']", "['[:]', '[:]']", "['[0, :]', '[1, :]']")
    This would plot array1[0,:] vs time.
    Returns:
        Image
    """
    if use_legend == "True":
        use_legend = True
    elif use_legend == "False":
        use_legend = False
    else:
        raise ValueError("use_legend must be either 'True' or 'False'.")
    plt.figure(figsize=(8,6))
    try:
        x_label_list = eval(x_label_list)
    except Exception as e:
        raise ValueError(f"Error while evaluating x_label_list: {e}")
    try:
        y_label_list = eval(y_label_list)
    except Exception as e:
        raise ValueError(f"Error while evaluating y_label_list: {e}")
    try:
        x_idcs_list = eval(x_idcs_list)
    except Exception as e:
        raise ValueError(f"Error while evaluating x_idcs_list: {e}")
    try:
        y_idcs_list = eval(y_idcs_list)
    except Exception as e:
        raise ValueError(f"Error while evaluating y_idcs_list: {e}")
    assert isinstance(x_label_list, list), "x_label_list must be a list of strings."
    assert isinstance(y_label_list, list), "y_label_list must be a list of strings."
    if len(x_label_list) != len(y_label_list):
        raise ValueError("x_label_list and y_label_list must have the same length.")
    for x_label, y_label, x_idcs, y_idcs in zip(x_label_list, y_label_list, x_idcs_list, y_idcs_list):
        x_data = _fields.get(x_label, None)
        y_data = _fields.get(y_label, None)
        if x_data is None:
            raise ValueError(f"No field with label {x_label} found.")
        if y_data is None:
            raise ValueError(f"No field with label {y_label} found.")
        try:
            x_to_plot = eval(f"x_data{x_idcs}")
            y_to_plot = eval(f"y_data{y_idcs}")
        except Exception as e:
            raise ValueError(f"Error while indexing the data with {x_idcs} and {y_idcs}: {e}")
        try:
            plt.plot(x_to_plot, y_to_plot, linestyle='-', label=f"{y_label}{y_idcs} vs {x_label}{x_idcs}")  
        except Exception as e:
            raise ValueError(f"Error while plotting {y_label} vs {x_label}: {e}")
    plt.xlabel("X")
    plt.ylabel("Y")
    if use_legend:
        plt.legend()
    plt.grid()
    result=get_image()
    return {'image': result}

def approx_equal(a1:jax.Array, a2:jax.Array) -> dict:
    """
    Check whether two arrays can be considered approximately the same. 
    This calculates their mean-square error and compares it to the mean square variation. 
    
    Args:
        a1: first array
        a2: second array
    Returns:
        statement: a string indicating the closeness of the two arrays
        ratio: (mean-square error) / max(mean square variation)
    """
    if isinstance(a1, np.ndarray):
        a1 = jnp.array(a1)
    if isinstance(a2, np.ndarray):
        a2 = jnp.array(a2)
    if not isinstance(a1, jax.Array):
        raise ValueError(f"First argument must be a jax.numpy array, but was {type(a1)}.")
    if not isinstance(a2, jax.Array):
        raise ValueError(f"Second argument must be a jax.numpy array, but was {type(a2)}.")
    if a1.shape != a2.shape:
        raise ValueError(f"Arrays must have the same shape, but have shapes {a1.shape} and {a2.shape}.")
    MSE = jnp.average( jnp.abs(a1-a2)**2 )
    avg_a1 = jnp.average( a1 )
    MSV_a1 = jnp.average( jnp.abs(a1 - avg_a1)**2 )
    avg_a2 = jnp.average( a2 )
    MSV_a2 = jnp.average( jnp.abs(a2 - avg_a2)**2 )
    MSV_max = max( MSV_a1, MSV_a2 )

    ratio = MSE / (MSV_max+1e-12)

    if ratio<1e-12:
        statement="precisely equal"
    elif ratio<1e-6:
        statement="almost precisely equal"
    elif ratio<1e-4:
        statement="very close"
    elif ratio<1e-2:
        statement="pretty close"
    else:
        statement="not really close"

    return {'statement': statement, 'ratio': ratio}

      
# Possiblke toolboxes-------------------------------------
toolboxes = {'eq_motion': [approx_equal, 
                           get_all_previously_saved_results,
                           plot_from_code,
                           execute_code,],
             'coding': [approx_equal, 
                        get_all_previously_saved_results,
                        plot_from_code,
                        execute_code,],
            }

# Plot tools for initial conditions-------------------------------------
def plot_single_1d_vst(input_dict:dict):
    for key, value in input_dict.items():
        if isinstance(value, jax.Array) and value.ndim == 1:
            ts = value
        elif isinstance(value, jax.Array) and value.ndim == 2:
            q_qdot = value
        else:
            raise ValueError(f"Field {key} is not a 1d array.")
    # plot q vs t, ddot vs t, and q vs qdot
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].plot(ts, q_qdot[:, 0], label='q', color='blue')
    ax[0].set_ylabel('q')
    ax[0].set_title('q vs t')
    ax[0].set_xlabel('t')
    ax[0].legend()
    ax[1].plot(ts, q_qdot[:, 1], label='qdot', color='orange')
    ax[1].set_ylabel('qdot')
    ax[1].set_title('qdot vs t')
    ax[1].set_xlabel('t')
    ax[1].legend()
    plt.tight_layout()
    plt.show()

def plot_single_1d(input_dict:dict):
    for key, value in input_dict.items():
        if isinstance(value, jax.Array) and value.ndim == 1:
            ts = value
        elif isinstance(value, jax.Array) and value.ndim == 2:
            q_qdot = value
            dim = value.shape[1]
        else:
            raise ValueError(f"Field {key} is not a 1d array.")
    if dim == 2:
        # plot q vs t, ddot vs t, and q vs qdot
        fig, ax = plt.subplots(3, 1, figsize=(10, 8))
        ax[0].plot(ts, q_qdot[:, 0], label='q', color='blue')
        ax[0].set_ylabel('q')
        ax[0].set_title('q vs t')
        ax[0].set_xlabel('t')
        ax[0].legend()
        ax[1].plot(ts, q_qdot[:, 1], label='qdot', color='orange')
        ax[1].set_ylabel('qdot')
        ax[1].set_title('qdot vs t')
        ax[1].set_xlabel('t')
        ax[1].legend()
        ax[2].plot(q_qdot[:, 0], q_qdot[:, 1], label='qdot vs q', color='green')
        ax[2].set_xlabel('q')
        ax[2].set_ylabel('qdot')
        ax[2].set_title('qdot vs q')
        ax[2].legend()
    elif dim == 4:
        fig, ax = plt.subplots(4, 1, figsize=(12, 8))
        ax[0].plot(ts, q_qdot[:, 0], label='q0')
        ax[0].plot(ts, q_qdot[:, 1], label='q1')
        ax[0].set_ylabel('q')
        ax[0].set_title('q vs t')
        ax[0].set_xlabel('t')
        ax[0].legend()
        ax[1].plot(ts, q_qdot[:, 2], label='qdot0')
        ax[1].plot(ts, q_qdot[:, 3], label='qdot1')
        ax[1].set_ylabel('qdot')
        ax[1].set_title('qdot vs t')
        ax[1].set_xlabel('t')
        ax[1].legend()
        ax[2].plot(q_qdot[:, 0], q_qdot[:, 2], label='qdot0 vs q0')
        ax[2].plot(q_qdot[:, 1], q_qdot[:, 3], label='qdot1 vs q1')
        ax[2].set_xlabel('q')
        ax[2].set_ylabel('qdot')
        ax[2].set_title('qdot vs q')
        ax[2].legend()
        ax[3].plot(q_qdot[:, 0], q_qdot[:, 1], label='q1 vs q0')
        ax[3].set_xlabel('q0')
        ax[3].set_ylabel('q1')
        ax[3].set_title('q1 vs q0')
        ax[3].legend()
    else:
        raise NotImplementedError()
    plt.tight_layout()
    plt.show()

def plot_single_2d(input_dict:dict):
    for key, value in input_dict.items():
        if isinstance(value, jax.Array) and value.ndim == 1:
            ts = value
        elif isinstance(value, jax.Array) and value.ndim == 2:
            q_qdot = value
        else:
            raise ValueError(f"Field {key} is not a 1d array.")
    # plot q vs t, ddot vs t, and q vs qdot
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    ax[0].plot(ts, q_qdot[:, 0], label='q0')
    ax[0].set_ylabel('q')
    ax[0].set_title('q vs t')
    ax[0].set_xlabel('t')
    ax[0].legend()
    ax[1].plot(ts, q_qdot[:, 1], label='qdot', color='orange')
    ax[1].set_ylabel('qdot')
    ax[1].set_title('qdot vs t')
    ax[1].set_xlabel('t')
    ax[1].legend()
    ax[2].plot(q_qdot[:, 0], q_qdot[:, 1], label='qdot vs q', color='green')
    ax[2].set_xlabel('q')
    ax[2].set_ylabel('qdot')
    ax[2].set_title('qdot vs q')
    ax[2].legend()
    plt.tight_layout()
    plt.show()
