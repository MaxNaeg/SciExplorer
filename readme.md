# SciExplorer version 2

This repository contains the code for the SciExplorer framework. SciExplorer builds on LLM APIs to create artificial scientist agents that can explore unknown systems by controlling experiments and running analysis code.

## Publication

SciExplorer was introduced and used in the following publication (please cite if you start using SciExplorer):

Maximilian Nägele, Florian Marquardt: "Agentic Exploration of Physics Models", arXiv:2509.24978 https://arxiv.org/abs/2509.24978

## WARNING

The tools provided to the agent enable automatic execution of LLM generated Python code. While we did not observe the agent acting malicious in our experiments, these tools should best be run in a save environment.

## Examples of things you can do with SciExplorer
- Create an agent with access to tools and an external memory (in the form of a dictionary storing past tool results) that iteratively solves a user defined task (examples/linear_exploration_example).
- Interactively chat with an AI with access to user-defined tools and external memory in a similar fashion to ChatGPT (examples/agentic_chat_example).
- Interactively chat with an AI about an already finished exploration, e.g. to ask follow-up quetsions (examples/human_follow_up_question_example).

## SciExplorer Framework

- The main function is run_exploration in sciexplorer/runs/linear_exploration.py, which runs the iterative exploration of a system using repeated reasoning and tool-use steps. For an example, see examples/linear_exploration_example.ipynb
- To learn how to define tools see examples/linear_exploration_example.ipynb
- For an example how to print saved conversations, see examples/print_previous_conversation.ipynb
- sciexplorer/tools holds the implementation of some useful default tools (coding, plotting, ...).
- sciexplorer/physics holds the implementation of some generic physics simulators.
- sciexplorer/runs containes different agentic loops, e.g. linear_exploration or human_in_ther_loop

## Installation

To use the sciexplorer module, clone this repository and install the package using 'pip install -e .' in the main directory containing the pyproject.toml file. SciExplorer was tested with python version 3.14.1.

## Results

The results obtained using the SciExplorer framework are available in the GitHub repository [SciExplorerResults](https://github.com/MaxNaeg/SciExplorerResults.git).








