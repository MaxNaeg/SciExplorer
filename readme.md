# SciExplorer

This repository contains the code for the SciExplorer framework. SciExplorer builds on LLM APIs to create artificial scientist agents that can explore unknown systems by controlling experiments and running analysis code.

## Publication

SciExplorer was introduced and used in the following publication (please cite if you start using SciExplorer):

Maximilian Nägele, Florian Marquardt: "Agentic Exploration of Physics Models", arXiv:2509.24978 https://arxiv.org/abs/2509.24978


## Installation

To use the sciexplorer module, install the package using 'pip install -e .' in the main directory of this repository.



## WARNING

The tools provided to the agent enable automatic execution of LLM generated Python code. While we did not observe the agent acting malicious in our experiments, these tools should best be run in a safe environment.



## SciExplorer Framework

- The main function is run_experiment in run_exp.py, which runs the iterative exploration of a system using repeated reasoning and tool-use steps. For an example, see sciexplorer/example_runs/example_define_new_simulator_analysis.ipynb.

- The function run of runner.py builds on top of this to automatically load predefined simulators, analysis tools, and prompts and save the resulting exploration to a file. For an example, see sciexplorer/example_runs/example_eom_run.ipynb.

- analyze_conv_utils contains files to print, compare, and analyze saved conversations.

- For an example of how to conveniently define a new experiment from scratch, see sciexplorer/example_runs/example_define_new_simulator_analysis.ipynb.



## Example implementations



- examples/simulators contains a separate Python file for each physics simulator.

- examples/tasks contains a markdown file defining system-independent tasks.

- examples/prompts contains markdown files defining system prompts and intermediate prompts.

- examples/analysis contains different tools for analysing experiments.



## Results

The results obtained using the SciExplorer framework are available in the GitHub repository [SciExplorerResults](https://github.com/MaxNaeg/SciExplorerResults.git).







