# EasyCompress: Automated Deep Learning Compression
This repository contains the source code for EasyCompress, a tool that automates and intelligently compresses deep learning models.    
You can conveniently use the tool through its interactive online interface [here](https://thesis.abelvansteenweghen.com).

The tool is designed to compress a model based on the degree of compression, the type of compression (model size, computations, or inference time), and a minimum required performance threshold. It uses pruning, knowledge distillation, and quantization techniques to achieve the desired compression configuration.

The code is organised as follows:
* ``./src/`` contains all the source code for the tool.
* ``./examples/`` contains notebooks that show how to use the tool.
* ``./experiments/`` contains code and notebooks for conducting experiments with the tool. 

EasyCompress was developed as a thesis project on automated model compression for deep learning.    
The thesis report can be found [here](https://url).
