# Project for Computational Linguistics
## An analysis of idiom sentence similarities for different transformer models
This project investigates on cosine similarities of sentence embeddings. The whole dataset can be found in data/sentences.txt\
In model_experiments.py the code for generating all datasets can be found. It includes model-loading from the huggingface library, similarity calculations and different score extractions. Exports are saved in data_exports/\
evaluations.py was used to generate plots from that data obtained in model_experiments.py. Plots are saved in evaluation_exports/
