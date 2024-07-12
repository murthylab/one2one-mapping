# one2one-mapping
Model weights, example stimuli, and code for the paper "One-to-one mapping between deep network units and real neurons uncovers a visual population code for social behavior".

## KO Simulations
To test knockout training with a toy problem (two-layer linear network), check out ```/code/extfig2_KO_simulations```.  This contains Python notebooks that can be run in Google colab. We show that knockout training can successfully recover the ground truth changes in behavior as well as ground truth responses!

## Getting LC responses and behavior from 1-to-1 network
Various scripts are in /code/ that show how to pass stimuli through the 1-to-1 network (KO, DO, and noKO networks are available). Also included is how to generate fictive female stimuli from the tracked joint positions of the male and female fruit flies. 
Note: Some of this code requires the tensorflow imports as well as GPU access.

## Issues/concerns
Please let us know if you have trouble running the code or have questions about the code in the "Issues" tab.
