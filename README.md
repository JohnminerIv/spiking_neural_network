# Spiking Neural Network (SNN) Implementaion

The goal of this project is to implement and train an SNN in a modular way.

Usage:

Clone the repo \
`$ python3 -m pip install -r requirements.txt` \
`$ jupyter notebook` \
Then open the file with the name the_project.ipynb



## Project Summary: 
### What is the goal? What do I plan to have accomplished by the end of the project?
The goal is to build a simple framework that I and possibly others can use to rapidly develop and train SNNs. This means I should have an interface to decide how many neurons are in the network, what the inputs should be, what the output should be and the type of learning algorithm the network should use.\

The MVP would be to simply build a SSN and train it to compare it against other models on some task.\

The strech goals are implementing any sort of interface for building models of different sizes or types.

### Is this possible to build in the next few weeks? \\ Implementation plan
I think that it is possible to implement this entire project in the next few weeks. I already have different implementations of the basic structure of the network here in the repo. The next steps are as folows:
1. Choose which neron and model implementation to use for the project.
2. Build the learning algorithm, which will be Spike-time-dependant-placticity(stdp)
3. Train the model on a data set and compare it to other models
4. Build an interface for building SSN's with different shapes and sizes so I don't have to build it manually everytime I want to build a new one.

### What is its potential value?

SNN’s are potentially more efficient to train and more efficient at processing data in production. The reason being that they can speed up by using more cores because their algorithm can be efficiently parallelized. This is also practical for edge computing or prediction directly on a user’s laptop because they won’t need to have a huge GPU or TPU hooked up in order to do predictions on large networks.\

SNN's are also stateful which means they have a lot of potential use for analyzing time series data.

### Describe the data set you will use. What is the URL link?
I am going to use the MNIST dataset to train the proof of concept models because its a simple dataset thats easy to work with. It can also help me benchmark my SNN against something that regular neural networks do very well.\

I also want to use the MNIST but feed the network a series of numbers for variable amounts of time and have them predict the number that they saw for the longest which may or may not work. I think it's something that a human could do, so a computer should be able to do it, but I don't know if LSTMs can do that, so I'm going to train both an LSTM and a SNN to do that hopefully.

### How much disk space will you need?
Exactly 11594722 bytes according to the MNIST website.

### Research summary
#### How have others approached similar problems in the past?
People have implemented models that train via Spike Time Dependent Plasticity(STDP), and also Hebbian learning.
#### What compute resources will you need? Do you think the project could be done on Google Colab free version?
I could probably build this on my computer, at least I think I can.
#### What would you have to learn to complete this project?
I’ll have to learn how to efficiently model a neuron, how to implement STDP and Hebbian learning algorithms which are used for supervised learning. I also want to look into developing reinforcement learning algorithms for SNNs.
#### Provide URL links AND summaries of a few related articles and code repos that you've read and studied.
SNN implementation in python: https://github.com/Shikhargupta/Spiking-Neural-Network \
They implement a version of a spiking neural network but I don’t think it’s very efficient \
Introduction to SNN: https://towardsdatascience.com/spiking-neural-networks-the-next-generation-of-machine-learning-84e167f4eb2b \
They discuss what an SNN is, how they work, why they are more computationally powerful than traditional neural networks. \
Here are a few research papers that I am reading in order to help flesh out my knowledge on neurons: \
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3779838/ \
https://www.frontiersin.org/articles/10.3389/fncir.2016.00023/full \
These articles generally go over how neurons work, how learning is thought to happen, and its connection with positive rewards. \
https://www.frontiersin.org/articles/10.3389/fnins.2018.00774/full This paper is like a meta-analysis of a bunch of different research areas in SNN’s I’ll have to read it more later. \ 
https://github.com/arnogranier/SNN This guy basically already built what I want to build but it is hard for me to understand his architecture. His Simulations seem to be very biologically accurate, which is cool. But I don’t know if that’s something I want to strive for in my implementation. \
https://www.researchgate.net/publication/336917842_Asynchronous_Spiking_Neurons_the_Natural_Key_to_Exploit_Temporal_Sparsity Here is a research paper where the discuss building an asynchronous type of model that is more optimal for situations where there isn’t a lot of data over a longish period of time.
