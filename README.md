# spiking_neural_network
Personal - DS - My implementation of an asynchronous SNN
Build my own Spiking Neural Network(SNN) Framework
Project Summary
-- What is the goal, i.e what would you plan to accomplish?
The goal is to build a framework that I and possibly others can use to rapidly develop and train SNNs at a high level like Keras.
-- What is its potential value?
SNN’s are potentially more efficient to train and more efficient at processing data in production. The reason being that they can speed up by using more cores because their algorithm can be efficiently parallelized. This is also practical for edge computing or prediction directly on a user’s laptop because they won’t need to have a huge GPU or TPU hooked up in order to do predictions on large networks.
-- Describe the data set you will use. What is the URL link?
I could probably use the MNIST dataset to train proof of concept models.
-- How much disk space will you need?
An MNIST dataset amount...
Research summary
-- How have others approached similar problems in the past?
People have implemented models that train via Spike Time Dependent Plasticity(STDP), and also Hebbian learning.
-- What compute resources will you need? Do you think the project could be done on Google Colab free version?
I could probably build this on my computer, at least I think I can.
-- What would you have to learn to complete this project?
I’ll have to learn how to efficiently model a neuron, how to implement STDP and Hebbian learning algorithms which are used for supervised learning. I also want to look into developing reinforcement learning algorithms for SNNs.
-- Provide URL links AND summaries of a few related articles and code repos that you've read and studied.
SNN implementation in python: https://github.com/Shikhargupta/Spiking-Neural-Network
They implement a version of a spiking neural network but I don’t think it’s very efficient
Introduction to SNN: https://towardsdatascience.com/spiking-neural-networks-the-next-generation-of-machine-learning-84e167f4eb2b
They discuss what an SNN is, how they work, why they are more computationally powerful than traditional neural networks.
Here are a few research papers that I am reading in order to help flesh out my knowledge on neurons:
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3779838/
https://www.frontiersin.org/articles/10.3389/fncir.2016.00023/full
These articles generally go over how neurons work, how learning is thought to happen, and its connection with positive rewards.
https://www.frontiersin.org/articles/10.3389/fnins.2018.00774/full This paper is like a meta-analysis of a bunch of different research areas in SNN’s I’ll have to read it more later.
https://github.com/arnogranier/SNN This guy basically already built what I want to build but it is hard for me to understand his architecture. His Simulations seem to be very biologically accurate, which is cool. But I don’t know if that’s something I want to strive for in my implementation.
https://www.researchgate.net/publication/336917842_Asynchronous_Spiking_Neurons_the_Natural_Key_to_Exploit_Temporal_Sparsity Here is a research paper where the discuss building an asynchronous type of model that is more optimal for situations where there isn’t a lot of data over a longish period of time.
