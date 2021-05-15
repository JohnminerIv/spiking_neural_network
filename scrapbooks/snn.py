from __future__ import print_function
import datetime as dt
import numpy as np
import math
import matplotlib.pyplot as plt


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Process
from multiprocessing import Queue
import multiprocessing as mp
import time
import random
import dill as pickle
if __name__ == '__main__':
    m = mp.Manager()

    def wrapper(func, *args, **kwargs):
        def wrapped():
            return func(*args, **kwargs)
        return wrapped

    SENTINEL = 'NO WORK'
    def facilitate_fire(queue):
        worker_name = mp.current_process().name
        while True:
            try:
                task = queue.get_nowait()
            except:
                # print(worker_name + ' found an empty queue. Sleeping for a while before checking again...')
                time.sleep(0.01)
            else:
                if task == SENTINEL:
                    print(worker_name + ' no more work left to be done. Exiting...')
                    queue.put(SENTINEL)
                    break
                for func in task():
                    queue.put(func)

    class Network:
        def __init__(self):
            self.queue = m.Queue()
            self.outputs = m.Queue()
            self.layers = []
        
        def add(self, layer):
            self.layers.append(layer)
        
        def full_connect(self):
            for i in range(len(self.layers)):
                for neuron in self.layers[i].neurons:
                    if i + 1 < len(self.layers):
                        for r_neuron in self.layers[i + 1].neurons:
                            neuron.add_recipient(r_neuron)
    
        def train(self, full_inputs, full_outputs, time_per_image, worker_amount):
            processes = []
            results = []
            for i in range(worker_amount):
                p = Process(target=facilitate_fire, args=[self.queue])
                p.name = 'worker' + str(i)
                processes.append(p)
                p.start()
            if len(full_inputs) != len(full_outputs):
                print('you have done something wrong!')
                return None
            for layer in self.layers:
                layer.set_update(True)
            for i in range(len(full_inputs)):
                self.layers[-1].set_training_outputs(full_outputs[i], self.outputs)
                for _ in range(time_per_image):
                    self.layers[0].receive_inputs(full_inputs[i], self.queue)
                try:
                    output = self.outputs.get_nowait()
                except:
                    output = None
                while output != None:
                    print(f'expected {np.argmax(full_outputs[i])} got {output}')
                    results.append(np.argmax(full_outputs[i]), output)
                    try:
                        output = self.outputs.get_nowait()
                    except:
                        output = None
            self.queue.put(SENTINEL)
            return results
        
        def predict(self, full_inputs, time_per_image, worker_amount):
            processes = []
            results = []
            for i in range(worker_amount):
                p = Process(target=facilitate_fire, args=[self.queue])
                p.name = 'worker' + str(i)
                processes.append(p)
                p.start()
            for layer in self.layers:
                layer.set_update(False)
            for i in range(len(full_inputs)):
                self.layers[-1].set_training_outputs([0 for _ in range(len(self.layers[-1].neurons))], self.outputs)
                for _ in range(time_per_image):
                    self.layers[0].receive_inputs(full_inputs[i], self.queue)
                try:
                    output = self.outputs.get_nowait()
                except:
                    output = None
                while output != None:
                    print(f'{output}')
                    results.append(output)
                    try:
                        output = self.outputs.get_nowait()
                    except:
                        output = None
            self.queue.put(SENTINEL)
            return results
                

            
        def receive_inputs(self, inputs):
            if len(inputs) != len(self.layers[0]):
                print('input len != len of first layer')
            self.layers[0].receive_input(inputs, self.queue)

        def set_training_outputs(self, desired_outputs):
            if len(desired_outputs) != len(self.layers[-1]):
                print('input len != len of first layer')
            self.layers[0].set_training_outputs(desired_outputs, self.outputs)
            
        
        def sparse_connect(self):
            pass
            

    class Layer:
        def __init__(self, size):
            self.neurons = [Neuron(i) for i in range(size)]
        
        def receive_inputs(self, inputs, queue):
            for i in range(len(inputs)):
                for func in self.neurons[i].receive_input(inputs[i]):
                    queue.put(func)

        def set_training_outputs(self, outputs, queue):
            for i in range(len(outputs)):
                self.neurons[i].set_outputs(outputs[i], queue)

        def set_update(self, b):
            for neuron in self.neurons:
                neuron.set_weight_update(b)

    class Neuron:
        def __init__(self, num):
            # a unique identifier
            self.num = num
            # When the neuron will fire
            self.action_potential = 105.0
            # The membrane potential when it was last checked
            self.membrane_potential = 0
            # references to neurons that will increase this one's potential
            self.incomming_connections = {}
            # references to neurons that will be increased by this one
            self.outgoing_connections = []
            # resting potential
            self.resting_potential = 0
            # leak ammount
            self.leak = 0.2
            # last fire
            self.last_fire = dt.datetime.now()
            # how long the neron has to physically wait befor it can fire again.
            self.fire_rate = dt.timedelta(microseconds=500)
            # expected output
            self.expected_output = None
            self.output_queue = None
            self.update_weights = True
        
        def set_weight_update(self, b):
            self.update_weights = b

        def set_outputs(self, output, queue):
            self.expected_output = output
            self.output_queue = queue
            
        
        def add_recipient(self, neuron):
            """
            Add a connection to a new neuron.
            """
            self.outgoing_connections.append(neuron)
        
        def leak(self):
            """
            A function that will cause a constant leakage of membrane potential.
            This keeps the neuron near "equilibrium".
            """
            if self.membrane_potential < self.resting_potential:
                # If we are below the resting potential we rise to it.
                self.membrane_potential += self.leak
            else:
                # If we are above the resting potential we lower to it.
                self.membrane_potential -= self.leak

        def fire(self):
            self.last_fire = dt.datetime.now()
            self.membrane_potential = -10
            return [wrapper(c.receive_input, self.num) for c in self.outgoing_connections]  
        
        def receive_input(self, amount):
            self.membrane_potential += amount
            if (dt.datetime.now() - self.last_fire) > self.fire_rate and self.membrane_potential >= self.action_potential:
                return self.fire()
            return []
        
        def receive_fire(self, neuron):
            weight = self.incomming_connections.get(neuron)
            if weight is not None:
                self.membrane_potential += weight
            else:
                # setting initial weight
                self.incomming_connections[neuron] = random.random() * 10 + 10
                self.membrane_potential += self.incomming_connections[neuron]
                
            if (dt.datetime.now() - self.last_fire) > self.fire_rate and self.membrane_potential >= self.action_potential:
                if self.expected_output != None:
                    self.output_queue.put(self.num)
                    if self.expected_output == 0 and self.update_weights:
                        self.incomming_connections[neuron] -= 1.5
                if (self.update_weights):
                    self.incomming_connections[neuron] += 1
                return self.fire()
            if self.update_weights:
                self.incomming_connections[neuron] -= 0.1 
            return []


    neuron0 = Neuron(0)
    neuron1 = Neuron(1)
    neuron0.add_recipient(neuron1)
    now = dt.datetime.now()

    for i in range(1000):
        # print(neuron0.membrane_potential)
        if len(neuron0.receive_input(1)) > 0:
            print(f'fire {dt.datetime.now() - now}')
    queue = Queue()
    lol = 10
    layer1 = Layer(lol)
    layer2 = Layer(lol)

    for i in range(lol):
        for j in range(lol):
            layer1.neurons[i].add_recipient(layer2.neurons[j])

    for i in range(1000):
        layer1.receive_inputs([random.random() for _ in range(lol)], queue)
    data = [[random.random() for __ in range(784)] for _ in range(10)]
    labels = [[1 if random.random() > 0.5 else 0 for __ in range(2)] for _ in range(10)]
    model = Network()
    model.add(Layer(784))
    model.add(Layer(2))
    model.full_connect()
    print(model.train(data, labels, 200, 6))
    print(model.predict([data[-1]], 500, 6))