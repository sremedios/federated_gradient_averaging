import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# disable GPU for server
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
import datetime
import time
import json
from itertools import cycle
from pathlib import Path

from flask import Flask, request
import pickle

import numpy as np
import tensorflow as tf

from models.cnn import *

app = Flask(__name__)

allowed_institutes = ["A", "B"]

avg_vals = None
expected_vals = {"A": None, "B": None}
returned_val = {"A": False, "B": False}
converged_dict = {"A": False, "B": False}
terminate_requests = {"A": False, "B": False}

# For model averaging, the server must keep a copy of the weights
server_weights = cnn(iter(get_kernel_initializer())).get_weights()

@app.route('/kernel_init')
def get_kernel_init():
    k_iter = iter(get_kernel_initializer())
    
    # this number should be at least the number of expected
    # layers in the model
    n_expected_layers = 100
    kernel_initializer = [next(k_iter) for _ in range(n_expected_layers)]

    return pickle.dumps(kernel_initializer)

@app.route('/get_converged')
def get_converged():
    global converged_dict
    
    # Return True if any site has converged
    return pickle.dumps(any(g for g in converged_dict.values()))

@app.route('/put_converged', methods=['PUT'])
def put_converged():
    global converged_dict
    
    # parse header
    h = request.headers
    key = h["site"]
    
    # update state
    converged_dict[key] = True
    
    # operation completed successfully
    return pickle.dumps(True)

@app.route('/put_val', methods=['PUT'])
def put_val():
    # Typically polled once per batch
    global expected_vals
    global avg_vals

    # parse header
    h = request.headers
    key = h["site"]
    
    # receive vals from client
    client_vals = pickle.loads(request.data)

    # store vals in local dict
    if expected_vals[key] is None:
        expected_vals[key] = client_vals

    # Check if ready to average and perform average
    received_all_vals = all(g for g in expected_vals.values())
    
    if received_all_vals:
        # perform average
        avg_vals = _average_vals(expected_vals)
        # reset `expected_vals`; no longer needed bc average already calculated
        expected_vals = {k: None for k in expected_vals.keys()}
        
    # operation completed successfully
    return pickle.dumps(True)

def _average_vals(expected_vals):
    # calculates mean of weights and biases along correct axis for N models
    return [tf.reduce_mean(model_tuple, axis=0) for model_tuple in zip(*expected_vals.values())]

@app.route('/get_avg_val', methods=['GET'])
def get_avg_val():
 
    # polled many times per batch until all sites are ready
    global avg_vals
    global returned_val
    global server_weights

    h = request.headers
    key = h["site"]
    val_type = h["val_type"]

    if val_type == "gradients":
        # there are two situations: The average val is ready or it is not
        avg_val_ready = avg_vals is not None
        
        if avg_val_ready:
            
            # deep copy vals to permit resetting global `avg_vals`
            # while also being able to return their values in the same call
            return_val = [g.numpy().copy() for g in avg_vals]
            
            # flag this site as having returned the value
            returned_val[key] = True
            
            # reset `avg_vals` once all vals have been returned
            all_vals_returned = all(v for v in returned_val.values())

            if all_vals_returned:
                # reset avg
                avg_vals = None
                # reset tracker
                returned_val = {k: False for k in returned_val.keys()}
                
        else:
            return_val = None

    elif val_type == "weights":
        # there are two situations: The average val is ready or it is not
        avg_val_ready = avg_vals is not None
        
        if avg_val_ready:
            # update server model with weighted average of `avg_vals`, the client server weights
            # For us, each client does 1 epoch so the weighting scalar is 1
            server_weights = [a + b for (a,b) in zip(server_weights, avg_vals)]

            # set return value as new weights
            return_val = server_weights

            # flag this site as having returned the value
            returned_val[key] = True
            
            # reset `avg_vals` once all vals have been returned
            all_vals_returned = all(v for v in returned_val.values())

            if all_vals_returned:
                # reset avg
                avg_vals = None
                # reset tracker
                returned_val = {k: False for k in returned_val.keys()}
                
        else:
            return_val = None
    
    return pickle.dumps(return_val)

@app.route('/put_cyclic_weights', methods=['PUT'])
def put_cyclic_weight():
    # Typically polled once per batch
    global expected_vals

    # parse header
    h = request.headers
    key = h["site"]
    
    # receive vals from client
    client_vals = pickle.loads(request.data)

    # store vals in local dict
    if expected_vals[key] is None:
        expected_vals[key] = client_vals

    # operation completed successfully
    return pickle.dumps(True)

@app.route('/get_cyclic_weights', methods=['GET'])
def get_cyclic_weight():
    '''
    This runs cyclic weight transfer in parallel
    Site A will train and store weights in expected_vals['A']
    Site B will train and store weights in expected_vals['B']
    ...
    Site Z will train and store weights in expected_vals['Z']

    When finished, site B will call this function and request
    weights from the previous site, in this case site A.

    If the weights at `expected_vals['A']` are not `None`:
        1. Make a deep copy of the weight values
        2. set the weight value at `expected_vals['A']` to `None`
        3. return the copied values
    
    '''

    # polled many times per batch until all sites are ready
    global expected_vals

    h = request.headers
    key = h["site"]

    # get cyclic reverse iterator over expected_vals keys
    keys = list(expected_vals.keys())
    cyclic_iter = cycle(iter(reversed(sorted(keys))))

    # step until current key
    cur = None
    while cur != key:
        cur = next(cyclic_iter)

    # The result is the next `cyclic_iter` will be the previous site
    prev_site_key = next(cyclic_iter)

    if expected_vals[prev_site_key] is not None:
        # deep copy vals to permit resetting
        # while also being able to return their values in the same call
        return_val = [vals.numpy().copy() for vals in expected_vals[prev_site_key]]
        # reset prev_site_key
        expected_vals[prev_site_key] = None
    else:
        return_val = None
    
    return pickle.dumps(return_val)



if __name__ == '__main__':
    app.run(port=10203)
