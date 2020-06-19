import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# disable GPU for server
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
import datetime
import time
import json
from pathlib import Path

from flask import Flask, request
import pickle

import numpy as np
import tensorflow as tf

from models.cnn import *

app = Flask(__name__)

allowed_institutes = ["A", "B"]
expected_grads = {"A": None,
                   "B": None, }
avg_grads = None
returned_grad = {"A": False,
                 "B": False, }


@app.route('/kernel_init')
def get_kernel_init():
    k_iter = iter(get_kernel_initializer())
    
    # this number should be at least the number of expected
    # layers in the model
    n_expected_layers = 100
    kernel_initializer = [next(k_iter) for _ in range(n_expected_layers)]

    return pickle.dumps(kernel_initializer)


@app.route('/put_grad', methods=['PUT'])
def put_grad():
    # Typically polled once per batch
    global expected_grads
    global avg_grads

    # parse header
    h = request.headers
    key = h["site"]
    
    # receive gradients from client
    client_grads = pickle.loads(request.data)

    # store gradients in local dict
    if expected_grads[key] is None:
        expected_grads[key] = client_grads

    # Check if ready to average and perform average
    received_all_grads = all(g for g in expected_grads.values())
    
    if received_all_grads:
        # perform average
        avg_grads = _average_grads(expected_grads)
        # reset `expected_grads`; no longer needed bc average already calculated
        expected_grads = {k: None for k in expected_grads.keys()}
        
    # operation completed successfully
    return pickle.dumps(True)

def _average_grads(expected_grads):
    # calculates mean of weights and biases along correct axis for N models
    return [tf.reduce_mean(model_tuple, axis=0) for model_tuple in zip(*expected_grads.values())]

@app.route('/get_avg_grad', methods=['GET'])
def get_avg_grad():
    # polled many times per batch until all sites are ready
    global avg_grads
    global returned_grad

    h = request.headers
    key = h["site"]

    # there are two situations: The average gradient is ready or it is not
    avg_gradient_ready = avg_grads is not None
    
    if avg_gradient_ready:
        
        # deep copy gradients to permit resetting global `avg_grads`
        # while also being able to return their values in the same call
        return_val = [g.numpy().copy() for g in avg_grads]
        
        # flag this site as having returned the value
        returned_grad[key] = True
         
        # reset `avg_grads` once all gradients have been returned
        all_grads_returned = all(v for v in returned_grad.values())

        if all_grads_returned:
            # reset avg
            avg_grads = None
            # reset tracker
            returned_grad = {k: False for k in returned_grad.keys()}
            
    else:
        return_val = None
    
    return pickle.dumps(return_val)



if __name__ == '__main__':
    app.run(port=10203)
