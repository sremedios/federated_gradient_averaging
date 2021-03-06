from tfdeterminism import patch
import random
from models import reduced_unet
from models import resnet
from models import cnn
import tensorflow as tf
import numpy as np
import pickle
from flask import Flask, request
from pathlib import Path
from itertools import cycle
import json
import time
import datetime
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# disable GPU for server
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Determinism
patch()
SEED = 0
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


dataset = sys.argv[1]  # MNIST or CT


app = Flask(__name__)

allowed_institutes = ["A", "B"]

avg_vals = None
expected_vals = {"A": None, "B": None}
returned_val = {"A": False, "B": False}
server_step = None


# For weight averaging, the server must keep a copy of the weights
if dataset == "MNIST":
    k_init = iter(cnn.get_kernel_initializer())
    server_weights = cnn.cnn(k_init, n_channels=1,
                             n_classes=10).trainable_variables
elif dataset == "CT":
    tf.keras.backend.set_floatx('float32')
    k_init = iter(reduced_unet.get_kernel_initializer())
    server_weights = reduced_unet.reduced_unet(
        k_init, ds=8).trainable_variables


@app.route('/kernel_init')
def get_kernel_init():
    if dataset == "MNIST":
        k_iter = iter(cnn.get_kernel_initializer())
    elif dataset == "CT":
        k_iter = iter(reduced_unet.get_kernel_initializer())

    # this number should be at least the number of expected
    # layers in the model
    n_expected_layers = 1000
    kernel_initializer = [next(k_iter) for _ in range(n_expected_layers)]

    return pickle.dumps(kernel_initializer)


@app.route('/put_val', methods=['PUT'])
def put_val():
    # Typically polled once per batch
    global expected_vals
    global avg_vals
    global server_step
    global server_weights

    # parse header
    h = request.headers
    key = h["site"]
    client_step = int(h['step'])
    mode = h['mode']

    # init
    if server_step == None:
        server_step = client_step

    print("=== {} PUT | Client Step {} | Server Step {} ===".format(
        key,
        client_step,
        server_step,
    ))

    # put on lockstep
    synchronized = server_step == client_step
    if not synchronized:
        return pickle.dumps(False)

    # receive vals from client
    client_vals = pickle.loads(request.data)

    # store vals in local dict
    expected_vals[key] = client_vals

    '''
    For both federated and weightavg modes, average the received values
    For cyclic mode, do not average
    '''
    if mode != "cyclic":
        # Check if ready to average and perform average
        received_all_vals = all(g for g in expected_vals.values())

        if received_all_vals:
            avg_vals = _average_vals(expected_vals)

            if mode == "weightavg":
                # update server weights with averaged client weight differences
                server_weights = [
                    a + b for (a, b) in zip(server_weights, avg_vals)]
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
    global server_step

    h = request.headers
    key = h["site"]
    val_type = h["val_type"]
    client_step = h['step']

    print("=== {} GET | Client Step {} | Server Step {} ===".format(
        key,
        client_step,
        server_step,
    ))

    # there are two situations: The average val is ready or it is not
    avg_val_ready = avg_vals is not None

    if avg_val_ready:

        # Handle weights and gradients slightly differently

        if val_type == "gradients":
            # deep copy vals to permit resetting global `avg_vals`
            # while also being able to return their values in the same call
            return_val = [g.numpy().copy() for g in avg_vals]
        elif val_type == "weights":
            # set return value as new weights
            return_val = server_weights

        # flag this site as having returned the value
        returned_val[key] = True

        # reset `avg_vals` once all vals have been returned
        all_vals_returned = all(v for v in returned_val.values())

        if all_vals_returned:
            # increment server step
            server_step += 1
            # reset avg
            avg_vals = None
            # reset tracker
            returned_val = {k: False for k in returned_val.keys()}

    else:
        return_val = None

    return pickle.dumps(return_val)


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
    global returned_val
    global server_step

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
        return_val = [vals.numpy().copy()
                      for vals in expected_vals[prev_site_key]]
        # reset prev_site_key
        expected_vals[prev_site_key] = None

        # flag this site as having returned the value
        returned_val[key] = True

        # reset `avg_vals` once all vals have been returned
        all_vals_returned = all(v for v in returned_val.values())

        if all_vals_returned:
            # increment server step
            server_step += 1
            # reset tracker
            returned_val = {k: False for k in returned_val.keys()}
    else:
        return_val = None

    return pickle.dumps(return_val)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10203)
