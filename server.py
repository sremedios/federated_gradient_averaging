from flask import Flask, request
import numpy as np
import pickle
import os
import tensorflow as tf
from utils.grad_ops import *
from models.shared_unet import init_full_unet

# disable GPU for server
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)
tf.enable_eager_execution()

allowed_institutes = ["VUMC", "CNRM"]
expected_models = {"CNRM segmentation": None,
                   "VUMC segmentation": None, }
num_grads = 0
returned_grad = {"CNRM segmentation": False,
                 "VUMC segmentation": False, }


initial_layers = init_full_unet(ds=2)


@app.route('/init_layers')
def init_layers():
    global initial_layers

    return pickle.dumps(initial_layers)


@app.route('/put_grad', methods=['PUT'])
def put_grad():
    global expected_models
    global num_grads

    h = request.headers
    client_grads, client_num_grads = pickle.loads(request.data)

    key = h["institute"] + " " + h["task"]

    expected_models[key] = client_grads
    num_grads += client_num_grads

    # operation completed successfully
    return pickle.dumps(True)


@app.route('/get_avg_grad', methods=['GET'])
def gradient_average():
    global expected_models
    global returned_grad
    global num_grads

    h = request.headers
    key = h["institute"] + " " + h["task"]

    # Ready when have gradients from all expected models
    # and only send gradient if we haven't yet
    ready_condition = all(g for g in expected_models.values())\
                        and not returned_grad[key]

    if ready_condition:
        avg_grads = [tf.reduce_sum(model_tuple, axis=0)
                     for model_tuple in zip(*expected_models.values())]
        avg_grads = [g / num_grads for g in avg_grads]
        x = pickle.dumps(avg_grads)

        returned_grad[key] = True

        # reset all when all gradients have been sent
        if all(v for v in returned_grad.values()):
            for k in returned_grad:
                returned_grad[k] = False

        return x

    # averaged gradients not yet ready -- waiting for other clients
    else:
        return pickle.dumps(False)


if __name__ == '__main__':
    app.run(port=10203)
