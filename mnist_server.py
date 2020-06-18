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
expected_models = {"CNRM classification": None,
                   "VUMC classification": None, }
returned_grad = {"CNRM classification": False,
                 "VUMC classification": False, }

weight_init = tf.glorot_uniform_initializer(seed=0)

initial_layers = [
    tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same', kernel_initializer=weight_init,),
    tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same', kernel_initializer=weight_init,),
    tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same', kernel_initializer=weight_init,),
    tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same', kernel_initializer=weight_init,),
    tf.keras.layers.Dense(10, kernel_initializer=weight_init,),
]


@app.route('/init_layers')
def init_layers():
    global initial_layers
    
    return pickle.dumps(initial_layers)


@app.route('/put_grad', methods=['PUT'])
def put_grad():
    global expected_models

    h = request.headers
    client_grads = pickle.loads(request.data)

    key = h["institute"] + " " + h["task"]

    expected_models[key] = client_grads

    # operation completed successfully
    return pickle.dumps(True)


@app.route('/get_avg_grad', methods=['GET'])
def gradient_average():
    global expected_models
    global returned_grad

    h = request.headers
    key = h["institute"] + " " + h["task"]

    # Ready when have gradients from all expected models
    # and only send gradient if we haven't yet
    ready_condition = all(g for g in expected_models.values())\
                        and not returned_grad[key]

    if ready_condition:
        avg_grads = [tf.reduce_mean(model_tuple, axis=0)
                        for model_tuple in zip(*expected_models.values())]
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
