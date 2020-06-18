import tensorflow as tf
import numpy as np

def get_diffs(new_model, orig_model):
    weight_diffs = []
    for layer, layer_orig in\
            zip(new_model.trainable_variables, orig_model.trainable_variables):
        diff = layer.numpy() - layer_orig.numpy()
        weight_diffs.append(diff)
    return weight_diffs


def grad(model, loss, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def get_batch_gradients(models_list,
                        loss_list,
                        x_train_list,
                        y_train_list,
                        scaling,
                        opt):
    loss_values = []
    grads = [{"weights": [],
              "biases": []} for _ in range(len(models_list))]

    for i, (model, loss_fn, x_train, y_train) in\
            enumerate(zip(models_list, loss_list, x_train_list, y_train_list)):

        with tf.GradientTape(persistent=True) as gradient_tape:
            var_list = model.trainable_variables

            def cur_loss_fn():
                return loss_fn(model, x_train, y_train)

            loss_values.append(np.mean(cur_loss_fn().numpy()))

            grads_and_vars = opt.compute_gradients(cur_loss_fn,
                                                   var_list,
                                                   gradient_tape=gradient_tape)

        # discard the var_list element of each tuple
        g = [tup[0] for tup in grads_and_vars]

        for j in range(len(g)):
            # even indices of the gradient correspond to weights
            if j % 2 == 0:
                if scaling:
                    grads[i]["weights"].append(
                        tf.multiply(g[j], scaling[i]))
                else:
                    grads[i]["weights"].append(g[j])
            # odd indices of the gradient correspond to biases
            else:
                if scaling:
                    grads[i]["biases"].append(
                        tf.multiply(g[j], scaling[i]))
                else:
                    grads[i]["biases"].append(g[j])

    return grads, loss_values


def average_gradients(grads, scaling, shared_layer_indices):
    # collect weights and biases for arbitrary averaging
    w = []
    b = []
    for i, g in enumerate(grads):
        # TODO: figure out scaling
        #w.append(tf.multiply(g['weights'], scaling[i]))
        #b.append(tf.multiply(g['biases'], scaling[i]))
        w.append(g['weights'])
        b.append(g['biases'])

    avg_grads = {"weights": [],
                 "biases": []}
    # shared_layer_indices refers to the Keras network layer
    # These indices correspond to ones which skip nonparametric layers
    # if there is one nonparametric layer between parametric layers,
    # then we can apply the following transform
    # In the future, this needs to be made generic
    transformed_indices = [x//2 for x in shared_layer_indices]

    # average gradients arbitrarily
    for i, (weights_tuple, biases_tuple) in enumerate(zip(zip(*w), zip(*b))):
        if i in transformed_indices:
            avg_grads["weights"].append(
                tf.reduce_mean([*weights_tuple], axis=0).numpy())
            avg_grads["biases"].append(
                tf.reduce_mean([*biases_tuple], axis=0).numpy())
        else:
            avg_grads["weights"].append((weights_tuple[0] * 0).numpy())
            avg_grads["biases"].append((biases_tuple[0] * 0).numpy())

    return avg_grads


def apply_gradient(model, shared_layer_indices, grad, avg_grads, optimizer):
    cur_shared_grad_idx = 0
    cur_unique_grad_idx = 0
    for i in range(len(model.layers)):
        if model.layers[i].trainable and len(model.layers[i].get_weights()) > 1:
            if i in shared_layer_indices:
                optimizer.apply_gradients(zip([avg_grads["weights"][cur_shared_grad_idx],
                                               avg_grads["biases"][cur_shared_grad_idx]],
                                              model.layers[i].trainable_variables))
                # advance pointer for avg'd grads; indices do not line up with layers
                cur_shared_grad_idx += 1
            else:
                # otherwise, apply normal gradient to the model
                optimizer.apply_gradients(zip([grad["weights"][cur_unique_grad_idx],
                                               grad["biases"][cur_unique_grad_idx]],
                                              model.layers[i].trainable_variables))
            # advance pointers for unique grads alongside trainable layers
            cur_unique_grad_idx += 1
