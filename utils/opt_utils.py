import tensorflow as tf

'''
Some utility functions to better investigate
momentums in the Adam optimizer
'''

def get_keys(opt):
    # Return zipped iter over (kernel name, bias name)
    key_names = opt._slots.keys()
    kernels = filter(lambda k: 'kernel' in k, key_names)
    biases = filter(lambda k: 'bias' in k, key_names)
    
    return zip(kernels, biases)

def get_momentums(opt, layer_name, slot_name):
    return opt._slots[layer_name][slot_name]