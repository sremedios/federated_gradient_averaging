import numpy as np

def pad(img):
    '''
    Pads at end of img tensor to next multiple of 2^5, the number of
    times the dimensions of the feature map is cut in half by reduced Unet.
    
    To undo, call unpad().
    '''
    
    target_dims = [int(np.ceil(x/32)) * 32 for x in img.shape[:-1]]
    
    pads = [
        (0, t - c) for t, c in zip(target_dims, img.shape)
    ]
    
    # skip padding of num slices
    pads.append((0,0))
    
    return np.pad(img, pads, 'constant', constant_values=0)

def unpad(img, target_shape):
    '''
    Assumes padding at the end of the img tensor as by pad() function.
    '''
    return img[:target_shape[0], :target_shape[1], :target_shape[2]]