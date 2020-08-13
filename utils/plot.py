import matplotlib.pyplot as plt
import numpy as np
import time
from IPython import display

plt.rcParams['figure.figsize'] = (20, 20)
plt.rcParams['image.cmap'] = 'Greys_r'
plt.rcParams['image.interpolation'] = 'none'

def multiplot(imgs, titles, vmin=None, vmax=None):
    fig, axs = plt.subplots(1, len(imgs))

    for ax, img, title in zip(axs, imgs, titles):
        if vmin is None:
            cur_vmin = img.min()
        if vmax is None:
            cur_vmax = img.max()
        ax.imshow(np.rot90(img), vmin=cur_vmin, vmax=cur_vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()
    
def anim_paired_patches(lr_patch, hr_patch):
    fig, axs = plt.subplots(1, 2)
    
    vmin = lr_patch.min()
    vmax = lr_patch.max()
    
    axs[0].imshow(np.rot90(lr_patch), vmin=vmin, vmax=vmax)
    axs[1].imshow(np.rot90(hr_patch), vmin=vmin, vmax=vmax)
    
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    display.display(plt.show())
    display.clear_output(wait=True)
    time.sleep(.1)