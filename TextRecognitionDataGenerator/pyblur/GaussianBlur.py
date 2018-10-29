import numpy as np
from PIL import ImageFilter

# gaussianbandwidths = [1, 1.5, 2] #, 2, 2.5, 3, 3.5]
gaussianbandwidths = np.random.choice([1, 1.5, 2], 1, p=[0.5, 0.3, 0.2])[0]

def GaussianBlur_random(img):
    gaussianidx = np.random.randint(0, len(gaussianbandwidths))
    gaussianbandwidth = gaussianbandwidths[gaussianidx]
    return GaussianBlur(img, gaussianbandwidth)

def GaussianBlur(img, bandwidth):
    img = img.filter(ImageFilter.GaussianBlur(bandwidth))
    return img