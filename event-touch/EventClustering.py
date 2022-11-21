import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage.filters import gaussian_filter

class EventClustering(object):

    def __init__(self):
        return

    def __enter__(self):
        return self
    
    def __exit__(
            self, 
            exception_type, 
            exception_value, 
            traceback):
        return
    
    
    def blur_event(self, event_frame, sigma, size):

        abs_event_frame = np.abs(event_frame)
        event_smooth = gaussian_filter(abs_event_frame, sigma)
        img = convolve2d(event_smooth, np.ones([size,size]), mode='same')

        return img

    