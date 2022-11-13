from .OnlineFilter import LowpassFilter
from .ObjectContact import ObjectContact
from collections import deque
import numpy as np
from copy import deepcopy

class ObjectTrajectory(object):

    MEAN = 0
    POSITIVE_MEAN = 1
    NEGATIVE_MEAN = 2
    VARIANCE = 3

    def __init__(self, history_length: int = None):

        if history_length is None:
            self.history_length = 20
        else:
            self.history_length = history_length
        self.trajectory = np.zeros([3,2,history_length])
        self.contact_variance = np.zeros([history_length])
        self.contact = ObjectContact()
        self.framerate = 60
        self.speed = 0

        return

    def __enter__(self):
        return self

    def __exit__(
            self, 
            exception_type, 
            exception_value, 
            traceback):
        return
    
    def update_trajectory(self, event_frame: np.array, contact_threshold: int = 10):

        event_mean, event_positive_mean, event_negative_mean = self.contact.event_position_mean(event_frame, contact_threshold)
        if event_mean is None:
            current_position = self.trajectory[:,:,0]
            event_variance = self.contact_variance[0]
        else:
            event_variance = self.contact.event_position_variance(event_frame, event_mean)
            current_position = np.array([
                event_mean,
                event_positive_mean,
                event_negative_mean])

        for element in range(1, self.history_length):
            self.trajectory[:, :, self.history_length-element] = self.trajectory[:, :, self.history_length-(element+1)]
            self.contact_variance[self.history_length-element] = self.contact_variance[self.history_length-(element+1)]
        self.trajectory[:,:,0] = current_position
        self.contact_variance[0] = event_variance

        return self.trajectory, self.contact_variance