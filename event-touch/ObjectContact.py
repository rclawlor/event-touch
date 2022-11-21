import numpy as np


class ObjectContact(object):

    def __init__(self):
        return

    def event_position_mean(self, event_frame: np.array, contact_event_threshold: int = 0):
        """
        Calcuates the center of all positive, negative and total events.

        Parameters
        ----------
            `event_frame`       - the last event matrix generated\n

        Returns
        -------
            `center_avg`        - coordinates of average event location\n
            `center_positive`   - coordinates of average positive event location\n
            `center_negative`   - coordinates of average negative event location
        """

        event_positive = np.nonzero(event_frame == 1)
        event_negative = np.nonzero(event_frame == -1)
        event_avg = [np.concatenate([event_positive[0], event_negative[0]]), np.concatenate([event_positive[1], event_negative[1]])]

        if (len(event_positive[0]) < 1 or len(event_negative[0]) < 1 or len(event_avg[0]) < contact_event_threshold):
            event_mean = None
            event_positive_mean = None
            event_negative_mean = None
        else:
            event_positive_mean = np.array([np.mean(event_positive[0]), np.mean(event_positive[1])])
            event_negative_mean = np.array([np.mean(event_negative[0]), np.mean(event_negative[1])])
            event_mean = np.array([np.mean(event_avg[0]), np.mean(event_avg[1])])

        return event_mean, event_positive_mean, event_negative_mean
    
    def event_position_variance(self, event_frame: np.array, event_mean: np.array):

        """
        Calcuates the center of all positive, negative and total events.

        Parameters
        ----------
            `event_frame`       - the last event matrix generated\n
            `event_mean`        - the mean of event locations\n

        Returns
        -------
            `event_variance`    - mean Euclidean distance of events from the center point
        """
        event_variance = np.sqrt(((np.argwhere(np.abs(event_frame)==1) - event_mean)**2).sum(1)).mean()

        return event_variance