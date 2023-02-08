import numpy as np
from .SimulateEvent import SimulateEvent
from scipy.ndimage import gaussian_filter1d

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
    
    def abs_event_position_mean(self, event_frame: np.array):
        """
        Calcuates the center of all events.

        Parameters
        ----------
            `event_frame`       - the last event matrix generated\n

        Returns
        -------
            `center_avg`        - coordinates of average event location
        """

        event_avg = np.nonzero(np.abs(event_frame) == 1)

        if (len(event_avg[0]) < 1):
            event_mean = None
        else:
            event_mean = np.array([np.mean(event_avg[0]), np.mean(event_avg[1])])

        return event_mean

    def event_buffer_position_mean(self, event_buffer: np.array):
        """
        Calcuates the center of all events for each buffer frame.

        Parameters
        ----------
            `event_buffer`      - the current event buffer\n

        Returns
        -------
            `buffer_mean`       - coordinates list of average event location for each frame
        """

        buffer_length = event_buffer.shape[-1]
        buffer_mean = []
        
        for buffer in range(buffer_length):

            event_avg = np.nonzero(np.abs(event_buffer[:,:,buffer]) == 1)

            if (len(event_avg[0]) < 1):
                event_mean = None
            else:
                event_mean = np.array([np.mean(event_avg[0]), np.mean(event_avg[1])])
            buffer_mean.append(event_mean)

        return buffer_mean
    
    def event_position_variance(self, event_frame: np.array, event_mean: np.array):

        """
        Calcuates the variance of an event frame about the specified pixel coordinate.

        Parameters
        ----------
            `event_frame`       - the last event matrix generated\n
            `event_mean`        - the mean of event locations\n

        Returns
        -------
            `event_variance`    - mean Euclidean distance of events from the center point
        """
        event_variance = (((np.argwhere(np.abs(event_frame)==1) - event_mean)**2).sum(1)).mean()

        return event_variance
    
    def event_distance_from_line(
            self, 
            event_frame: np.array, 
            start_coord: np.array,
            end_coord: np.array):

        """
        Calcuates the mean distance of an event frame from the specified line.

        Parameters
        ----------
            `event_frame`       - the last event matrix generated\n
            `start_coord`       - lie starting point\n
            `end_coord`         - lie ending point\n

        Returns
        -------
            `event_dist`        - mean Euclidean distance of events from the specified line
        """

        # Line coefficients for ax+by+c = 0
        a = start_coord[1] - end_coord[1]
        b = end_coord[0] - start_coord[0]
        c = start_coord[0]*end_coord[1] - end_coord[0]*start_coord[1]
        
        event_dist = (np.abs((np.argwhere(np.abs(event_frame)==1)*np.array([a,b])).sum(1) + c) / np.sqrt(a**2 + b**2)).mean()

        return event_dist
    
    def event_variance_about_line(
            self, 
            event_frame: np.array, 
            start_coord: np.array,
            end_coord: np.array):

        """
        Calcuates the variance of an event frame about the specified line.

        Parameters
        ----------
            `event_frame`       - the last event matrix generated\n
            `start_coord`       - lie starting point\n
            `end_coord`         - lie ending point\n

        Returns
        -------
            `event_var`        - mean square Euclidean distance of events from the specified line
        """

        # Line coefficients for ax+by+c = 0
        a = start_coord[1] - end_coord[1]
        b = end_coord[0] - start_coord[0]
        c = start_coord[0]*end_coord[1] - end_coord[0]*start_coord[1]
        
        event_var = (np.square((np.argwhere(np.abs(event_frame)==1)*np.array([a,b])).sum(1) + c) / (a**2 + b**2)).mean()

        return event_var
    
    def event_kurtosis(self, event_frame, event_variance, event_mean):

        kurtosis = ((((np.argwhere(np.abs(event_frame)==1) - event_mean)**2).sum(1))**2 / event_variance**2).mean()

        return kurtosis

    def event_central_moments(self, event_frame):

        event_mean = self.abs_event_position_mean(event_frame)

        if event_mean is None:
            return None

        event_centralised_coordinate = np.argwhere(np.abs(event_frame)==1) - event_mean
        event_square_distances = (event_centralised_coordinate**2).sum(1)
        event_variance = event_square_distances.mean()
        event_kurtosis = (event_square_distances**2).mean() / event_variance**2

        return event_mean, event_variance, event_kurtosis

    def _diamond(self, n):

        a = np.arange(n)
        b = np.minimum(a,a[::-1])
        return (b[:,None]+b)>=(n-1)//2
        
    def is_contact(
            self, 
            event: SimulateEvent, 
            neighbour_search_region: np.array = np.ones((5,5)), 
            contact_threshold: int = 50) -> bool:
        
        event_neighbour_frame = event.event_neighbour_number(event.event_frame, neighbour_search_region)

        count = np.sum(event_neighbour_frame)
        if (count>contact_threshold):
            return True, count
        else:
            return False, count