import numpy as np

def contact_centre(event_frame: np.array, contact_event_threshold: int = 0):
    """
    Calcuates the centre of all positive, negative and total events.

    Parameters
    ----------
        `event`             - the last event matrix generated\n

    Returns
    -------
        `centre_avg`        - coordinates of average event location\n
        `centre_positive`   - coordinates of average positive event location\n
        `centre_negative`   - coordinates of average negative event location
    """

    event_positive = np.nonzero(event_frame == 1)
    event_negative = np.nonzero(event_frame == -1)
    event_avg = [np.concatenate([event_positive[0], event_negative[0]]), np.concatenate([event_positive[1], event_negative[1]])]

    if (len(event_positive[0]) < 1 or len(event_negative[0]) < 1 or len(event_avg[0]) < contact_event_threshold):
        centre_avg = None
        centre_positive = None
        centre_negative = None
    else:
        centre_positive = np.array([np.mean(event_positive[0]), np.mean(event_positive[1])])
        centre_negative = np.array([np.mean(event_negative[0]), np.mean(event_negative[1])])
        centre_avg = np.array([np.mean(event_avg[0]), np.mean(event_avg[1])])

    return centre_avg, centre_positive, centre_negative