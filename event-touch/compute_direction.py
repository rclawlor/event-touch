import numpy as np

def compute_direction(centre_positive: np.array, centre_negative: np.array):
    """
    DESCRIPTION

    Parameters
    ----------
        `centre_positive`   - the centre of current positive events\n
        `centre_negative`   - the centre of current negative events\n

    Returns
    -------
        `direction`         - the angle between centre_positive and centre_negative
    """
    diff = centre_positive - centre_negative
    if diff[0]==0:
        direction = np.sign(diff[1])*np.pi/2
    else:
        direction = np.arctan(diff[1] / diff[0])

    return direction