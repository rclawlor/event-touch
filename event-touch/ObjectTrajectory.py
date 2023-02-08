from .OnlineFilter import LowpassFilter
from .ObjectContact import ObjectContact
from collections import deque
import numpy as np
from scipy.linalg import null_space

class ObjectTrajectory(object):
    """
    Calculates location and size of contacting objects, storing their trajectories.
    """
    MEAN = 0
    POSITIVE_MEAN = 1
    NEGATIVE_MEAN = 2
    VARIANCE = 3

    def __init__(self, history_length: int = None, filter: LowpassFilter = None, method: str = 'mean'):
        """
        Initialise trajectory tracking

        Parameters
        ----------
            `history_length`- length of stored history\n
            `filter`        - LowpassFilter object\n
            `method`        - either 'mean' or 'shadow_adjusted'\n
        """

        if history_length is None:
            self.history_length = 20
        else:
            self.history_length = history_length
        self.trajectory = deque([np.array([0.,0.])], maxlen=history_length)
        self.contact_variance = deque([np.array([0.,0.])], maxlen=history_length)
        self.contact = ObjectContact()
        self.filter = filter
        self.method = method
        self.frame_shape = None
        self.std = None

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
        if self.frame_shape is None:

            self.frame_shape = event_frame.shape

            self.LED_B = np.array([self.frame_shape[0]-1,int(self.frame_shape[1]/2)])
            self.LED_G = np.array([0,0])
            self.LED_R = np.array([0,self.frame_shape[1]-1])
        
        event_mean = self.contact.abs_event_position_mean(event_frame)
        if event_mean is None:
            current_position = self.trajectory[0]
            event_variance = self.contact_variance[0]
        else:
            event_variance = self.contact.event_position_variance(event_frame, event_mean)
            if self.filter is None:
                pass
            else:
                event_mean = self.filter.filter(event_mean)

            if self.method == 'mean':
                current_position = event_mean
            elif self.method == 'shadow_adjusted':
                self.std = (
                    np.sqrt(self.contact.event_variance_about_line(event_frame, np.array([event_frame.shape[0]-1,int(event_frame.shape[1]/2)]), event_mean)),
                    np.sqrt(self.contact.event_variance_about_line(event_frame, np.array([0,0]), event_mean)),
                    np.sqrt(self.contact.event_variance_about_line(event_frame, np.array([0,event_frame.shape[1]-1]), event_mean)))
                
                DB = self.LED_B-event_mean
                DG = self.LED_G-event_mean
                DR = self.LED_R-event_mean

                event_mean += 1*self.std[0]*DB / np.linalg.norm(DB)
                event_mean += 1*self.std[1]*DG / np.linalg.norm(DG)
                event_mean += 1*self.std[2]*DR / np.linalg.norm(DR)

                current_position = event_mean
            else:
                raise ValueError('Invalid method: choose from `mean` or `shadow_adjusted`')

        self.trajectory.appendleft(current_position)
        self.contact_variance.appendleft(event_variance)

        return self.trajectory, self.contact_variance

    def estimate_contact_area(self):
        DB = self.LED_B-self.trajectory[0]
        DG = self.LED_G-self.trajectory[0]
        DR = self.LED_R-self.trajectory[0]

        DB = DB / np.linalg.norm(DB)
        DG = DG / np.linalg.norm(DG)
        DR = DR / np.linalg.norm(DR)

        center = self.trajectory[0]

        points = [
            center + self.std[0]*DB,
            center - self.std[0]*DB,
            center + self.std[1]*DG,
            center - self.std[1]*DG,
            center + self.std[2]*DR,
            center - self.std[2]*DR
        ]

        A = np.ones((6,6))
        for i, point in enumerate(points):
            A[i,0] = point[0]**2
            A[i,1] = point[0]*point[1]
            A[i,2] = point[1]**2
            A[i,3] = point[0]
            A[i,4] = point[1]
        
        a,b,c,d,e,f = tuple(null_space(A).T[0])

        num = 2*((a*e**2 - b*d*e + c*d**2)/(4*a*c - b**2) - f)
        major = np.sqrt(num / (a+c - np.sqrt((a-c)**2 + b**2)))
        minor = np.sqrt(num / (a+c + np.sqrt((a-c)**2 + b**2)))
        angle = 180*(0.5*np.arctan2(2*b,(a-c))) / np.pi

        return center, major, minor, angle

    def velocity_from_trajectory(self):
        velocity = (self.trajectory[0,:,0] - self.trajectory[0,:,2])/2


        return velocity