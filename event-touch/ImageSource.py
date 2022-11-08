from abc import ABC, abstractmethod
from digit_interface import Digit
import cv2
import numpy as np

class ImageSource(ABC):

    @abstractmethod
    def get_frame(self):
        pass

    @abstractmethod
    def disconnect(self):
        pass

class DigitSensor(ImageSource):
    """
    Digit sensor class
    """
    def __init__(self):
        self.frame_count = 0
        self.serialno = 'D20431'
        self.name = 'DIGIT'
        self.resolution = 'QVGA'
        self.framerate = 60
        self.digit = None

    def __enter__(self):
        # Connect to Digit sensor
        self.digit = Digit(self.serialno, self.name)
        self.digit.connect()

        # Configure sensor resolution
        resolution = Digit.STREAMS[self.resolution]
        self.digit.set_resolution(resolution)

        # Configure sensor framerate
        framerate = Digit.STREAMS[self.resolution]["fps"]["{}fps".format(self.framerate)]
        self.digit.set_fps(framerate)

        return self
    
    def get_frame(self, transpose: bool = False):
        self.frame_count += 1
        return self.digit.get_frame(transpose)

    def disconnect(self):
        # Disconnect Digit sensor
        self.digit.disconnect()
    
    def __exit__(self, exception_type, exception_value, traceback):
        # Disconnect Digit sensor
        self.disconnect()
    
class Dataset(ImageSource):
    """
    Digit dataset class
    """
    def __init__(self):
        self.frame_count = 0
        self.directory = './dataset/'
        self.filename = 'arc_img_'
        return
    
    def __enter__(self):
        return self
    
    def get_frame(self, transpose: bool = False) -> np.ndarray:
        """
        Returns a single image frame for the device

        Parameters
        ----------
        `transpose`     - Show direct output from the image sensor, WxH instead of HxW\n
        """
        try:
            frame = cv2.imread('{}img/{}{}.png'.format(self.directory, self.filename, str(self.frame_count).zfill(3)))
        except:
            self.__exit__()

        self.frame_count += 1
        
        if transpose:
            frame = cv2.transpose(frame, frame)
            frame = cv2.flip(frame, 0)
        return frame
    
    def disconnect(self):
        return
    
    def __exit__(self, exception_type, exception_value, traceback):
        # Disconnect Digit sensor
        self.disconnect()