from abc import ABC, abstractmethod
from digit_interface import Digit
import cv2
import numpy as np
import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
import pybullet as p
import pybulletX as px
import tacto
import os 

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
    def __init__(self, serialno: str = 'D20431', resolution: str = 'QVGA', framerate: int = 60):
        self.frame_count = 0
        self.serialno = serialno
        self.name = 'DIGIT'
        self.resolution = resolution
        self.framerate = framerate
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
    
class TactoSimulator(ImageSource):

    def __init__(
            self, 
            config_path: str = None, 
            environment_config_name: str = 'digit', 
            sensor_config_path: str = None, 
            sensor_config_name: str = 'config_digit', 
            use_panel: bool = True,
            noise_std: np.array = np.array([0.1,0.25,1])):

        if config_path is None:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            self.config_path = dir_path+'/conf'
        else:
            self.config_path = config_path
        
        if sensor_config_path is None:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            self.sensor_config_path = dir_path+'/conf'
        else:
            self.sensor_config_path = sensor_config_path

        self.environment_config_name = environment_config_name
        self.sensor_config_name = sensor_config_name
        self.panel = use_panel

        initialize_config_dir(version_base=None, config_dir=self.config_path, job_name="digit")
        self.cfg = compose(config_name=self.environment_config_name)

        self.t = None

        self.obj = None

        self.depth = None

        self.noise_std = noise_std

        return

    def __enter__(self):

        # Initialize digits
        bg = cv2.imread(f'{self.config_path}/bg_digit_240_320.jpg')
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)

        self.digits = tacto.Sensor(
            **self.cfg.tacto,
            **{"config_path": f'{self.sensor_config_path}/{self.sensor_config_name}.yml'},
            background=bg
        )
        # Initialize World
        px.init()
        p.resetDebugVisualizerCamera(**self.cfg.pybullet_camera)

        # Create and initialize DIGIT
        digit_body = px.Body(**self.cfg.digit)
        self.digits.add_camera(digit_body.id, [-1])

        # Add object to pybullet and tacto simulator
        self.obj = px.Body(**self.cfg.object)
        self.digits.add_body(self.obj)

        # Create control panel to control the 6DoF pose of the object
        if self.panel==True:
            panel = px.gui.PoseControlPanel(self.obj, **self.cfg.object_control_panel)
            panel.start()

        # run p.stepSimulation in another thread
        self.t = px.utils.SimulationThread(real_time_factor=1.0)
        self.t.start()
        
        return self
    
    def __exit__(self, exception_type, exception_value, traceback):
        # Disconnect Digit sensor
        self.disconnect()
    
    def _add_noise(self, digit_frame):

        digit_frame_float = digit_frame.astype('float32')
        digit_frame_float[:,:,0] += np.random.normal(0, self.noise_std[0], digit_frame.shape[:-1])
        digit_frame_float[:,:,1] += np.random.normal(0, self.noise_std[1], digit_frame.shape[:-1])
        digit_frame_float[:,:,2] += np.random.normal(0, self.noise_std[2], digit_frame.shape[:-1])

        digit_frame_float[digit_frame_float<0] = 0
        digit_frame_float[digit_frame_float>255] = 255

        return digit_frame_float.astype('uint8')
        
    def get_frame(self):
        colors, self.depth = self.digits.render(noise=False)
        digit_frame = np.concatenate(colors, axis=1)
        digit_frame = self._add_noise(digit_frame)
        return digit_frame
    
    def disconnect(self):
        GlobalHydra.instance().clear()
        return
    
class Dataset(ImageSource):
    """
    Digit dataset class
    """
    def __init__(self, directory):
        self.frame_count = 0
        self.directory = directory
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
            frame = cv2.imread(self.directory+str(self.frame_count).zfill(4)+'.png')
        except:
            raise IndexError('Requested frame out of range')

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