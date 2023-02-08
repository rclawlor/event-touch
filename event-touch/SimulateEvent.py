import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

class SimulateEvent(object):
    """
    Simulate an event camera using traditional RGB images
    """
    def __init__(
            self, 
            colorspace: str, 
            event_threshold: np.array, 
            buffer_length: int = None, 
            buffer_method: str = 'cumulative'):

        self.colorspace = colorspace
        self.pixel_intensity_memory = None
        self.temp = None

        self.event_threshold = event_threshold
        self.event_frame = None

        self.buffer_length = buffer_length
        self.buffer_method = buffer_method
        self.event_buffer = None

        self.colorspace_code = {'GRAY': 6, 'BGR': None, 'YCrCb': 36, 'HSV': 40, 'HLS': 52, 'YUV': 82}

        return

    def __enter__(self):
        return self
    
    def __exit__(
            self, 
            exception_type, 
            exception_value, 
            traceback):
        return
    
    def initialise_pixel_memory(
            self, 
            frame: np.array):
        """
        Initialises the pixel intensity memory with the input Digit frame, and fills event buffer with zeros.

        Parameters
        ----------
            `frame`         - current image frame in chosen colorspace\n
        """
        if self.colorspace=='BGR':
            self.pixel_intensity_memory = frame
        else:
            self.pixel_intensity_memory = cv2.cvtColor(frame, self.colorspace_code[self.colorspace])
        
        self.event_buffer = np.zeros([frame.shape[0], frame.shape[1], self.buffer_length])
        self.temp = frame

        return
    
    def simulate_event(
            self, 
            digit_frame: np.array) -> np.array:
        """
        Calculates the generated events using the specified colorspace.

        Parameters
        ----------
            `digit_frame`   - current image frame in chosen colorspace\n

        Returns
        -------
            `event_frame`   - an array containing +1 (positive event), -1 (negative event) and 0 (no event)\n
        """
        self.event_frame = np.zeros([digit_frame.shape[0], digit_frame.shape[1]], dtype=np.float32)

        if self.colorspace_code[self.colorspace] == None:
            cvt_frame = digit_frame
            diff = (np.log2(digit_frame).astype('float32') - np.log2(self.pixel_intensity_memory).astype('float32'))
        else:
            cvt_frame = cv2.cvtColor(digit_frame, int(self.colorspace_code[self.colorspace]))
            diff = (np.log2(cvt_frame).astype('float32') - np.log2(self.pixel_intensity_memory).astype('float32'))
        # self.diff = diff
        if len(diff.shape) == 3:
            index_positive = np.nonzero(
                (diff[:,:,0]>self.event_threshold[0]) 
                | (diff[:,:,1]>self.event_threshold[1]) 
                | (diff[:,:,2]>self.event_threshold[2]))

            index_negative = np.nonzero(
                (diff[:,:,0]<-self.event_threshold[0]) 
                | (diff[:,:,1]<-self.event_threshold[1]) 
                | (diff[:,:,2]<-self.event_threshold[2]))
        else:
            index_positive = np.nonzero(diff>self.event_threshold[0])
            index_negative = np.nonzero(diff<-self.event_threshold[0])
        
        self.event_frame[tuple([index_positive[0], index_positive[1]])] = 1
        self.event_frame[tuple([index_negative[0], index_negative[1]])] = -1

        self.pixel_intensity_memory[index_positive] = cvt_frame[index_positive]
        self.pixel_intensity_memory[index_negative] = cvt_frame[index_negative]

        # self.temp[index_positive] = digit_frame[index_positive]

        return self.event_frame
    
    def simulate_event_buffer(
            self, 
            digit_frame: np.array,
            filter: bool = False,
            structure: np.array = None) -> np.array:
        """
        Computes and updates the temporal buffer using the most recent event frame and the current event buffer.

        Parameters
        ----------
            `digit_frame`   - current image frame in chosen colorspace\n
            `filter`        - bool, dictates whether event neighbour filter is used\n
            `structure`     - neighbours to be checked if using filter\n

        Returns
        -------
            `event_buffer`  - updated buffer
        """

        self.simulate_event(digit_frame)
        if filter==True:
            self.event_frame = self.filter_events_by_neighbours(structure)

        if self.buffer_method=='discrete':
            for element in range(1, self.buffer_length):
                self.event_buffer[:, :, self.buffer_length-element] = self.event_buffer[:, :, self.buffer_length-(element+1)]
        elif self.buffer_method=='or':
            for element in range(1, self.buffer_length):
                self.event_buffer[:, :, self.buffer_length-element] = np.logical_or(self.event_buffer[:, :, self.buffer_length-(element+1)], self.event_frame)
        elif self.buffer_method=='cumulative':
            for element in range(1, self.buffer_length):
                self.event_buffer[:, :, self.buffer_length-element] = np.rint(self.event_buffer[:, :, self.buffer_length-(element+1)] + self.event_frame)
            self.event_buffer[self.event_buffer > 1] = 1
            self.event_buffer[self.event_buffer < -1] = -1
        else:
            raise ValueError('Invalid method input : choose from `discrete`, `or` and `cumulative`')
        
        self.event_buffer[:,:,0] = self.event_frame

        return self.event_buffer
    
    @staticmethod
    def new_subsample_event(event_frame, threshold: int):
        
        if threshold==1:
            sub_frame = np.pad(np.abs(event_frame), [(0,1), (1,0)], 'constant').astype('uint8')
            sub_frame[0:-1,1:] |= sub_frame[0:-1,0:-1]
            sub_frame[0:-1,1:] |= sub_frame[1:,1:]
        elif threshold==4:
            sub_frame = np.pad(np.abs(event_frame), [(0,1), (1,0)], 'constant').astype('uint8')
            sub_frame[0:-1,1:] &= sub_frame[0:-1,0:-1]
            sub_frame[0:-1,1:] &= sub_frame[1:,1:]
        else:
            sub_frame_a = np.pad(np.abs(event_frame), [(0,1), (1,0)], 'constant').astype('uint8')
            sub_frame_a[0:-1,1:] &= sub_frame_a[0:-1,0:-1]
            sub_frame_a[0:-1,1:] |= sub_frame_a[1:,1:]

            sub_frame_b = np.pad(np.abs(event_frame), [(0,1), (1,0)], 'constant').astype('uint8')
            sub_frame_b[0:-1,1:] |= sub_frame_b[0:-1,0:-1]
            sub_frame_b[0:-1,1:] &= sub_frame_b[1:,1:]
            if threshold==2:
                sub_frame = sub_frame_a | sub_frame_b
            elif threshold==3:
                sub_frame = sub_frame_a & sub_frame_b
            else:
                raise ValueError("Invalid threshold: select from (1,2,3,4)")

        return sub_frame[1:,0:-1][::2,::2]

    def event_noise_filter(
            self, 
            event_frame: np.array,
            sigma: int = 5,
            threshold: float = 0.2):

        """
        Subsamples the current event frame into lower resolutions.

        Parameters
        ----------
            `digit_frame`   - current image frame in chosen colorspace\n
            `iterations`    - the number of subsampling iterations\n

        Returns
        -------
            `sample_space`  - list of event frames in decreasing resolutions
        """
        abs_event_frame = np.abs(event_frame)
        filtered_frame = np.zeros_like(abs_event_frame)
        tmp = gaussian_filter(abs_event_frame, sigma)
        tmp[tmp < threshold] = 0
        filtered_frame[np.nonzero(tmp)] = 1

        return event_frame*filtered_frame
    
    def filter_events_by_neighbours(self, structure: np.array):

        row, column = self.event_frame.shape
        y_offset, x_offset = structure.shape
        y_offset -= 1
        x_offset -= 1
        structure[int(x_offset/2), int(y_offset/2)] = 0

        # Create padded array to share memory instead of using np.roll()
        event_frame_padded = np.pad(np.abs(self.event_frame), [(int(y_offset/2),), (int(x_offset/2),)], 'constant').astype('uint8')
        # Find elements to pad
        y, x = np.nonzero(structure!=0)
        # Create 'rolled' arrays
        event_frame_neighbours = np.copy(
                                event_frame_padded[x_offset-x[0]:x_offset-x[0]+row, y_offset-y[0]:y_offset-y[0]+column]
                                & event_frame_padded[int(x_offset/2):-int(x_offset/2),int(y_offset/2):-int(y_offset/2)])

        for i in range(1, x.shape[0]):
            neighbour_frame = (
                            event_frame_padded[x_offset-x[i]:x_offset-x[i]+row, y_offset-y[i]:y_offset-y[i]+column]
                            & event_frame_padded[int(x_offset/2):-int(x_offset/2),int(y_offset/2):-int(y_offset/2)])
            event_frame_neighbours |= neighbour_frame

        return event_frame_neighbours.astype(self.event_frame.dtype)*self.event_frame
    
    def filter_events_by_neighbours_polarity(self, structure: np.array):

        row, column = self.event_frame.shape
        y_offset, x_offset = structure.shape
        y_offset -= 1
        x_offset -= 1
        structure[int(x_offset/2), int(y_offset/2)] = 0

        event_p = np.copy(self.event_frame)
        event_p[event_p == -1] = 0
        event_n = np.abs(self.event_frame) - event_p

        # Create padded array to share memory instead of using np.roll()
        event_p_padded = np.pad(event_p, [(int(y_offset/2),), (int(x_offset/2),)], 'constant').astype('uint8')
        event_n_padded = np.pad(event_n, [(int(y_offset/2),), (int(x_offset/2),)], 'constant').astype('uint8')
        # Find elements to pad
        y, x = np.nonzero(structure!=0)
        # Create 'rolled' arrays
        event_p_neighbours = np.copy(
                                event_p_padded[x_offset-x[0]:x_offset-x[0]+row, y_offset-y[0]:y_offset-y[0]+column]
                                & event_p_padded[int(x_offset/2):-int(x_offset/2),int(y_offset/2):-int(y_offset/2)])
        event_n_neighbours = np.copy(
                                event_n_padded[x_offset-x[0]:x_offset-x[0]+row, y_offset-y[0]:y_offset-y[0]+column]
                                & event_n_padded[int(x_offset/2):-int(x_offset/2),int(y_offset/2):-int(y_offset/2)])

        for i in range(1, x.shape[0]):
            neighbour_p = (
                            event_p_padded[x_offset-x[i]:x_offset-x[i]+row, y_offset-y[i]:y_offset-y[i]+column]
                            & event_p_padded[int(x_offset/2):-int(x_offset/2),int(y_offset/2):-int(y_offset/2)])
            neighbour_n = (
                            event_n_padded[x_offset-x[i]:x_offset-x[i]+row, y_offset-y[i]:y_offset-y[i]+column]
                            & event_n_padded[int(x_offset/2):-int(x_offset/2),int(y_offset/2):-int(y_offset/2)])
            event_p_neighbours |= neighbour_p
            event_n_neighbours |= neighbour_n
        # .astype(event_frame.dtype)
        return (event_p_neighbours.astype('float32') - event_n_neighbours.astype('float32')).astype(self.event_frame.dtype)

    def filter_events_by_neighbour_number(self, structure: np.array, threshold: int = 1):

        row, column = self.event_frame.shape
        y_offset, x_offset = structure.shape
        y_offset -= 1
        x_offset -= 1
        structure[int(x_offset/2), int(y_offset/2)] = 0

        # Create padded array to share memory instead of using np.roll()
        event_frame_padded = np.pad(np.abs(self.event_frame), [(int(y_offset/2),), (int(x_offset/2),)], 'constant').astype('uint8')
        # Find elements to pad
        y, x = np.nonzero(structure!=0)
        # Create 'rolled' arrays
        event_frame_neighbour_number = np.copy(
                                event_frame_padded[x_offset-x[0]:x_offset-x[0]+row, y_offset-y[0]:y_offset-y[0]+column]
                                & event_frame_padded[int(x_offset/2):-int(x_offset/2),int(y_offset/2):-int(y_offset/2)])

        for i in range(1, x.shape[0]):
            neighbour_frame = (
                            event_frame_padded[x_offset-x[i]:x_offset-x[i]+row, y_offset-y[i]:y_offset-y[i]+column]
                            & event_frame_padded[int(x_offset/2):-int(x_offset/2),int(y_offset/2):-int(y_offset/2)])
            event_frame_neighbour_number += neighbour_frame
        # .astype(event_frame.dtype)

        return (event_frame_neighbour_number>threshold).astype(self.event_frame.dtype)*self.event_frame
    
    @staticmethod
    def event_neighbour_number(event_frame: np.array, structure: np.array):

        row, column = event_frame.shape
        y_offset, x_offset = structure.shape
        y_offset -= 1
        x_offset -= 1
        structure[int(x_offset/2), int(y_offset/2)] = 0

        # Create padded array to share memory instead of using np.roll()
        event_frame_padded = np.pad(np.abs(event_frame), [(int(y_offset/2),), (int(x_offset/2),)], 'constant').astype('uint8')
        # Find elements to pad
        y, x = np.nonzero(structure!=0)
        # Create 'rolled' arrays
        event_frame_neighbours = np.copy(
                                event_frame_padded[x_offset-x[0]:x_offset-x[0]+row, y_offset-y[0]:y_offset-y[0]+column]
                                & event_frame_padded[int(x_offset/2):-int(x_offset/2),int(y_offset/2):-int(y_offset/2)]).astype('float32')

        for i in range(1, x.shape[0]):
            neighbour_frame = (
                            event_frame_padded[x_offset-x[i]:x_offset-x[i]+row, y_offset-y[i]:y_offset-y[i]+column]
                            & event_frame_padded[int(x_offset/2):-int(x_offset/2),int(y_offset/2):-int(y_offset/2)])
            event_frame_neighbours += neighbour_frame

        return event_frame_neighbours

    def event_feature_similarity(self, structure: np.array):

        row, column = self.event_frame.shape
        y_offset, x_offset = structure.shape
        y_offset -= 1
        x_offset -= 1
        structure[int(x_offset/2), int(y_offset/2)] = 0

        # Create padded array to share memory instead of using np.roll()
        event_frame_padded = np.pad(np.abs(self.event_frame), [(int(y_offset/2),), (int(x_offset/2),)], 'constant').astype('uint8')
        # Find elements to pad
        yo, xo = np.nonzero(structure!=0)
        yz, xz = np.nonzero(structure==0)
        # Create 'rolled' arrays
        event_feature_similarity = np.copy(
                                event_frame_padded[x_offset-xo[0]:x_offset-xo[0]+row, y_offset-yo[0]:y_offset-yo[0]+column]
                                & event_frame_padded[int(x_offset/2):-int(x_offset/2),int(y_offset/2):-int(y_offset/2)])

        for i in range(1, xo.shape[0]):
            neighbour_frame = (
                            event_frame_padded[x_offset-xo[i]:x_offset-xo[i]+row, y_offset-yo[i]:y_offset-yo[i]+column]
                            & event_frame_padded[int(x_offset/2):-int(x_offset/2),int(y_offset/2):-int(y_offset/2)])
            event_feature_similarity += neighbour_frame

        for i in range(1, xz.shape[0]):
            neighbour_frame = (
                            event_frame_padded[x_offset-xz[i]:x_offset-xz[i]+row, y_offset-yz[i]:y_offset-yz[i]+column]
                            & event_frame_padded[int(x_offset/2):-int(x_offset/2),int(y_offset/2):-int(y_offset/2)])
            event_feature_similarity += np.invert(neighbour_frame)

        return event_feature_similarity