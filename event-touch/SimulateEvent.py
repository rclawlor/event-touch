import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

class SimulateEvent(object):

    def __init__(
            self, 
            colorspace: str, 
            event_threshold: np.array, 
            buffer_length: int = None, 
            buffer_method: str = 'cumulative'):

        self.colorspace = colorspace
        self.pixel_intensity_memory = None

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
    
    def intialise_pixel_memory(
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

        return self.event_frame
    
    def simulate_event_buffer(
            self, 
            digit_frame: np.array) -> np.array:
        """
        Computes and updates the temporal buffer using the most recent event frame and the current event buffer.

        Parameters
        ----------
            `digit_frame`   - current image frame in chosen colorspace\n

        Returns
        -------
            `event_buffer`  - updated buffer
        """

        self.simulate_event(digit_frame)

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
    
    def subsample_event(
            self, 
            event_frame: np.array, 
            iterations: int,
            method: str = 'signed',
            threshold: int = 1) -> list:
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

        sample_space = [event_frame]

        if method=='signed':
            temp = np.copy(event_frame)
            for _ in range(iterations):

                temp += np.roll(event_frame, shift=1, axis=0)
                temp += np.roll(temp, shift=1, axis=1)
                event_sub = temp[1::2,1::2]
                event_sub[event_sub>1] = 1
                event_sub[event_sub<-1] = -1
                sample_space.append(event_sub)
                event_frame = np.copy(event_sub)
                temp = np.copy(event_frame)
        elif method=='unsigned':
            abs_event_frame = np.abs(event_frame)
            temp = np.copy(abs_event_frame)
            for _ in range(iterations):

                temp += np.roll(abs_event_frame, shift=1, axis=0)
                temp += np.roll(temp, shift=1, axis=1)
                event_sub = temp[1::2,1::2]
                event_sub[event_sub<threshold] = 0
                event_sub[event_sub>threshold] = 1
                sample_space.append(event_sub)
                abs_event_frame = np.copy(event_sub)
                temp = np.copy(abs_event_frame)
        
        return sample_space

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
    
    def filter_events_by_neighbours(self):

        row, column = self.event_frame.shape

        # Create padded array to share memory instead of using np.roll()
        event_frame_padded = np.pad(np.abs(self.event_frame), ((2,2), (2,2)), 'constant').astype('uint8')
        # Create 'rolled' arrays and check for neighbours
        s1 = event_frame_padded[3:3+row, 2:2+column] & event_frame_padded[2:-2, 2:-2]
        s2 = event_frame_padded[4:4+row, 2:2+column] & event_frame_padded[2:-2, 2:-2]
        e1 = event_frame_padded[2:2+row, 3:3+column] & event_frame_padded[2:-2, 2:-2]
        e2 = event_frame_padded[2:2+row, 4:4+column] & event_frame_padded[2:-2, 2:-2]
        n1 = event_frame_padded[1:1+row, 2:2+column] & event_frame_padded[2:-2, 2:-2]
        n2 = event_frame_padded[0:0+row, 2:2+column] & event_frame_padded[2:-2, 2:-2]
        w1 = event_frame_padded[2:2+row, 1:1+column] & event_frame_padded[2:-2, 2:-2]
        w2 = event_frame_padded[2:2+row, 0:0+column] & event_frame_padded[2:-2, 2:-2]

        event_frame_neighbours = ((s1 | s2 | e1 | e2 | n1 | n2 | w1 | w2).astype(self.event_frame.dtype))*self.event_frame

        return event_frame_neighbours

    def xor_event_buffer(self):
        xor_event_buffer = np.zeros([self.event_frame.shape[0], self.event_frame.shape[1], self.buffer_length-1])
        for element in range(0, self.buffer_length-1):
            xor_event_buffer[:,:,element] = np.logical_xor(np.abs(self.event_buffer[:,:,element]), np.abs(self.event_buffer[:,:,element+1]))
        
        return xor_event_buffer
    
    def calculate_event_neighbours(self, size: int = 5):

        return convolve2d(np.abs(self.event_frame), np.ones([size,size]))