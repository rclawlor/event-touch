import numpy as np
import cv2
import math

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
            digit_frame: np.array, 
            iterations: int) -> list:
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

        event_frame = self.simulate_event(digit_frame)
        temp = np.copy(event_frame)
        sample_space = [event_frame]

        for _ in range(iterations):

            temp += np.roll(event_frame, shift=1, axis=0)
            temp += np.roll(temp, shift=1, axis=1)
            event_sub = temp[1::2,1::2]
            event_sub[event_sub>1] = 1
            event_sub[event_sub<-1] = -1
            sample_space.append(event_sub)
            event_frame = np.copy(event_sub)
            temp = np.copy(event_frame)
        
        return sample_space