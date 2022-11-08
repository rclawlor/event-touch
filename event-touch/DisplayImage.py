import cv2
import numpy as np
import pickle

class DisplayImage(object):

    def __init__(self):
        self.window_name = 'DIGIT'
        return

    def __enter__(self):
        return self
    
    def draw_arrow(self, image, start, end, color=(255,255,255), thickness=2):
        """
        Draws an arrow on the input image between the specified coordinates

        Parameters
        ----------
            `image`     - the image on which to draw an arrow\n
            `start`     - start coordinate\n
            `end`       - end coordinate\n
            `color`     - arrow color as RGB tuple\n
            `thickness` - arrow thickness in pixels
        """

        if (start==None).any() or (end==None).any():
            return image
        else:
            s = np.flip(start)
            e = np.flip(end)
            return cv2.arrowedLine(image, tuple(np.rint(s).astype(int)), tuple(np.rint(e).astype(int)), color, thickness)
    
    def map_event_to_color(self, event_frame: np.array, positive_color: tuple, negative_color: tuple, background_color: tuple = (0,0,0)):
        """
        Maps an event array (-1, 1, 0) to a BGR image 

        Parameters
        ----------
            `event_frame`   - the current event frame\n
            `positive_color`- the desired BGR value for positive events\n
            `negative_color`- the desired BGR value for negative events\n

        Returns
        -------
            `event_color`   - the colored event frame
        """
        
        B = 0 
        G = 1
        R = 2
        event_shape = event_frame.shape
        index_positive = np.nonzero(event_frame == 1)
        index_negative = np.nonzero(event_frame == -1)
        event_color = cv2.merge([
            np.full([event_shape[0], event_shape[1]], background_color[0], dtype=np.uint8),
            np.full([event_shape[0], event_shape[1]], background_color[1], dtype=np.uint8),
            np.full([event_shape[0], event_shape[1]], background_color[2], dtype=np.uint8)])

        # event_color = np.zeros([event_shape[0], event_shape[1], 3], dtype=np.uint8)

        event_color[index_positive+tuple(np.array([np.full(index_positive[0].shape[0], fill_value=B)]))] = positive_color[B]
        event_color[index_positive+tuple(np.array([np.full(index_positive[0].shape[0], fill_value=G)]))] = positive_color[G]
        event_color[index_positive+tuple(np.array([np.full(index_positive[0].shape[0], fill_value=R)]))] = positive_color[R]

        event_color[index_negative+tuple(np.array([np.full(index_negative[0].shape[0], fill_value=B)]))] = negative_color[B]
        event_color[index_negative+tuple(np.array([np.full(index_negative[0].shape[0], fill_value=G)]))] = negative_color[G]
        event_color[index_negative+tuple(np.array([np.full(index_negative[0].shape[0], fill_value=R)]))] = negative_color[R]

        return event_color
    
    def map_buffer_to_image(self, event_buffer):
        image = event_buffer[:,:,0]
        for frame in range(1,event_buffer.shape[2]):
            image = np.concatenate([image, event_buffer[:,:,frame]], axis=1)
        
        return image

    def show_image(self, images):
        img_pad = []
        for image in images:
            if (image.shape[0] < images[0].shape[0]) or (image.shape[1] < images[0].shape[1]):
                print('Padding...')
                print(images[0].shape[0]-image.shape[0], images[0].shape[1]-image.shape[1])
                print(image.shape)
                image = np.pad(
                    image, 
                    pad_width=[
                        (int((images[0].shape[0]-image.shape[0])/2), int((images[0].shape[0]-image.shape[0])/2)),
                        (int((images[0].shape[1]-image.shape[1])/2), int((images[0].shape[1]-image.shape[1])/2)),
                        (0, 0)], 
                    mode='constant')
                print(image.shape)
            img_pad.append(image)
        img = np.concatenate(img_pad, axis=1)
        cv2.imshow(self.window_name, img)
        return

    def save_array(self, images, directory, filename, frame_number, frame_skip: int = 4, visualise: bool = False):
        img = np.concatenate(images, axis=1)
        if frame_number%frame_skip==0:
            if visualise:
                cv2.imshow(self.window_name, img)
            # cv2.imwrite('{}{}_{}.png'.format(directory,filename,str(frame_number).zfill(5)), img)
            with open('{}{}_{}.pickle'.format(directory,filename,str(frame_number).zfill(5)), 'wb') as handle:
                pickle.dump(img, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return
    
    def __exit__(self, exception_type, exception_value, traceback):
        # Disconnect Digit sensor
        cv2.destroyAllWindows()
        return