import cv2
import numpy as np
import pickle

class DisplayImage(object):
    """
    Class responsible for the conversion of event frames into meaningful images. Also provides convenient functions
    for modifying and displaying images.
    """

    def __init__(self):
        self.window_name = 'DIGIT'
        return

    def __enter__(self):
        return self
    
    def draw_arrow(
            self, 
            image: np.array, 
            start: np.array, 
            end: np.array, 
            color: tuple = (255,255,255), 
            thickness: int = 2) -> np.array:
        """
        Draws an arrow on the input image between the specified coordinates

        Parameters
        ----------
            `image`     - the image on which to draw an arrow\n
            `start`     - start coordinate\n
            `end`       - end coordinate\n
            `color`     - arrow color as RGB tuple\n
            `thickness` - arrow thickness in pixels\n

        Returns
        -------
            `event_color`       - the colored event frame in BGR colorspace
        """

        if (start==None).any() or (end==None).any():
            return image
        else:
            s = np.flip(start)
            e = np.flip(end)
            img_arrow = cv2.arrowedLine(image, tuple(np.rint(s).astype(int)), tuple(np.rint(e).astype(int)), color, thickness)
            return img_arrow
        
    def draw_circle(
            self, 
            image: np.array, 
            center: np.array, 
            radius: float, 
            color: tuple = (255,255,255), 
            thickness: int = 2) -> np.array:
        """
        Draws a circle on the input image

        Parameters
        ----------
            `image`     - the image on which to draw an arrow\n
            `center`    - center coordinate\n
            `radius`    - circle radius\n
            `color`     - circle color as RGB tuple\n
            `thickness` - circle thickness in pixels\n

        Returns
        -------
            `event_color`       - the colored event frame in BGR colorspace
        """
        if center[0] < 0:
            center[0] = 0
        elif center[0] > image.shape[0]:
            center[0] = image.shape[0]-1
        if center[1] < 0:
            center[1] = 0
        elif center[1] > image.shape[1]:
            center[1] = image.shape[1]-1
        try:
            img_circle = cv2.circle(image, tuple(np.rint(np.flip(center)).astype(int)), np.rint(np.max(radius,0)).astype(int), color, thickness)
        except:
            img_circle = image
        return img_circle
    
    def map_event_to_color(
            self, 
            event_frame: np.array, 
            positive_color: tuple, 
            negative_color: tuple, 
            background_color: tuple = (0,0,0)) -> np.array:
        """
        Maps an event array to a BGR image.

        Parameters
        ----------
            `event_frame`       - the current event frame\n
            `positive_color`    - the desired BGR value for positive events\n
            `negative_color`    - the desired BGR value for negative events\n
            `background_color`  - the desired BGR value for the background\n

        Returns
        -------
            `event_color`       - the colored event frame in BGR colorspace
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

        event_color[index_positive+tuple(np.array([np.full(index_positive[0].shape[0], fill_value=B)]))] = positive_color[B]
        event_color[index_positive+tuple(np.array([np.full(index_positive[0].shape[0], fill_value=G)]))] = positive_color[G]
        event_color[index_positive+tuple(np.array([np.full(index_positive[0].shape[0], fill_value=R)]))] = positive_color[R]

        event_color[index_negative+tuple(np.array([np.full(index_negative[0].shape[0], fill_value=B)]))] = negative_color[B]
        event_color[index_negative+tuple(np.array([np.full(index_negative[0].shape[0], fill_value=G)]))] = negative_color[G]
        event_color[index_negative+tuple(np.array([np.full(index_negative[0].shape[0], fill_value=R)]))] = negative_color[R]

        return event_color
    
    def map_buffer_to_image(
            self, 
            event_buffer) -> np.array:
        """
        Maps an event buffer to one image

        Parameters
        ----------
            `event_buffer`  - the current event frame\n

        Returns
        -------
            `image`         - an array of +1, -1 and 0 corresponding to events
        """

        image = event_buffer[:,:,0]
        for frame in range(1,event_buffer.shape[2]):
            image = np.concatenate([image, event_buffer[:,:,frame]], axis=1)
        
        return image
    
    def show_image(
            self, 
            images: list,
            resize: str = 'pad',
            window_scale: float = 1.0,
            image_frame: bool = False,
            n_rows: int = 1):
        """
        Display list of images in a single window using OpenCV

        Parameters
        ----------
            `images`       - list of np.array instances to be displayed.\n
            `resize`       - either 'pad' to add black border or 'repeat' to scale image to correct resolution using nearest neighour.\n
            `window_scale` - global scaling of displayed image.\n
            `image_frame`  - add black border to displayed images.\n
            `n_rows`       - number of rows the images are displayed on: len(images)%n_rows must equal 0.
        """

        img_pad = []
        n_images = len(images)
        images_per_row = int(n_images/n_rows)

        if (n_images%n_rows != 0):
            raise ValueError("The number of images dispayed must be divisible by n_rows")
        
        img_rows = []
        for row in range(n_rows):
            img_pad = []
            for i in range(row*images_per_row, (1+row)*images_per_row):
                image = images[i]
                if (images[i].shape[0] < images[0].shape[0]) or (images[i].shape[1] < images[0].shape[1]):
                    if resize=='pad':
                        image = np.pad(
                            images[i], 
                            pad_width=[
                                (int((images[0].shape[0]-images[i].shape[0])/2), int((images[0].shape[0]-images[i].shape[0])/2)),
                                (int((images[0].shape[1]-images[i].shape[1])/2), int((images[0].shape[1]-images[i].shape[1])/2)),
                                (0, 0)], 
                            mode='constant')
                    elif resize=='repeat':
                        img_tmp = np.repeat(images[i], repeats=int(images[0].shape[0]/images[i].shape[0]), axis=0)
                        image = np.repeat(img_tmp, repeats=int(images[0].shape[0]/images[i].shape[0]), axis=1)
                    else:
                        raise ValueError('Invalid resize method')
                if image_frame==True:
                    image = np.pad(image, [(1,1),(1,1),(0,0)], mode='constant')
                img_pad.append(image)
            img_rows.append(np.concatenate(img_pad, axis=1))
        if n_rows>1:
            img = np.concatenate(img_rows, axis=0)
        else:
            img = img_rows[0]
        img = cv2.resize(img, (int(window_scale*img.shape[1]),int(window_scale*img.shape[0])))
        cv2.imshow(self.window_name, img)

        return

    def save_array(
            self, 
            images: list, 
            directory: str, 
            filename: str, 
            frame_number: int, 
            frame_skip: int = 4, 
            visualise: bool = False):
        """
        Saves current frame using pickle

        Parameters
        ----------
            `images`       - list of np.array instances to be displayed\n
            `directory`    - the directory to save images\n
            `filename`     - the filename of saved images\n
            `frame_number` - the current frame count\n
            `visualise`    - display the current image using cv2
        """

        img = np.concatenate(images, axis=1)
        if frame_number%frame_skip==0:
            if visualise:
                cv2.imshow(self.window_name, img)
            # cv2.imwrite('{}{}_{}.png'.format(directory,filename,str(frame_number).zfill(5)), img)
            with open('{}{}_{}.pickle'.format(directory,filename,str(frame_number).zfill(5)), 'wb') as handle:
                pickle.dump(img, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return
    
    def __exit__(
            self, 
            exception_type, 
            exception_value, 
            traceback):
            
        # Disconnect Digit sensor
        cv2.destroyAllWindows()
        return