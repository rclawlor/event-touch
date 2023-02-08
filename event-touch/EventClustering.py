import numpy as np
import cv2
import faiss
from .ObjectContact import ObjectContact
from scipy.optimize import linear_sum_assignment

class EventClustering(object):

    def __init__(self):
        self._contact = ObjectContact()
        self._cluster_centroids = None
        self._past_cluster_centroids = None
        self._cluster_1 = None
        self._cluster_2 = None
        return

    def __enter__(self):
        return self
    
    def __exit__(
            self, 
            exception_type, 
            exception_value, 
            traceback):
        return
    
    def compute_event_neighbours(self, event_frame):
        
        n1 = event_frame.astype('uint8') & np.roll(event_frame, shift=1, axis=0).astype('uint8')
        n2 = event_frame.astype('uint8') & np.roll(event_frame, shift=2, axis=0).astype('uint8')

        north_event_neighbours = n1 | n2

        w1 = event_frame.astype('uint8') & np.roll(event_frame, shift=1, axis=1).astype('uint8')
        w2 = event_frame.astype('uint8') & np.roll(event_frame, shift=2, axis=1).astype('uint8')

        west_event_neighbours = w1 | w2

        return north_event_neighbours, west_event_neighbours
    
    def dilate_event_frame(self, event_frame):

        row, column = event_frame.shape

        # Create padded array to share memory instead of using np.roll()
        event_frame_padded = np.pad(np.abs(event_frame), ((2,2), (2,2)), 'constant').astype('uint8')
        # Create 'rolled' arrays
        n1 = event_frame_padded[3:3+row, 2:2+column]
        n2 = event_frame_padded[4:4+row, 2:2+column]
        w1 = event_frame_padded[2:2+row, 3:3+column]
        w2 = event_frame_padded[2:2+row, 4:4+column]
        s1 = event_frame_padded[1:1+row, 2:2+column]
        s2 = event_frame_padded[0:0+row, 2:2+column]
        e1 = event_frame_padded[2:2+row, 1:1+column]
        e2 = event_frame_padded[2:2+row, 0:0+column]
        ne1 = event_frame_padded[3:3+row, 1:1+column]
        se1 = event_frame_padded[1:1+row, 1:1+column]
        sw1 = event_frame_padded[1:1+row, 3:3+column]
        nw1 = event_frame_padded[3:3+row, 3:3+column]

        event_frame_dilated = (s1 | s2 | e1 | e2 | n1 | n2 | w1 | w2 | ne1 | se1 | sw1 | nw1).astype(event_frame.dtype)

        return event_frame_dilated
    
    def dilate_event_frame_test(self, event_frame: np.array, structure: np.array):

        row, column = event_frame.shape
        y_offset, x_offset = structure.shape
        y_offset -= 1
        x_offset -= 1

        # Create padded array to share memory instead of using np.roll()
        event_frame_padded = np.pad(np.abs(event_frame), [(int(y_offset/2),), (int(x_offset/2),)], 'constant').astype('uint8')
        # Find elements to pad
        y, x = np.nonzero(structure!=0)
        # Create 'rolled' arrays
        event_frame_dilated = np.copy(event_frame_padded[x_offset-x[0]:x_offset-x[0]+row, y_offset-y[0]:y_offset-y[0]+column])

        for i in range(1, x.shape[0]):
            event_frame_shifted = event_frame_padded[x_offset-x[i]:x_offset-x[i]+row, y_offset-y[i]:y_offset-y[i]+column]
            event_frame_dilated |= event_frame_shifted
        # .astype(event_frame.dtype)
        return event_frame_dilated.astype(event_frame.dtype)
    
    def dilate_event_frame_polarity(self, event_frame: np.array, structure: np.array):

        row, column = event_frame.shape
        y_offset, x_offset = structure.shape
        y_offset -= 1
        x_offset -= 1

        event_p = np.copy(event_frame)
        event_p[event_p==-1] = 0
        event_n = np.abs(event_frame - event_p)

        # Create padded array to share memory instead of using np.roll()
        event_p_padded = np.pad(event_p, [(int(y_offset/2),), (int(x_offset/2),)], 'constant').astype('uint8')
        event_n_padded = np.pad(event_n, [(int(y_offset/2),), (int(x_offset/2),)], 'constant').astype('uint8')

        # Find elements to pad
        y, x = np.nonzero(structure!=0)
        # Create 'rolled' arrays
        event_p_dilated = np.copy(event_p_padded[x_offset-x[0]:x_offset-x[0]+row, y_offset-y[0]:y_offset-y[0]+column])
        event_n_dilated = np.copy(event_n_padded[x_offset-x[0]:x_offset-x[0]+row, y_offset-y[0]:y_offset-y[0]+column])

        for i in range(1, x.shape[0]):
            event_p_shifted = event_p_padded[x_offset-x[i]:x_offset-x[i]+row, y_offset-y[i]:y_offset-y[i]+column]
            event_p_dilated |= event_p_shifted

            event_n_shifted = event_n_padded[x_offset-x[i]:x_offset-x[i]+row, y_offset-y[i]:y_offset-y[i]+column]
            event_n_dilated |= event_n_shifted

        # .astype(event_frame.dtype)
        return (event_p_dilated.astype(event_frame.dtype) - event_n_dilated.astype(event_frame.dtype)).astype(event_frame.dtype)
    
    def cluster_event_frame(self, event_frame: np.array, niter: int = 10):

        # Reduce number of connected components
        dilated_event_frame = self.dilate_event_frame(np.abs(event_frame))
        # Find and label each connected component
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated_event_frame.astype('uint8'), 8, cv2.CV_32S)
        # Delete background componenet
        stats = np.delete(stats, 0, axis=0)

        # Fit Kmeans to connected component centroids
        d = centroids.shape[1]
        kmeans = faiss.Kmeans(d, k=2, niter=niter, verbose=False)

        # Assert there are at least 2 points
        if centroids.shape[0] > 1:
            kmeans.train(centroids)
            D, I = kmeans.index.search(centroids, 1)

            cluster_frame = np.copy(labels)
            # Create event frame for each cluster
            palette = list(np.arange(0,n,1))
            key = np.concatenate([np.array([0]), 2*I.T[0]-1])
            index = np.digitize(cluster_frame.ravel(), palette, right=True)
            cluster_frame = key[index].reshape(labels.shape)
        
            event_cluster_frame = cluster_frame*np.abs(event_frame)
            event_cluster_1 = np.zeros_like(event_cluster_frame).astype('uint8')
            event_cluster_1[event_cluster_frame==1] = 1

            event_cluster_2 = np.zeros_like(event_cluster_frame).astype('uint8')
            event_cluster_2[event_cluster_frame==-1] = 1

            # event_clusters = (event_cluster_1, event_cluster_2)
            # return event_cluster_frame

            if self._cluster_1 is None and self._cluster_2 is None:
                self._cluster_1 = event_cluster_1
                self._cluster_2 = event_cluster_2
                return event_frame
            else:
                and_1 = event_cluster_1 & self._cluster_1
                and_2 = event_cluster_2 & self._cluster_1

                if np.sum(and_1)>np.sum(and_2):
                    self._cluster_1 = event_cluster_1
                    self._cluster_2 = event_cluster_2
                else:
                    self._cluster_1 = event_cluster_2
                    self._cluster_2 = event_cluster_1
                
                # return self._cluster_1 - self._cluster_2
                return self._cluster_2


            # # Compute cluster centroids
            # cluster_centroids = []
            # cluster_centroids.append(self._contact.abs_event_position_mean(event_cluster_1))
            # cluster_centroids.append(self._contact.abs_event_position_mean(event_cluster_2))

            # if cluster_centroids[0] is None or cluster_centroids[1] is None:
            #     print('Invalid centroids')
            #     return event_frame, event_frame
            # else:
            #     if self._cluster_centroids is None:
            #         self._cluster_centroids = np.zeros([2,2])
            #         self._cluster_centroids[0,:] = cluster_centroids[0]
            #         self._cluster_centroids[1,:] = cluster_centroids[1]
            #         col_ind = np.array([0,1])
            #     else:
            #         d = np.array([[np.linalg.norm(cluster_centroids[0] - self._cluster_centroids[0,:]),
            #                         np.linalg.norm(cluster_centroids[1] - self._cluster_centroids[1,:])],
            #                         [np.linalg.norm(cluster_centroids[0] - self._cluster_centroids[0,:]),
            #                         np.linalg.norm(cluster_centroids[1] - self._cluster_centroids[1,:])]])
            #         row_ind, col_ind = linear_sum_assignment(d)
            #         self._cluster_centroids[0,:] = cluster_centroids[col_ind[0]]
            #         self._cluster_centroids[1,:] = cluster_centroids[col_ind[1]]
        else:
            print('Insufficient points')
            col_ind = np.array([0,1])
            event_clusters = (event_frame, event_frame)

        # return event_clusters[col_ind[0]], event_clusters[col_ind[1]]
        return event_frame