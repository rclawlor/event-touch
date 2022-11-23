import numpy as np
import cv2
import faiss

class EventClustering(object):

    def __init__(self):
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
        kmeans.train(centroids)
        D, I = kmeans.index.search(centroids, 1)

        cluster_frame = np.copy(labels)
        for i in range(1,n):
            # print(stats[int(i),:])
            cluster_frame[cluster_frame==i] = 2*I[int(i),0] - 1
    
        event_cluster_frame = cluster_frame*np.abs(event_frame)

        return event_cluster_frame