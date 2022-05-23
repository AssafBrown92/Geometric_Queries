import numpy as np

class Miniball(object):

    def __init__(self, points):
        self.d = points[0].shape[0]
        self.n = len(points)
        self.points = points

        #Setting the initial center
        self.current_c = np.zeros(self.d)

        self.pivot_mb(self.n)

    def pivot_mb(self):

