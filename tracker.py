"""
COMP9517: Computer Vision, 2020 Term 2
Group Project - Manits

trasker.py 
"""

from scipy.spatial import distance as dist
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict
import numpy as np

## CentroidTracker
## Adapeted from simple object tracker from pyimagesearch 
## https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/

class CentroidTracker():
    def __init__(self, maxDisappeared=10):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.trajectory = OrderedDict()
        self.maxDisappeared = maxDisappeared
        
    def register(self, centroid):
        # When registered use the next available ID to store the centroid
        ID = self.nextObjectID
        self.nextObjectID += 1
        self.objects[ID] = centroid
        self.disappeared[ID] = 0
        self.trajectory[ID] = [centroid]
    
    def deregister(self, ID):
        # Remove Objects 
        del self.objects[ID]
        del self.disappeared[ID]
        del self.trajectory[ID]
        
    def update(self, boxes):
        # input "boxes" list of bounding boxes (startX, startY, endX, endY)
        if len(boxes) == 0:
            #no detections in a frame, increase all disapeared count and deregister
            for ID in list(self.disappeared.keys()):
                self.disappeared[ID] += 1
                # Deregister objects not detected for maxDisappeared frames
                if self.disappeared[ID] > self.maxDisappeared:
                    self.deregister(ID)
            
            return self.objects, self.trajectory
        
        # initalise array of input centroids
        inputCentroids = np.zeros((len(boxes), 2), dtype="int")
        
        # Define centroids 
        for (i, (startX, startY, endX, endY)) in enumerate(boxes):
            cX = int((startX + endX)/2)
            cY = int((startY + endY)/2)
            inputCentroids[i] = (cX, cY)
        
        # If not currently tracking any objects define all inputs
        if len(self.objects) == 0:
            for x in inputCentroids:
                self.register(x)
            return self.objects, self.trajectory
        
        #Otherwise do some work
        IDs = list(self.objects.keys())
        centroids = list(self.objects.values())
        
        # Calculate the distance measure bewteen the currently tracked objects and the input
        D = dist.cdist(np.array(centroids), inputCentroids)
        
        if len(inputCentroids) == len(centroids):
            # If the input and the current object are the same number
            # match the input to the objects using an ontimum linear sum
            rows, cols = linear_sum_assignment(D)
            for (row, col) in zip(rows, cols):
                ID = IDs[row]
                self.objects[ID] = inputCentroids[col]
                self.trajectory[ID].append(inputCentroids[col])
                self.disappeared[ID] = 0
        
        else: 
            # If objects and input are different size will need to register/disapear some objects
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            usedRows = set()
            usedCols = set()
            
            # Find the closes input for each object
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols: continue
                
                ID = IDs[row]
                self.objects[ID] = inputCentroids[col]
                self.trajectory[ID].append(inputCentroids[col])
                self.disappeared[ID] = 0
                
                usedRows.add(row)
                usedCols.add(col)
            
            unusedRows = set(range(len(centroids))).difference(usedRows)
            unusedClos = set(range(len(inputCentroids))).difference(usedCols)
            
            for row in unusedRows:
                ID = IDs[row]
                self.disappeared[ID] += 1
                if self.disappeared[ID] > self.maxDisappeared:
                    self.deregister(ID)
                    
            for col in unusedClos:
                self.register(inputCentroids[col])
        
        return self.objects, self.trajectory
    
    def analyse(self, point):
        x, y = point
        
        IDs = list(self.objects.keys())
        shapes = list(self.objects.values())
        centers = [(i[0],i[1]) for i in shapes]
        
        D = [self.pointDist(p1, (x,y)) for p1 in centers]
        ID = IDs[D.index(min(D))]
        
        # Initalise metrics 
        Speed = 0
        TotalDistance = 0
        NetDistance = 0
        ConfinementRatio = 0 
        
        # If there are more than 1 record in the object trjectory analyse
        if len(self.trajectory[ID]) > 1:
            path = self.trajectory[ID]
            Speed = self.pointDist(path[-1], path[-2])
            NetDistance = self.pointDist(path[0], path[-1])
            for i in range(len(path)-1):
                TotalDistance += self.pointDist(path[i], path[i+1])
            
            if NetDistance != 0: ConfinementRatio = TotalDistance / NetDistance
        
        return Speed, TotalDistance, NetDistance, ConfinementRatio
    
    def pointDist(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return ((x1-x2)**2 + (y1-y2)**2)**0.5
    
    
# Extended Object Tracker 
class ExtTracker():
    def __init__(self, maxDisappeared=10, alpha1 = 0.01, alpha2 = 0.01, alpha3 = 0.01):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.trajectory = OrderedDict()
        self.maxDisappeared = maxDisappeared
        # alpha 1,2,3 are the weights applied to each reature for tracking, 
        # default is set low to prioritise the centroid feature
        self.a1 = alpha1
        self.a2 = alpha2
        self.a3 = alpha3
        
    def register(self, centroid):
        # When registered use the next available ID to store the centroid
        cX, cY, F1, F2, F3 = centroid
        ID = self.nextObjectID
        self.nextObjectID += 1
        
        self.objects[ID] = centroid
        self.disappeared[ID] = 0
        self.trajectory[ID] = [(cX, cY)]
    
    def deregister(self, ID):
        # Remove Objects 
        del self.objects[ID]
        del self.disappeared[ID]
        del self.trajectory[ID]
        
    def update(self, shapes):
        # input "shapes" list of shape discriptions with 
        # (boundingBoxStartX, boundingBoxStartY, boundingBoxEndX, boundingBoxEndY,
        #  Feature1, Feature2, Feature3)
        if len(shapes) == 0:
            #no detections in a frame, increase all disapeared count and deregister
            for ID in list(self.disappeared.keys()):
                self.disappeared[ID] += 1
                # Deregister objects not detected for maxDisappeared frames
                if self.disappeared[ID] > self.maxDisappeared:
                    self.deregister(ID)
            
            return self.objects, self.trajectory
        
        # initalise array of input centroids
        inputs = np.zeros((len(shapes), 5), dtype="int")
        
        # Define inputs 
        for (i, (startX, startY, endX, endY, F1, F2, F3)) in enumerate(shapes):
            cX = int((startX + endX)/2)
            cY = int((startY + endY)/2)
            inputs[i] = (cX, cY, F1*self.a1, F2*self.a2, F3*self.a3)
        
        # If not currently tracking any objects define all inputs
        if len(self.objects) == 0:
            for x in inputs:
                self.register(x)
            return self.objects, self.trajectory
        
        #Otherwise do some work
        IDs = list(self.objects.keys())
        shapes = list(self.objects.values())
        
        # Calculate the distance measure bewteen the currently tracked objects and the input
        D = dist.cdist(np.array(shapes), inputs)
        
        if len(inputs) == len(shapes):
            # If the input and the current object are the same number
            # match the input to the objects using an ontimum linear sum
            rows, cols = linear_sum_assignment(D)
            for (row, col) in zip(rows, cols):
                ID = IDs[row]
                self.objects[ID] = inputs[col]
                self.trajectory[ID].append((self.objects[ID][0], self.objects[ID][1]))
                self.disappeared[ID] = 0
        
        else: 
            # If objects and input are different size will need to register/disapear some objects
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            usedRows = set()
            usedCols = set()
            
            # Find the closes input for each object
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols: continue
                
                ID = IDs[row]
                self.objects[ID] = inputs[col]
                self.trajectory[ID].append((self.objects[ID][0], self.objects[ID][1]))
                self.disappeared[ID] = 0
                
                usedRows.add(row)
                usedCols.add(col)
            
            unusedRows = set(range(len(shapes))).difference(usedRows)
            unusedClos = set(range(len(inputs))).difference(usedCols)
            
            for row in unusedRows:
                ID = IDs[row]
                self.disappeared[ID] += 1
                if self.disappeared[ID] > self.maxDisappeared:
                    self.deregister(ID)
                    
            for col in unusedClos:
                self.register(inputs[col])
        
        return self.objects, self.trajectory

    def analyse(self, point):
        x, y = point
        
        IDs = list(self.objects.keys())
        shapes = list(self.objects.values())
        centers = [(i[0],i[1]) for i in shapes]
        
        D = [self.pointDist(p1, (x,y)) for p1 in centers]
        ID = IDs[D.index(min(D))]
        
        # Initalise metrics 
        Speed = 0
        TotalDistance = 0
        NetDistance = 0
        ConfinementRatio = 0 
        
        # If there are more than 1 record in the object trjectory analyse
        if len(self.trajectory[ID]) > 1:
            path = self.trajectory[ID]
            Speed = self.pointDist(path[-1], path[-2])
            NetDistance = self.pointDist(path[0], path[-1])
            for i in range(len(path)-1):
                TotalDistance += self.pointDist(path[i], path[i+1])
            
            if NetDistance != 0: ConfinementRatio = TotalDistance / NetDistance
        
        return Speed, TotalDistance, NetDistance, ConfinementRatio
    
    def pointDist(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return ((x1-x2)**2 + (y1-y2)**2)**0.5
    