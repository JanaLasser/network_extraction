
"""
Created on Thu Jan 22 14:36:58 2015

@author: jana
"""
from libc.math cimport sqrt
from libc.math cimport round
import numpy as np
cimport numpy as np
import scipy
from bisect import bisect_left

ctypedef np.int_t DTYPE_t 


#helper function to calculate the distance between two points on a plane
cdef distance(Cpoint p1,Cpoint p2):
    cdef double dist = sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y))
    return dist

#extension of the scipy.sparse lil_matrix which implements a function for
#column and row removal 
class lil2(scipy.sparse.lil.lil_matrix):
    def removecol(self,int j):
        if j < 0:
            j += self.shape[1]

        if j < 0 or j >= self.shape[1]:
            raise IndexError('column index out of bounds')

        rows = self.rows
        data = self.data
        cdef int i
        for i in xrange(self.shape[0]):
            pos = bisect_left(rows[i], j)
            if pos == len(rows[i]):
                continue
            elif rows[i][pos] == j:
                rows[i].pop(pos)
                data[i].pop(pos)
                if pos == len(rows[i]):
                    continue
            for pos2 in xrange(pos,len(rows[i])):
                rows[i][pos2] -= 1

        self._shape = (self._shape[0],self._shape[1]-1)

    def removerow(self,int i):
        if i < 0:
            i += self.shape[0]

        if i < 0 or i >= self.shape[0]:
            raise IndexError('row index out of bounds')

        self.rows = np.delete(self.rows,i,0)
        self.data = np.delete(self.data,i,0)
        self._shape = (self._shape[0]-1,self.shape[1]) 

'''
Custom class for a point in 2D. Has x- and y coordinates as members.
Allows access to members via getters to reduce python overhead.
Implements __richcmp__ to allow for comparisons between points. A point is
larger than another point if its x- and y coordinates are larger than the 
other point's coordinates respectively. Two points are only equal if their
coordinates are equal (not if they are the same object!)     
Also implements a __str__ method for debugging reasons which allows to
visualize a point as "(x,y)".
'''   
cdef class Cpoint:    
    cdef double x,y
    
    def __cinit__(self,float x,float y):
        self.x = x
        self.y = y     
    
    def __richcmp__(Cpoint self, Cpoint other not None, int op):
        """Cython equivalent of functools.totalordering
        Implements compare for Cpoints, check x coordinate first then y"""
        cdef int compare
        if self.x > other.x:
            compare = 1
        elif self.x < other.x:
            compare = -1
        else:
            if self.y > other.y:
                compare = 1
            elif self.y < other.y:
                compare = -1
            else:
                compare = 0
        return richcmp_helper(compare, op)
        
    def get_x(self):
        return self.x
        
    def get_y(self):
        return self.y
        
    def __str__(self):
        cdef str s = "(" + str(self.x) + "," + str(self.y) + ")"
        return s

'''        
Custom class for a segment in 2D. A segment consists of two Cpoints. Segments
are used to build triangles. A segment also stores pointers to the triangles 
it is part of to make later navigation easier. 
Implements __richcmp__ for comparisons. A segment is larger than another
segment if both its points are larger than the other segment's points.
Implements __str__ for debugging reasons which allows visualization of the
segment as "(x1,y1) -> (x2,y2)".
  
Exported methods:
   - get_triangle: returns a pointer to the first or second triangle a segment
                   is part of
   - set_triangle: exposes methods to python for setting of pointers to 
                   triangles
   - get_midpoint: returns a Cpoint object which lies in the middle of the
                   points the segment consists of
   - connects:     returns True if two segments share a point and false
                   otherwise
   - get_cp:       returns the shared point between two segments or None if
                   there is no shared point
'''
cdef class Csegment:    
    cdef Cpoint p1,p2,midpoint
    cdef CshapeTriangle triangle1,triangle2
    
    def __init__(Cpoint self,Cpoint p1, Cpoint p2):      
        #order points so alsways p1 < p2
        if p1 < p2:
            self.p1 = p1
            self.p2 = p2
        else:
            self.p1 = p2
            self.p2 = p1         
        self.midpoint = Cpoint((self.p1.x + self.p2.x)/2, (self.p1.y + self.p2.y)/2)
   
    def __str__(self):
        return "(%1.2f,%1.2f) -> (%1.2f,%1.2f)"%\
                (self.p1.x,self.p1.y,self.p2.x,self.p2.y)

    def __richcmp__(Csegment self, Csegment other not None, int op):
        """Cython equivalent of functools.totalordering
        Implements compare for Cpoints, check p1 first then p2"""
        cdef int compare
        if self.p1 > other.p1:
            compare = 1
        elif self.p1 < other.p1:
            compare = -1
        else:
            if self.p2 > other.p2:
                compare = 1
            elif self.p2 < other.p2:
                compare = -1
            else:
                compare = 0
        return richcmp_helper(compare, op)
     
    cpdef get_p1(self): return self.p1
    cpdef get_p2(self): return self.p2
    cpdef get_triangle1(self): return self.triangle1 
    cpdef set_triangle1(self,CshapeTriangle t): self.triangle1 = t           
    cpdef get_triangle2(self): return self.triangle2          
    cpdef set_triangle2(self,CshapeTriangle t): self.triangle2 = t             
    cpdef get_midpoint(self): return self.midpoint     
       
    cpdef connects(self,Csegment other):
        if (self == other): return False              
        elif((self.p1 == other.p1) or (self.p2 == other.p1) \
            or (self.p1 == other.p2) or (self.p2 == other.p2)): return True        
        else: return False
                      
    cpdef get_cp(Csegment self,Csegment seg):
        cdef Cpoint cp 
        if(self.connects(seg)):
            if(self.p1 == seg.p1 or self.p1 == seg.p2): cp = self.p1             
            else: cp = self.p2             
        else: cp = None
        return cp
 
'''
Custom class for a triangle in 2D.
Members:
    - edge_i:   Edges the triangle is constructed with
    - p_i:      Cpoints that span the triangle
    - typ:      type of the triangle (junction, normal, end or isolated)
    - centroid: "midpoint" of the triangle, point with the largest distance
                to the nearest edge of the foreground structure. It is only
                called "centroid" because of historical reasons :-)
    - radius: distance to the nearest edge of the foreground structure
    - index:    counter used to fix the triangles position in a triangle
                adjacency matrix
    - neighbor: pointers to the triangles neighbors (3 to 0 possible neighbors)
                for junction, normal, end and isolated triangles. The pointers
                point to None at triangle construction and are set later by
                the triangle's "set_type" method.
    - Cpoints h, opp_point and half_ext_edge:
                helper points for the calculation of centroid and radius
            
Exposed functions:
    
'''       
cdef class CshapeTriangle:
    cdef Csegment edge1, edge2, edge3
    cdef Cpoint p1, p2, p3    
    cdef typ 
    cdef Cpoint centroid
    cdef double radius
    cdef int index
    #might be more convenient to store them in a list but so far they are
    #three separate members
    cdef CshapeTriangle neighbor1, neighbor2, neighbor3
    cdef Cpoint h, opp_point, half_ext_edge
   
    #Constructor creates a triangle from three segments, sets the three points
    #of the triangle and the segment's pointer to the newly created triangle
    def __init__(self, Csegment edge1, Csegment edge2, Csegment edge3):
        #type cannot be determined at creation and has to be set at a later
        #point via the set_type function
        self.typ = None                     
        self.edge1 = edge1
        self.edge2 = edge2
        self.edge3 = edge3     
        #as a triangle is composed of edges, it is not immediately clear 
        #by which points it is spanned
        self.p1 = self.edge1.p1
        self.p2 = self.edge1.p2
        if(self.edge2.p1 == self.p1 or self.edge2.p1 == self.p2):
            self.p3 = self.edge2.p2
        else:
            self.p3 = self.edge2.p1                  
        #make sure that whenever a new triangle is constructed, the edges it
        #is composed of have their pointers set to the new triangle
        if edge1.triangle1 == None: edge1.triangle1 = self
        else: edge1.triangle2 = self
        if edge2.triangle1 == None: edge2.triangle1 = self
        else: edge2.triangle2 = self 
        if edge3.triangle1 == None: edge3.triangle1 = self
        else: edge3.triangle2 = self
    
    #getter and setter for the member variables
    cpdef get_radius(self): return self.radius
    cpdef get_centroid(self): return self.centroid      
    cpdef get_p1(self): return self.p1       
    cpdef get_p2(self): return self.p2       
    cpdef get_p3(self): return self.p3   
    cpdef get_index(self): return self.index
    cpdef set_index(self,index): self.index = index
    cpdef get_typ(self): return self.typ 
    cpdef get_neighbor1(self): return self.neighbor1
    cpdef get_neighbor2(self): return self.neighbor2
    cpdef get_neighbor3(self): return self.neighbor3
    
    #set_typ is a function typically called after all triangles have been
    #initially created and all segment-pointers to triangles have been set.
    #It checks how many neighbors a triangle has and therefore sets its typ
    #to junction (3 neighbors), normal (2 neighbors), end (1 neighbor) or 
    #isolated (0 neighbors) for fast type-checks later on.
    #It also sets pointers to the triangle's neighbors so we are able to locate
    #them without searching through long lists.
    #The function also sets helper points for the calculation of the
    #triangle's radius and centroid. At the and, a helper function for the
    #actual calculation of radius and centroid based on the euclidean 
    #distance map of the image is called.
    #This is done only after determination of the triange's type because 
    #methods for setting the radius and centroid differ depending on type.
    cpdef set_typ(self,np.ndarray distance_map):
        #helper variables
        cdef int neighbors = 0
        cdef Cpoint h1,h2,h3
        cdef double abs_ab1,abs_ab2,abs_ab3
        
        #set the triangels neighbors based on the pointers stored in the edges
        if self.edge1.triangle1 == self: self.neighbor1 = self.edge1.triangle2           
        else: self.neighbor1 = self.edge1.triangle1
        if self.edge2.triangle1 == self: self.neighbor2 = self.edge2.triangle2           
        else: self.neighbor2 = self.edge2.triangle1
        if self.edge3.triangle1 == self: self.neighbor3 = self.edge3.triangle2           
        else: self.neighbor3 = self.edge3.triangle1
  
        #count the number of non-None neighbors      
        if self.neighbor1 != None:
            neighbors += 1
        if self.neighbor2 != None:
            neighbors += 1
        if self.neighbor3 != None:
            neighbors += 1
            
        #junction triangle if we have 3 neighbors
        if neighbors == 3:
            self.typ = "junction"  
            
            #create helper points for calculation of triangle center and radius
            #midpoints of the edges:
            h1 = Cpoint((self.p1.x+self.p2.x)/2.0, (self.p1.y+self.p2.y)/2.0)
            h2 = Cpoint((self.p1.x+self.p3.x)/2.0, (self.p1.y+self.p3.y)/2.0)
            h3 = Cpoint((self.p2.x+self.p3.x)/2.0, (self.p2.y+self.p3.y)/2.0) 
            #distances between edge-midpoints and opposing points
            abs_ab1 = distance(h1,self.p3)        
            abs_ab2 = distance(h2,self.p2)
            abs_ab3 = distance(h3,self.p1)     
            
            #set the helper variables for the longest angle bisection line and
            #its opposing point
            if (abs_ab1 >= abs_ab2 and abs_ab1 >= abs_ab3):
                self.h = h1
                self.opp_point = self.p3
            elif(abs_ab2 >= abs_ab1 and abs_ab2 >= abs_ab3):
                self.h = h2
                self.opp_point = self.p2
            else:
                self.h = h3
                self.opp_point = self.p1    
                                   
        #normal triangle if we have two neighbors
        elif neighbors == 2:
            self.typ = "normal"
            
            #find out which of the edges is the external edge (edge shared
            #with the contour) and set its opposing point accordingly
            if self.neighbor1 == None: #-> edge1 is the external edge
                self.half_ext_edge = self.edge1.get_midpoint()
                self.opp_point = self.edge2.get_cp(self.edge3)
                self.neighbor1 = self.neighbor3
                self.neighbor3 = None                            
            elif self.neighbor2 == None:# -> edge2 is the external edge
                self.half_ext_edge = self.edge2.get_midpoint()
                self.opp_point = self.edge1.get_cp(self.edge3)
                self.neighbor2 = self.neighbor3
                self.neighbor3 = None
            else: # -> edge3 is the external edge
                self.half_ext_edge = self.edge3.get_midpoint()
                self.opp_point = self.edge1.get_cp(self.edge2)          
            
        #end triangle if we have one neighbor
        elif neighbors == 1:
            self.typ = "end"
            if self.neighbor3 != None:
                self.neighbor1 = self.neighbor3
                self.neighbor3 = None
            elif self.neighbor2 != None:
                self.neighbor1 = self.neighbor2
                self.neighbor2 = None
            else:
                pass
                    
        #isolated triangle if we have zero neighbors
        else:
            self.typ = "isolated"
            
        #calculate position of triangle centroid and radius
        return self.set_radius_and_center(distance_map)

    #helper function to calculate and set the radius and centroid of the
    #triangle. If the radius cannot be found by looking up the coordinates
    #of the centroid in the distance-map, the radius defaults to 1.0. Every
    #time this is the case, the function returns 1 instead of the usual 0
    #so we are able to track the number of times we defaulted to 1.0.
    #In general, this happens for weird geometries every once in a while (100 
    #cases for 100k triangles is completely normal and nothing to worry about).
    #If the number of defaults does not get out of hand, the impact of the 
    #default-triangle-radii will be negligible
    cdef set_radius_and_center(CshapeTriangle self, \
                                 np.ndarray[DTYPE_t,ndim=2] distance_map):
        #helper variables
        cdef double x1, y1, x2, y2, length, lamb, curr_x, curr_y
        cdef int int_x, int_y
        cdef double direction_x, direction_y, max_distance
        cdef double max_x, max_y
        #use typed indexing for fast array acces
        DTYPE = np.int
        cdef DTYPE_t curr_distance
        assert distance_map.dtype == DTYPE   
        #these initializations are the same for all triangles
        max_distance = 0
        max_x = -1
        max_y = -1
        lamb = 0.1 
        
        #for end-triangles, the triangle centroid will be set to the shared
        #point between the two external segments. This is the case because
        #the centroid later is used to represent the position of the corres-
        #ponding node in the network. As end-triangles are situated at the 
        #network's tip and we want to capture as much of the network's struc-
        #ture as possible, it is logical to set the centroid to the outermost
        #point.
        #To set the triangles radius, we are looking for the point with the
        #largest distance to the edges of the structure along a line from the 
        #middle of the triangle's only internal edge its opposing point.        
        if self.typ == "end": 
            #initialize helper-variables
            x1 = self.p3.x        
            y1 = self.p3.y
            x2 = (self.p1.x + self.p2.x)/2.0
            y2 = (self.p1.y + self.p2.y)/2.0
            direction_x = x2-x1
            direction_y = y2-y1
            length = sqrt((y2-y1)**2 + (x2-x1)**2)
            curr_x = x1
            curr_y = y1
            #search along a line for a local maximum in the distance map
            while(sqrt((curr_y - y1)**2 + (curr_x - x1)**2) < length):
                int_x = int(round(curr_x))
                int_y = int(round(curr_y))
                curr_distance = distance_map[int_y, int_x]
                if curr_distance > max_distance:
                    max_distance = curr_distance
                curr_x += lamb*direction_x
                curr_y += lamb*direction_y
            self.radius = max_distance       
            #set the center to the "tip point"
            if self.edge1.triangle2 != None:
                self.centroid = self.edge2.get_cp(self.edge3)
            elif self.edge2.triangle2 != None:
                self.centroid = self.edge1.get_cp(self.edge3)
            else:
                self.centroid = self.edge1.get_cp(self.edge2)
            #we have found a local maximum so there was no need to default to
            #1.0 so we return 0.
            return 0
        
        #For normal triangles, we will search for a local maximum in the dist-
        #ance map along the line from the triangle's single external edge to
        #its opposing point. We will set the radius to the value of the local
        #maximum and the position of the centroid to the position of the 
        #maximum.
        elif self.typ == "normal": 
            #initialize helper-variables
            x1 = self.opp_point.x        
            y1 = self.opp_point.y
            x2 = self.half_ext_edge.x
            y2 = self.half_ext_edge.y
            curr_x = x1
            curr_y = y1
            direction_x = x2-x1
            direction_y = y2-y1
            length = sqrt((y2-y1)**2 + (x2-x1)**2)
            #search along a line for a local maximum in the distance map
            while(sqrt((curr_y - y1)**2 + (curr_x - x1)**2) < length):
                int_x = int(round(curr_x))
                int_y = int(round(curr_y))
                curr_distance = distance_map[int_y, int_x]
                if curr_distance > max_distance:
                    max_distance = curr_distance
                    max_x = curr_x
                    max_y = curr_y
                curr_x += lamb*direction_x
                curr_y += lamb*direction_y
            #if we found a maximum, set radius and centroid and return 0
            if (max_y != -1 and max_x != -1):
                self.radius = max_distance
                self.centroid = Cpoint(max_x,max_y)
                return 0
            #if we didn't find a maximum, default to a radius of 1.0, set the
            #centroid to the real centroid of the triangle and return 1
            else:
                self.centroid = Cpoint( \
                            round((self.p1.x+self.p2.x+self.p3.x)*(1.0/3.0)),\
                            round((self.p1.y+self.p2.y+self.p3.y)*(1.0/3.0)))                                                      
                self.radius = 1
                return 1
         
        #For junction triangles we will search for a local maximum in the dist-
        #ance map along the longest angle-bisection line of the triangle.
        #The starting and endpoint of the longest angle bisection line (h and
        #opp_point have been determined alongside the determination of the
        #triangle's type).              
        elif self.typ == "junction":
            #initialization of helper-variables
            x1 = self.opp_point.x        
            y1 = self.opp_point.y
            curr_x = x1
            curr_y = y1
            x2 = self.h.x
            y2 = self.h.y
            direction_x = x2-x1
            direction_y = y2-y1
            length = sqrt((y2-y1)**2 + (x2-x1)**2)
            #search along a line for a local maximum in the distance map
            while(sqrt((curr_y - y1)**2 + (curr_x - x1)**2) < length):
                int_x = int(round(curr_x))
                int_y = int(round(curr_y))
                curr_distance = distance_map[int_y, int_x]
                if curr_distance > max_distance:
                    max_distance = curr_distance
                    max_x = curr_x
                    max_y = curr_y
                curr_x += lamb*direction_x
                curr_y += lamb*direction_y
            #if we found a maximum, set radius and centroid and return 0
            if (max_y != -1 and max_x != -1):
                self.radius = max_distance
                self.centroid = Cpoint(max_x,max_y)
                return 0
            else:
            #if we didn't find a maximum, default to a radius of 1.0, set the
            #centroid to the real centroid of the triangle and return 1
                self.centroid = Cpoint(\
                            round((self.p1.x+self.p2.x+self.p3.x)*(1.0/3.0)),\
                            round((self.p1.y+self.p2.y+self.p3.y)*(1.0/3.0))) 
                self.radius = 1.0
                return 1
        #We do not set radius and centroid for isolated triangles because
        #we ignore them in the following processing anyways.
        else:
            return 0
        
    def __richcmp__(CshapeTriangle self, CshapeTriangle other, int op):            
        #special case if we try to compare to a None-object
        cdef int compare
        if type(other) == type(None):
            compare = -1
            return richcmp_helper(compare,op)
        #we only need the "==" comparison later on so we do not resolve the
        #other cases. It is kind of subjective how to handle "<" and ">" any-
        #ways.
        if op == 0 or op == 1 or op == 4 or op == 5:
            return False
        #obviously, triangles are euqal if all their points are the same (not
        #only if they are the same object!)
        if ( (self.p1 == other.p1 or self.p1 == other.p2 or self.p1 == other.p3) and \
             (self.p2 == other.p1 or self.p2 == other.p2 or self.p2 == other.p3) and \
             (self.p3 == other.p1 or self.p3 == other.p2 or self.p3 == other.p3) ):
            compare = 0
        else: compare = 1 
        return richcmp_helper(compare,op)

#reusable helper for __richcmp__                
cdef inline bint richcmp_helper(int compare, int op):
    if op == 2: # ==
        return compare == 0
    elif op == 3: # !=
        return compare != 0
    elif op == 0: # <
        return compare < 0
    elif op == 1: # <=
        return compare <= 0
    elif op == 4: # >
        return compare > 0
    elif op == 5: # >=
        return compare >= 0

#Builds CshapeTriangles from a list of points. The function is passed
#a list of 2D coordinates and a list of index-triples. So if we have an
#index triple (i,j,k) we look up the points corresponding to the triangle
#in the list of points at index i,j and k.
#At some point in the algorithm we need to create neighborhood relations
#between triangles. The most efficient way to do this I could come up with
#is a lookup in a python dictionary (should scale O(1) on average).
#We therefore first create three segments for each triangle. We store these
#segments in a dictionary where the key is made up of the segment's point's
#coordinates like ((x1,y1),(x2,y2)) and the value is the Csegment object.
#We make sure that we construct each segment only once. Then we use the 
#segments created this way to construct the triangles and set the neighborhood-
#relations accordingly.
#Returns a list of the newly created triangles.
def CbuildTriangles(list points, list triangle_point_indices):
    #definition of helper variables
    cdef i
    cdef dict segment_dict = {}
    cdef int x1, x2, x3, y1, y2, y3, N
    cdef Cpoint p1,p2,p3
    cdef Csegment s1,s2,s3
    cdef CshapeTriangle new_trianlge
    cdef list t,triangles
    
    triangles = []
    N = len(triangle_point_indices)
    for i in range(N):
        #resolve the indices to coordinates
        t = triangle_point_indices[i]
        x1 = points[t[0]][0]
        y1 = points[t[0]][1]
        x2 = points[t[1]][0]
        y2 = points[t[1]][1]
        x3 = points[t[2]][0]
        y3 = points[t[2]][1]
        
        #construct points from coordinates
        p1 = Cpoint(x1,y1)
        p2 = Cpoint(x2,y2)
        p3 = Cpoint(x3,y3)
        
        #Construct segments from points.        
        #segments have their points ordered internally at creation so that
        #p1 < p2. This is done so the identification via a ((x1,y1),(x2,y2))
        #key is unique for each segment (without order it could also be 
        #((x2,y2),(x1,y1))).
        s1 = Csegment(p1,p2)
        s2 = Csegment(p1,p3)
        s3 = Csegment(p2,p3)
        
        #create keys from coordinates (important: acces the segment's points
        #so we ensure proper ordering)
        key1 = ((s1.get_p1().get_x(),s1.get_p1().get_y()),\
                (s1.get_p2().get_x(),s1.get_p2().get_y()))
        key2 = ((s2.get_p1().get_x(),s2.get_p1().get_y()),\
                (s2.get_p2().get_x(),s2.get_p2().get_y()))
        key3 = ((s3.get_p1().get_x(),s3.get_p1().get_y()),\
                (s3.get_p2().get_x(),s3.get_p2().get_y()))
        
        #only update the dictionary if the created segment is new, else we
        #would overide the already existing segment which probably already has 
        #one of its triangle pointers set
        if key1 not in segment_dict:
            segment_dict.update({key1:s1})
        if key2 not in segment_dict:
            segment_dict.update({key2:s2})
        if key3 not in segment_dict:
            segment_dict.update({key3:s3})
        
        #build the triangles with segments taken from the segment dict, 
        #ensuring the triangle's neighbors can be set correctly
        new_triangle = CshapeTriangle(segment_dict[key1],segment_dict[key2],segment_dict[key3])
        triangles.append(new_triangle)
    
    return triangles
    
cdef get_neighbor(CshapeTriangle triangle,Csegment edge):
    cdef CshapeTriangle neighbor = edge.get_triangle1()
    if neighbor == triangle:
        neighbor = edge.get_triangle2()
    else:
        neighbor = edge.get_triangle1()
    return neighbor

def CcreateTriangleAdjacencyMatrix(list triangles not None):
    cdef int dim, j, i
    cdef CshapeTriangle neighbor1, neighbor2, neighbor3,t
                
    dim = len(triangles)
    adjacency_matrix = scipy.sparse.lil_matrix((dim,dim))
    
    for j in range(dim):
        adjacency_matrix[j,j]=0
        triangles[j].set_index(j)
        
    for i in range(dim):
        t = triangles[i]
        
        if t.typ == "junction":
            adjacency_matrix[i,t.get_neighbor1().get_index()] = \
                    distance(t.get_centroid(),t.get_neighbor1().get_centroid())
            adjacency_matrix[i,t.get_neighbor2().get_index()] = \
                    distance(t.get_centroid(),t.get_neighbor2().get_centroid())
            adjacency_matrix[i,t.get_neighbor3().get_index()] = \
                    distance(t.get_centroid(),t.get_neighbor3().get_centroid())
            adjacency_matrix[t.get_neighbor1().get_index(),i] = \
                    distance(t.get_centroid(),t.get_neighbor1().get_centroid())
            adjacency_matrix[t.get_neighbor2().get_index(),i] = \
                    distance(t.get_centroid(),t.get_neighbor2().get_centroid())
            adjacency_matrix[t.get_neighbor3().get_index(),i] = \
                    distance(t.get_centroid(),t.get_neighbor3().get_centroid())
        elif t.typ == "normal":
            adjacency_matrix[i,t.get_neighbor1().get_index()] = \
                    distance(t.get_centroid(),t.get_neighbor1().get_centroid())
            adjacency_matrix[i,t.get_neighbor2().get_index()] = \
                    distance(t.get_centroid(),t.get_neighbor2().get_centroid())
            adjacency_matrix[t.get_neighbor1().get_index(),i] = \
                    distance(t.get_centroid(),t.get_neighbor1().get_centroid())
            adjacency_matrix[t.get_neighbor2().get_index(),i] = \
                    distance(t.get_centroid(),t.get_neighbor2().get_centroid())
        else:
            adjacency_matrix[i,t.get_neighbor1().get_index()] = \
                    distance(t.get_centroid(),t.get_neighbor1().get_centroid())
            adjacency_matrix[t.get_neighbor1().get_index(),i] = \
                    distance(t.get_centroid(),t.get_neighbor1().get_centroid())
                          
    return adjacency_matrix
    
def CbruteforcePruning(adjacency_matrix,np.ndarray triangles,int order,verbose):
    cdef int curr_order = 0
    cdef int i
    cdef list indices
    cdef np.ndarray rows

    while curr_order < order:
        if verbose:
            print "\t from bruteforcePruning: current order",curr_order
        
        rows = adjacency_matrix.rows
        indices = []
        for i in xrange(adjacency_matrix.shape[0]):
            if(len(rows[i]) == 1):
                indices.append(i)
                
        indices = list(set(indices).symmetric_difference(set(range(adjacency_matrix.shape[0]))))
        adjacency_matrix = adjacency_matrix.tocsc()
        adjacency_matrix = adjacency_matrix[:,indices]
        adjacency_matrix = adjacency_matrix[indices,:]
        adjacency_matrix = lil2(adjacency_matrix) 
        triangles = triangles[indices]
        curr_order += 1
        
    return (adjacency_matrix,triangles)
                
        