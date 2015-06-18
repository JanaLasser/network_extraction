
"""
Created on Thu Jan 22 14:36:58 2015

@author: Jana Lasser
"""

'''
Copyright (C) 2015 Jana Lasser GPL-3.0
'''

#python setup_C_net_functions.py build_ext --inplace

from libc.math cimport sqrt
from libc.math cimport round
import numpy as np
import time
cimport numpy as np
#import scipy
from scipy.sparse.lil import lil_matrix
from bisect import bisect_left

ctypedef np.int_t DTYPE_t 


#helper function to calculate the distance between two points on a plane
cdef distance(Cpoint p1,Cpoint p2):
    cdef double dist = sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y))
    #be aware that this is a dirty hack necessary because the centerpoints are 
    #originating from a discrete space!
    if dist == 0:
        dist = 0.1
    return dist


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
        self.midpoint = Cpoint((self.p1.x+self.p2.x)/2,(self.p1.y+self.p2.y)/2)
   
    def __str__(self):
        return "(%d,%d) -> (%d,%d)"%\
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
    - center: "midpoint" of the triangle, point with the largest distance
                to the nearest edge of the foreground structure. 
    - radius: distance to the nearest edge of the foreground structure
    - index:    counter used to fix the triangles position in a triangle
                adjacency matrix
    - neighbor_ist: pointers to the triangles neighbors (3 to 0 possible
                neighbors) for junction, normal, end and isolated triangles.
                The pointers point to None at triangle construction and are set
                later by the triangle's "set_type" method.
    - Cpoints angle_bisection_start and angle_bisection_end
                helper points for the calculation of center and radius
                
Exposed functions:
    - get_center(): returns the center of the triangle
    - set_center(): set the center of a triangle using the distance_map
    - get_radius(): returns the radius of the structure at the center
    - get_pi(): returns the ith point of the triangle
    - get_edgei(): returns the ith edge of the triangle
    - get_index(): returns the index of the triangle (used for creation of the
                adjacency matrix)
    - set_index(): sets the index of the triangle
    - get_type(): returns the type of the triangle
    - set_type(): sets the type of the triangle to either "junction", "normal",
                "end" or "isolated"

            
    
'''       
cdef class CshapeTriangle:
    cdef Csegment edge1, edge2, edge3
    cdef Cpoint p1, p2, p3, center, angle_bisection_start, angle_bisection_end
    cdef str typ 
    cdef double radius
    cdef int index
    cdef list neighbor_list 
   
    #Constructor creates a triangle from three segments, sets the three points
    #of the triangle and the segment's pointer to the newly created triangle
    def __init__(self, Csegment edge1, Csegment edge2, Csegment edge3):
        #type cannot be determined at creation and has to be set at a later
        #point via the set_type function as is the neighbor_list
        self.typ = None   
        self.neighbor_list = []                
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
    
    #getter and setter for the center, the setter calls sub-functions for
    #each triangle type. The setter also sets the radius of the triangle.
    #the function returns 0 if the radius has been set correctly or if an 
    #isolated triangle or triangle of unknown type is encountered. If the 
    #radius defaulted to zero it will return 1 (for debugging reasons)
    cpdef get_center(self): return self.center
    cpdef set_center(self,np.ndarray[DTYPE_t,ndim=2] distance_map):
        cdef int return_val = 0
        if self.typ == None:
            print "tried to set center of None-type triangle, nothing happened"
        elif self.typ == "junction":
            return_val = self.set_center_junction(distance_map)
        elif self.typ == "normal":
            return_val = self.set_center_normal(distance_map)
        elif self.typ == "end":
            return_val = self.set_center_end(distance_map)
        elif self.typ == "isolated":
            pass
        else:
            print "unknown triangle type!"
        
        if type(self.center) == type(None):
            self.center = Cpoint(\
                        round((self.p1.x+self.p2.x+self.p3.x)*(1.0/3.0)),\
                        round((self.p1.y+self.p2.y+self.p3.y)*(1.0/3.0)))
                        
        return return_val
        
    cpdef get_radius(self): return self.radius

    #getter and setter for the triangle's points, edges and index        
    cpdef get_p1(self): return self.p1       
    cpdef get_p2(self): return self.p2       
    cpdef get_p3(self): return self.p3   
    cpdef get_edge1(self): return self.edge1      
    cpdef get_edge2(self): return self.edge2       
    cpdef get_edge3(self): return self.edge3 
    cpdef get_index(self): return self.index
    cpdef set_index(self,int index): self.index = index
    
    #getter and setter for the triangle's type. The type is determined based
    #on how many neighbors the triangle has
    cpdef get_type(self): return self.typ 

    cpdef init_triangle_mesh(self):
        self.set_neighbors()
        self.set_type()
    
    cpdef set_neighbors(self):
        self.neighbor_list = []
        #set the triangels neighbors based on the pointers stored in the edges
        #check for neighbors at edge1
        if self.edge1.triangle1 == self and self.edge1.triangle2 != None:
            self.add_neighbor(self.edge1.triangle2) 
        elif self.edge1.triangle2 == self and self.edge1.triangle1 != None:  
            self.add_neighbor(self.edge1.triangle1) 
        
        #check for neighbors at edge2
        if self.edge2.triangle1 == self and self.edge2.triangle2 != None:
            self.add_neighbor(self.edge2.triangle2) 
        elif self.edge2.triangle2 == self and self.edge2.triangle1 != None:  
            self.add_neighbor(self.edge2.triangle1) 
         
        #check for neighbors at edge3
        if self.edge3.triangle1 == self and self.edge3.triangle2 != None:
            self.add_neighbor(self.edge3.triangle2) 
        elif self.edge3.triangle2 == self and self.edge3.triangle1 != None:  
            self.add_neighbor(self.edge3.triangle1) 
        
    cpdef set_type(self):
        cdef int neighbors = 0 
        cdef dict type_dict = {0:"isolated", 1:"end", 2:"normal", 3:"junction"}   
        neighbors = len(self.neighbor_list)
        if neighbors > 3:
            print "number of neighbors not in (0,1,2,3)... really weird!"
        else:
            self.typ = type_dict[neighbors]            
        
    #getter and setter for the triangle's neighbors        
    cpdef get_neighbor(self,int index):
        return self.neighbor_list[index]
    cpdef set_neighbor(self,int index,CshapeTriangle n):
        self.neighbor_list[index] = n 
    cpdef add_neighbor(self, CshapeTriangle n):
        if n not in self.neighbor_list:
            self.neighbor_list.append(n)
    cpdef remove_neighbor(self, CshapeTriangle n):
        self.neighbor_list.remove(n)
        
    
    #helper function which recieves two triangles as input and returns the
    #shared edge if they are neighbors or "None" if they are not.
    cdef get_connecting_edge(self,CshapeTriangle n):
        if self.edge1 in [n.get_edge1(),n.get_edge2(),n.get_edge3()]:
            return self.edge1
        elif self.edge2 in [n.get_edge1(),n.get_edge2(),n.get_edge3()]:
            return self.edge2
        elif self.edge3 in [n.get_edge1(),n.get_edge2(),n.get_edge3()]:
            return self.edge3
        else:
            print "triangles ckecked for connection which are not neighbors!"
            return None
    
    #Helper function to set the start and end of the correct angle bisection
    #line if the triangle is a junction.
    #We set the angle bisection line to be the longest of the three angle
    #bisection lines in the triangle
    cdef set_center_junction(self,np.ndarray[DTYPE_t,ndim=2] distance_map):
        cdef Cpoint mid_p1p2, mid_p1p3, mid_p2p3
        cdef double abs_bisect1,abs_bisect2,abs_bisect3
        #midpoints of the edges:
        mid_p1p2 = Cpoint((self.p1.x+self.p2.x)/2.0, (self.p1.y+self.p2.y)/2.0)
        mid_p1p3 = Cpoint((self.p1.x+self.p3.x)/2.0, (self.p1.y+self.p3.y)/2.0)
        mid_p2p3 = Cpoint((self.p2.x+self.p3.x)/2.0, (self.p2.y+self.p3.y)/2.0) 
        #distances between edge-midpoints and opposing points
        abs_bisect1 = distance(mid_p1p2,self.p3)        
        abs_bisect2 = distance(mid_p1p3,self.p2)
        abs_bisect3 = distance(mid_p2p3,self.p1)     
        #set angle_bisection_start and angle_bisection_end
        if (abs_bisect1 >= abs_bisect2 and abs_bisect1 >= abs_bisect3):
            self.angle_bisection_start = mid_p1p2
            self.angle_bisection_end = self.p3
        elif(abs_bisect2 >= abs_bisect1 and abs_bisect2 >= abs_bisect3):
            self.angle_bisection_start = mid_p1p3
            self.angle_bisection_end = self.p2
        else:
            self.angle_bisection_start = mid_p2p3
            self.angle_bisection_end = self.p1   
        #call the helper to calculate the local maximum in the distance map
        #alonge the angle bisection line and set the triangle's center and 
        #radius. Return 0 or 1 dependin on the correct finding of the radius
        return self.find_local_maximum(distance_map)
        
    #Helper function to set the start and end of the correct angle bisection
    #line if the triangle is normal.
    #We set the angle bisection line to be bisection line of the internal angle
    #(i.e. the only angle in the triangle not enclosed by the external edge).
    cdef set_center_normal(self,np.ndarray[DTYPE_t,ndim=2] distance_map):
        #find out which of the edges is the external edge (edge shared
        #with the contour) and set its opposing point accordingly
    
        if self.edge1.triangle1 == None or self.edge1.triangle2 == None:
            self.angle_bisection_start = self.edge1.get_midpoint()
            self.angle_bisection_end = self.edge2.get_cp(self.edge3)
        elif self.edge2.triangle1 == None or self.edge2.triangle2 == None:
            self.angle_bisection_start = self.edge2.get_midpoint()
            self.angle_bisection_end = self.edge1.get_cp(self.edge3)
        else:
            self.angle_bisection_start = self.edge3.get_midpoint()
            self.angle_bisection_end = self.edge2.get_cp(self.edge1)
        
        return self.find_center(distance_map)


    #Helper function to set the start and end of the correct angle bisection
    #line if the triangle is an end.   
    #We set the angle bisection line to be the bisection line of the external
    #angle (i.e. the only angle that is only enclosed by external edges.)
    cdef set_center_end(self,np.ndarray[DTYPE_t,ndim=2] distance_map):
        #find out which of the edges are the external edges:
        cdef Csegment external1,external2, internal     
        if self.edge1.triangle1 != None and self.edge1.triangle2 != None:
            internal = self.edge1
            external1 = self.edge2
            external2 = self.edge3
        elif self.edge2.triangle1 != None and self.edge2.triangle2 != None:
            internal = self.edge2
            external1 = self.edge1
            external2 = self.edge3
        else:
            internal = self.edge3
            external1 = self.edge1
            external2 = self.edge2            
        self.angle_bisection_start = internal.get_midpoint()
        self.angle_bisection_end = external1.get_cp(external2)
        #call the helper to calculate the local maximum in the distance map
        #alonge the angle bisection line and set the triangle's center and 
        #radius. Return 0 or 1 dependin on the correct finding of the radius 
        return self.find_local_maximum(distance_map)


    #Helper function to find the "center" of a normal triangle.
    #It has turned out that finding the center by looking for a local maxium
    #in the distance map produces  zigzag graphs. To improve this, for normal
    #triangles we choose as point which is contributed to the graph the 
    #midpoint of the shorter of the two internal edges
    cdef find_center(self, np.ndarray[DTYPE_t,ndim=2] dm):
        cdef float radius
        if self.edge1.triangle1 == None or self.edge1.triangle2 == None:
            if distance(self.edge2.p1,self.edge2.p2) < \
               distance(self.edge3.p1,self.edge3.p2):
                self.center = self.edge2.get_midpoint()
            else:
                self.center = self.edge3.get_midpoint()

        elif self.edge2.triangle1 == None or self.edge2.triangle2 == None:
            if distance(self.edge1.p1,self.edge1.p2) < \
               distance(self.edge3.p1,self.edge3.p2):
                self.center = self.edge1.get_midpoint()
            else:
                self.center = self.edge3.get_midpoint()
        else:
            if distance(self.edge1.p1,self.edge1.p2) < \
               distance(self.edge2.p1,self.edge2.p2):
                self.center = self.edge1.get_midpoint()
            else:
                self.center = self.edge2.get_midpoint()

        radius = dm[int(self.center.y),int(self.center.x)]
        if radius == 0:
            radius = 1
            self.radius = radius
            return 1
        else:
            self.radius = radius
            return 0

    #Helper function to find a local maximum in the distance map along a line.
    #Start and end of the line are stored in the triangle's variables
    #"angle_bisection_start" and "angle_bisection_end"
    cdef find_local_maximum(self,np.ndarray[DTYPE_t,ndim=2] distance_map):
        #declaration of helper variables
        cdef double x1, y1, x2, y2, length, lamb, curr_x, curr_y
        cdef int int_x, int_y
        cdef double direction_x, direction_y, max_distance
        cdef double max_x, max_y
        
        #use typed indexing for fast array acces
        DTYPE = np.int
        cdef DTYPE_t curr_distance
        assert distance_map.dtype == DTYPE  
        
        #initialization of helper variables
        max_distance = 0
        max_x = -1.0
        max_y = -1.0
        lamb = 0.1 
        x1 = self.angle_bisection_end.x        
        y1 = self.angle_bisection_end.y
        x2 = self.angle_bisection_start.x
        y2 = self.angle_bisection_start.y
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
            
        #if we found a maximum, set radius and center and return 0
        if (max_y != -1 and max_x != -1):
            self.radius = max_distance
            #if the triangle is an end triangle, set the center to the tip
            if self.typ == "end":
                if self.edge1.triangle2!=None and self.edge1.triangle1!=None:
                    self.center = self.edge2.get_cp(self.edge3)
                elif self.edge2.triangle2!=None and self.edge2.triangle1!=None:
                    self.center = self.edge1.get_cp(self.edge3)
                else:
                    self.center = self.edge1.get_cp(self.edge2) 
            #else set the center to the location of the local maximum
            else: 
                self.center = Cpoint(max_x,max_y)
            return 0
            
        #if we didn't find a maximum, default to a radius of 1.0, set the
        #center to the centroid of the triangle and return 1
        else:
            self.center = Cpoint(\
                        round((self.p1.x+self.p2.x+self.p3.x)*(1.0/3.0)),\
                        round((self.p1.y+self.p2.y+self.p3.y)*(1.0/3.0))) 
            self.radius = 1.0
            return 1
            
    #Ordering of triangles. We only need to check if triangles are equal,
    #i.e. if they are composed of the same points. All other comparisions
    #are not implemented
    def __richcmp__(CshapeTriangle self, CshapeTriangle other, int op):            
        #special case if we try to compare to a None-object
        cdef int compare
        if type(other) == type(None):
            compare = -1
            return richcmp_helper(compare,op)
        #we only need the "==" comparison later on so we do not resolve the
        #other cases. It is kind of ambiguous how to handle "<" and ">" any-
        #ways.
        if op == 0 or op == 1 or op == 4 or op == 5:
            return False
        #obviously, triangles are euqal if all their points are the same (not
        #only if they are the same object!)
        if ( (self.p1==other.p1 or self.p1==other.p2 or self.p1==other.p3) and\
             (self.p2==other.p1 or self.p2==other.p2 or self.p2==other.p3) and\
             (self.p3==other.p1 or self.p3==other.p2 or self.p3==other.p3) ):
            compare = 0
        else: compare = 1 
        return richcmp_helper(compare,op)
        
    def __str__(self):
        s = self.edge1.__str__() + " | " + self.edge1.__str__() + " | " + \
            self.edge3.__str__()
        return s

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
    cdef int x1, x2, x3, y1, y2, y3
    cdef Cpoint p1,p2,p3
    cdef Csegment s1,s2,s3
    cdef CshapeTriangle new_trianlge
    cdef list t,triangles
    
    triangles = []
    for i,t in enumerate(triangle_point_indices):
        #resolve the indices to coordinates
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
        new_triangle = CshapeTriangle(segment_dict[key1],segment_dict[key2],\
                                      segment_dict[key3])
        triangles.append(new_triangle)
    return triangles


#Helper function to update the pointers to the triangles stored in edges when
#a triangle is deleted during the pruning process    
def update_edge_pointers(CshapeTriangle end, CshapeTriangle neighbor):
    cdef Csegment edge = end.get_connecting_edge(neighbor)
    if edge.get_triangle1() == end:
        edge.set_triangle1(None)
    elif edge.get_triangle2() == end:
        edge.set_triangle2(None) 
    else:
        #TODO: figure out why this can happen!
        print "edge relations wrong!"
    neighbor.set_neighbors()
    end.set_neighbors()
    end.set_type()
    neighbor.set_type()

#Helper function to update the pointers to the neighboring triangles stored
#in the triangle itself when a triangle is deleted during the pruning process        
#def update_neighbor_pointers(CshapeTriangle end, CshapeTriangle neighbor):    
#    #disconnect the end from the neighbor
#    end.remove_neighbor(neighbor)
#    neighbor.remove_neighbor(end)
    


#Helper function to traverse along a part of the network given the previous
#and the current triangle. Will return the next triangle along the edge
def traverse_triangles(CshapeTriangle prev, CshapeTriangle curr):
    cdef CshapeTriangle neighbor1, neighbor2, nextTriangle
    if curr.get_type() != "normal":
        print "tried to traverse from non-normal triangle, aborting!"
        return
    neighbor1 = curr.get_neighbor(0)
    neighbor2 = curr.get_neighbor(1)
 
    if neighbor1 == prev:
        nextTriangle = neighbor2
    else:
        nextTriangle = neighbor1
    
    if nextTriangle == None:
        print "neighborhood relations during traverse triangle botched!"
        print curr.neighbor_list
        
    return nextTriangle
    

#Helper function to confirm if an end-triangle is part of a surplus branch.
#It will check for the occurrence of a junction triangle "order" steps down 
#the road, if it does not find a junction, the end-triangle is a real tip and
#will be spared from pruning.
#Makes use of the "traverse_triangles" function to search downstream from the
#end triangle.
def confirm_surplus_branch(CshapeTriangle end,int order):
    cdef CshapeTriangle curr = end.get_neighbor(0)
    cdef CshapeTriangle prev = end
    cdef CshapeTriangle temp
    cdef int j = 0

    while j < order:
        if curr.get_type() == "junction" or curr.get_type() == "end":
            return True
        elif curr.get_type() == "normal":
            temp = traverse_triangles(prev,curr)
            prev = curr
            curr = temp
            j += 1
        else:
            return False
    return False
    
    
#Often a noisy contour leads to the creation of end-triangles which are
#are directly attached or very close to junction triangles. We prune away these
#structures to prevent the creation of fake junctions and therefore fake 
#branches. The order up to which triangles will be pruned away can be
#controlled by the main script.
#TODO: maybe improve surplus branch confirmation so it does not search all
#tips every time
def CbruteforcePruning(np.ndarray triangles,int order,bint verbose):
    cdef int curr_order,i
    cdef list indices
    cdef CshapeTriangle t,neighbor

    curr_order = 0
    for t in triangles:
        t.init_triangle_mesh()
    
    while curr_order < order:
        indices = []
        if verbose:
            print "\t from bruteforcePruning: current order",curr_order
        
        #if a triangle is an end, it will get pruned
        for i,t in enumerate(triangles):
            if t.get_type() == "end":
                #confirm that the end we are dealing with really is part of a
                # end + N*normal + junction structure and not a real tip 
                if confirm_surplus_branch(t,order):
                    indices.append(i)
                    if len(t.neighbor_list) > 1:
                        print "end trinalge with too many neighbors detected!"
                    neighbor = t.get_neighbor(0)
                    update_edge_pointers(t,neighbor)

        indices = list(set(indices).symmetric_difference(set\
                 (range(len(triangles)))))                
        triangles = triangles[indices]
        
        for t in triangles:
            t.init_triangle_mesh()
        curr_order += 1
    
    #make sure no isolated triangles make it into the final triangle list    
    indices = []
    for i,t in enumerate(triangles):
        if t.get_type() == "isolated":
            indices.append(i)
            
    indices = list(set(indices).symmetric_difference(set\
             (range(len(triangles)))))                
    triangles = triangles[indices]
    return triangles

#Create the adjacency matrix of the graph from the list of triangles.
#First we give each triangle an index according to its position in the list.
#If we find that a triangle with index i neighbors another triangle with index
#j, we create an entry in the adjacency matrix at position (i,j). The value
#of the entry is the distance between the centers of the two triangles.
def CcreateTriangleAdjacencyMatrix(list triangles not None):
    cdef int dim, j, i, index
    cdef float dist
    cdef CshapeTriangle n,t
     
    #initialize the adjacency matrix and triangle indices        
    dim = len(triangles)
    adjacency_matrix = lil_matrix((dim,dim))
    adjacency_matrix.setdiag(np.zeros((dim,1)))    
    
    #iterate over all triangles and create entries in the adjacency matrix
    for i in range(dim):
        triangles[i].set_index(i)
        
    for j in range(dim):
        t = triangles[j]
        
        for n in t.neighbor_list:
            dist = distance(t.get_center(),n.get_center())                
            index = n.get_index()
            adjacency_matrix[j,index] = dist
            
    return adjacency_matrix
                    
    