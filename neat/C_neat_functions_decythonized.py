import numpy as np
import time
from scipy.sparse.lil import lil_matrix
from bisect import bisect_left
from numpy import sqrt, round

def distance(p1,p2):
    dist = sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y))
    if dist == 0:
        dist = 0.1
    return dist
 
class Cpoint:        
    def __init__(self, x, y):
        self.x = x
        self.y = y     
    
    def __richcmp__(self, other, op):
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
        s = "(" + str(self.x) + "," + str(self.y) + ")"
        return s

class Csegment:        
    def __init__(self, p1, p2):      
        if p1 < p2:
            self.p1 = p1
            self.p2 = p2
        else:
            self.p1 = p2
            self.p2 = p1         
        self.midpoint = Cpoint((self.p1.x+self.p2.x)/2,(self.p1.y+self.p2.y)/2)
        self.triangle1 = None
        self.triangle2 = None
   
    def __str__(self):
        return "(%d,%d) -> (%d,%d)"%\
                (self.p1.x,self.p1.y,self.p2.x,self.p2.y)

    def __richcmp__(self, other, op):
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
     
    def get_p1(self): return self.p1
    def get_p2(self): return self.p2
    def get_triangle1(self): return self.triangle1 
    def set_triangle1(self, t): self.triangle1 = t           
    def get_triangle2(self): return self.triangle2          
    def set_triangle2(self, t): self.triangle2 = t             
    def get_midpoint(self): return self.midpoint     
       
    def connects(self, other):
        if (self == other): return False              
        elif((self.p1 == other.p1) or (self.p2 == other.p1) \
            or (self.p1 == other.p2) or (self.p2 == other.p2)): return True        
        else: return False
                      
    def get_cp(self, seg): 
        if(self.connects(seg)):
            if(self.p1 == seg.p1 or self.p1 == seg.p2): cp = self.p1             
            else: cp = self.p2             
        else: cp = None
        return cp

class CshapeTriangle:
    def __init__(self, edge1, edge2, edge3):
        self.typ = None   
        self.neighbor_list = []                
        self.edge1 = edge1
        self.edge2 = edge2
        self.edge3 = edge3     
        self.p1 = self.edge1.p1
        self.p2 = self.edge1.p2
        if(self.edge2.p1 == self.p1 or self.edge2.p1 == self.p2):
            self.p3 = self.edge2.p2
        else:
            self.p3 = self.edge2.p1                  
    
        if edge1.triangle1 == None: edge1.triangle1 = self
        else: edge1.triangle2 = self
        if edge2.triangle1 == None: edge2.triangle1 = self
        else: edge2.triangle2 = self 
        if edge3.triangle1 == None: edge3.triangle1 = self
        else: edge3.triangle2 = self
    
    def get_center(self): return self.center
    def set_center(self, distance_map):
        return_val = 0
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
        
    def get_radius(self): return self.radius

    def get_p1(self): return self.p1       
    def get_p2(self): return self.p2       
    def get_p3(self): return self.p3   
    def get_edge1(self): return self.edge1      
    def get_edge2(self): return self.edge2       
    def get_edge3(self): return self.edge3 
    def get_index(self): return self.index
    def set_index(self, index): self.index = index
    
    def get_type(self): return self.typ 

    def init_triangle_mesh(self):
        self.set_neighbors()
        self.set_type()
    
    def set_neighbors(self):
        self.neighbor_list = []
        if self.edge1.triangle1 == self and self.edge1.triangle2 != None:
            self.add_neighbor(self.edge1.triangle2) 
        elif self.edge1.triangle2 == self and self.edge1.triangle1 != None:  
            self.add_neighbor(self.edge1.triangle1) 
        
        if self.edge2.triangle1 == self and self.edge2.triangle2 != None:
            self.add_neighbor(self.edge2.triangle2) 
        elif self.edge2.triangle2 == self and self.edge2.triangle1 != None:  
            self.add_neighbor(self.edge2.triangle1) 
    
        if self.edge3.triangle1 == self and self.edge3.triangle2 != None:
            self.add_neighbor(self.edge3.triangle2) 
        elif self.edge3.triangle2 == self and self.edge3.triangle1 != None:  
            self.add_neighbor(self.edge3.triangle1) 
        
    def set_type(self):
        neighbors = 0 
        type_dict = {0:"isolated", 1:"end", 2:"normal", 3:"junction"}   
        neighbors = len(self.neighbor_list)
        if neighbors > 3:
            print "number of neighbors not in (0,1,2,3)... really weird!"
        else:
            self.typ = type_dict[neighbors]            
        
    #getter and setter for the triangle's neighbors        
    def get_neighbor(self, index):
        return self.neighbor_list[index]
    def set_neighbor(self, index, n):
        self.neighbor_list[index] = n 
    def add_neighbor(self, n):
        if n not in self.neighbor_list:
            self.neighbor_list.append(n)
    def remove_neighbor(self, n):
        self.neighbor_list.remove(n)
        
    
    #helper function which recieves two triangles as input and returns the
    #shared edge if they are neighbors or "None" if they are not.
    def get_connecting_edge(self, n):
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
    def set_center_junction(self, distance_map):
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
    def set_center_normal(self, distance_map):
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
    def set_center_end(self, distance_map):
        #find out which of the edges are the external edges:
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
    def find_center(self, dm):
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
    def find_local_maximum(self, distance_map):
        
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
    def __richcmp__(self, other, op):            
        #special case if we try to compare to a None-object
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
def richcmp_helper(compare, op):
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
def CbuildTriangles(points, triangle_point_indices):
    #definition of helper variables
    
    triangles = []
    segment_dict = {}
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
def update_edge_pointers(end, neighbor):
    edge = end.get_connecting_edge(neighbor)
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
def traverse_triangles(prev, curr):
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
def confirm_surplus_branch(end, order):
    curr = end.get_neighbor(0)
    prev = end
    j = 0

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
def CbruteforcePruning(triangles, order, verbose):
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
def CcreateTriangleAdjacencyMatrix(triangles): 
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
                    
    