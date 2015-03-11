# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 13:34:10 2015

@author: Jana Lasser
"""
'''
Copyright (C) 2015 Jana Lasser GPL-3.0
'''

from PIL import Image
import numpy as np
import cv2
import math
import scipy
from scipy import sparse,spatial
import cv
import networkx as nx
import matplotlib.pyplot as plt
plt.ioff()
from os.path import join
import time

from C_vectorize_functions import CbuildTriangles, CbruteforcePruning
from C_vectorize_functions import CcreateTriangleAdjacencyMatrix, Cpoint





def getImage(image_path):
    '''
    Helper for opening an image, converting it to a numpy array filled
    with floats

    Parameters
    ----------
    image_path: string
        path to the file containing the image
        
    Returns
    -------
    image: ndarray
        array containing the image data read from the file
        
    '''
    
    image = Image.open(image_path)
    image = image.convert('L')
    image = np.asarray(image,dtype=np.uint8)
    return image

def RGBtoGray(image):
    '''
    Helper for opening an image, converting it to grayscale and converting it to
    a numpy array filled with floats

    Parameters
    ----------
        image_path: string
            path to the file containing the image
        
    Returns
    -------
        image: ndarray
            array containing the image data read from the file
        
    '''
    
    if(isinstance(image,str)):
        image = getImage(image)
    if (image.ndim == 2): return image      
    else:
        image = image.astype(np.float32).sum(axis=2);image/=3.0
        return image

def otsu_threshold(image):
    
    '''
    Implementation of "Otsu Thresholding" which will look for the threshold of the
    image that has the least in class variance.

    Parameters
    ----------
    image: ndarray
        image for which the threshold will be calculated, needs to be single channel and
        grayscale
    
    Returns
    -------
    t: integer
        optimal threshold for the image
    
    '''
    
    #create histogram and convert to array with length 256 containing
    #only the number of pixels with each brightness value
    histData = np.histogram(image, np.arange(256))
    histData = np.asarray(histData[0])

    #np.histogramm doesn't create a full list with 256 entries if the last entries are 0,
    #need to append zeroes so the loops in the body of the function work properly
    l = 256 - len(histData)
    while(l != 0):
        histData = np.append(histData,0)
        l -=1
    
    
    #initialize variables
    total = image.size
    sum = 0.0
    t = 0
    
    #calculate weighted sum
    while(t < 256):
        sum += t*float(histData[t])
        t +=1

    
    #initialize variables
    sumB = 0.0
    wB = 0
    wF = 0
    varMax = 0.0
    threshold = 0
    t = -1
    
    #main loop iterates over all the bins, calculating the between class variance.
    #the maximum of the between class variance is the minimum of the in class variance
    #so the threshold at the maximum is selected
    while(t < 256):
        t += 1
        
        #calculate weight background
        wB += histData[t]
        if(wB == 0):
            continue
        
        #calculate weight foreground
        wF = total - wB;
        if(wF == 0):
            break
            
        sumB += float(t)*float(histData[t])
        
        #calculate mean background
        mB = float(sumB) / float(wB)
        
        #calculate mean foregroud
        mF = float(sum - sumB) / float(wF)
        
        #calculate between class variance
        varBetween = float(wB) * float(wF) * (float(mB) - float(mF)) * (float(mB) - float(mF))
        
        #check if new maximum found
        if (varBetween > varMax):
            varMax = varBetween
            threshold = t
            
    return threshold
    

        
def getContours(image):
    '''
    Wrapper around openCV's cv2.findContours() function
    (see: http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#cv2.findContours)
    Sets convenient options and converts contours to a list of ndarrays.
    
    Parameter:
    ---------
        image: ndarray, input image
        
    Returns:
    --------
        contours: list, list of contours
    '''
    
    image = image.astype(np.uint8)
    contours,hirachy = cv2.findContours(image,cv2.RETR_CCOMP,\
                                        cv2.CHAIN_APPROX_TC89_L1)
    contours = np.asarray(contours)   
    
    return contours
    
def flattenContours(raw_contours):
    '''
    Helper function for flattening contours consisting of nested ndarrays
    
    Parameter:
    ----------
        contours: list, list of ndarrays
        
    Returns:
    --------
        flattened_contours: list, list of flattened ndarrays
    '''
    
    converted_contours = []
    for contour in raw_contours:
        new_contour = []
        for point in contour:
            x,y = (float(point[0][0]),float(point[0][1]))
            new_point = [x,y]
            new_contour.append(new_point)
        converted_contours.append(new_contour)    
    return converted_contours
    
def countDuplicates(thresholded_contours):
    '''
    Debugging function intended to count the number of nodes that have similar
    x coordinates or similar y coordinates.
    
    Parameter:
    ---------
        thresholded_contours: list of lists of lists, list of contours composed
        of lists of 2-element-lists containing the coordinates
    '''
    #create nx2 dimensional array containing x- and y coordinates
    contour_list = np.asarray(thresholded_contours)
    all_points = []
    for tc in contour_list:
        all_points.extend(np.asarray(tc).flatten())
    all_points = np.asarray(all_points).reshape(len(all_points)/2,2)
    
    #count duplicates
    number_all = len(all_points)
    number_unique_x = len(set(all_points[0:,0]))
    number_unique_y = len(set(all_points[0:,1]))
    duplicate_x = number_all - number_unique_x
    duplicate_y = number_all - number_unique_y
    print "found %d x-duplicates\nfound %d y-duplicates"%(duplicate_x,duplicate_y)
    
def thresholdContours(contours,threshold):
    '''
    Thresholds a given list of contours by length.
    
    Parameter:
    ---------
        contours: list, list of contours
        threshold: integer, length-threshold
        
    Returns:
    --------
        filtered_cont: list, list of contours
    '''
    
    thresholded_contours = []
    for c,i in zip(contours,xrange(len(contours))):
        if (len(c) > threshold):
            thresholded_contours.append(c)           
    return thresholded_contours

def roundTripConnect(start, end):
    '''
    Connects the last point in a contour to the first point.
    
    Parameter:
    ---------
        start: integer, index of first point
        end: integer, index of last point
    
    Returns:
    ---------
        list of tuples, contour connected to a circle
    
    '''
    return [(i, i+1) for i in range(start, end)] + [(end, start)]
  

def getInteriorPoint(contour):
    '''
    Finds an interior point in a polygon. The function gets the whole polygon
    
    Parameter:
    ---------
        contour: contour which we want to find an interior point for
        
    Returns:
    ---------
        interior point (member of the custom "point" class)
    
    '''
    
    from shapely.geometry import Point
    from shapely.geometry import Polygon
    poly = Polygon(contour)    
    
    
    #check if the centroid of the contour already qualifies as interior point
    x = 0
    y = 0
    for point in contour:
        x += point[0]
        y += point[1]
    x /= len(contour)
    y /= len(contour)
    
    centroid = Point(x,y)
    if centroid.within(poly):
        return [x,y]
    
    
    #if the centroid is no good, invoke a more sofisticated point-finding method
    p1 = Cpoint(contour[0][0],contour[0][1])
    cp = Cpoint(contour[1][0],contour[1][1])
    p2 = Cpoint(contour[2][0],contour[2][1])
    
    #rotation matrix
    def rotate(angle,vec):
        angle = math.radians(angle)
        rot = np.array([math.cos(angle),-math.sin(angle), \
              math.sin(angle),math.cos(angle)]).reshape(2,2)
        return np.dot(rot,vec) 

    N = 0.5
    
    seg1 = [cp.get_x()-p1.get_x(),cp.get_y()-p1.get_y()]
    seg2 = [p2.get_x()-cp.get_x(),p2.get_y()-cp.get_y()]
    
    #angle between the segments
    phi_plus = math.atan2(seg2[1],seg2[0])
    phi_minus = math.atan2(seg1[1],seg1[0])
    phi = math.degrees((math.pi - phi_plus + phi_minus)%(2*math.pi))
    
    #Contour finding seems to not always go counter-clockwise around contours
    #which makes life difficult -> need to check if found interior point is
    #inside the polygon and if not, take 360-phi and find another point with this
    #angle
    is_interior_point = False
    
    #180 degree case, maybe obsolete
    if(phi == 180):
        rot_seg = rotate(90,seg2)
        int_point = [cp.get_x() - N*rot_seg[0] , cp.get_y() - N*rot_seg[1]]
        test_point = Point(int_point)
        
        if(not test_point.within(poly)):
            rot_seg = rotate(-90,seg2)
            int_point = [cp.get_x() - N*rot_seg[0] , cp.get_y() - N*rot_seg[1]]
            test_point = Point(int_point)
            
            if test_point.within(poly):
                is_interior_point = True
            
            #if nothing else helps: find interior point by sampling all points on a
            # circle with radius N with 1 degree difference
            else:
                angle = 0
                while (is_interior_point == False and angle < 361):
                    rot_seg = rotate(angle,seg2)
                    int_point = [cp.get_x() + N*rot_seg[0],cp.get_y() +N*rot_seg[1]]
                    test_point = Point(int_point)
                    angle +=1
                    if test_point.within(poly):
                        is_interior_point = True
            
    else:
        #"normal" case
        rot_seg = rotate(0.5*phi,seg2)
        int_point = [cp.get_x() + N*rot_seg[0],cp.get_y() +N*rot_seg[1]]
        test_point = Point(int_point)
    
        if(not test_point.within(poly)):
            rot_seg = rotate(-0.5*phi,seg2)
            int_point = [cp.get_x() + N*rot_seg[0],cp.get_y() +N*rot_seg[1]]
            
            test_point = Point(int_point)
            if test_point.within(poly):
                is_interior_point = True
                
            #if nothing else helps: find interior point by sampling all points on a
            # circle with radius N with 1 degree difference
            else:
                angle = 0
                while (is_interior_point == False and angle < 361):
                    rot_seg = rotate(angle,seg2)
                    int_point = [cp.get_x() + N*rot_seg[0],cp.get_y() +N*rot_seg[1]]
                    test_point = Point(int_point)
                    angle +=1
                    if test_point.within(poly):
                        is_interior_point = True         
    return (int_point[0],int_point[1])
 
 
def createTriangleAdjacencyMatrix(all_triangles):
    return CcreateTriangleAdjacencyMatrix(list(all_triangles))
    
def buildTriangles(triangulation):
    points = list(triangulation.points)
    for p in points:
        p[0] = np.round(p[0])
        p[1] = np.round(p[1])
    triangle_point_indices = list(triangulation.elements)
    return CbuildTriangles(points, triangle_point_indices)
    
def bruteforcePruning(adjacency_matrix,triangles,order,verbose):
    return CbruteforcePruning(adjacency_matrix,np.asarray(triangles),order,verbose)
    
def cvDistanceMap(image):
    '''
    Wrapper for openCV's DistTransform function
    (see http://docs.opencv.org/modules/imgproc/doc/miscellaneous_transformations.html#cv.DistTransform)
    which sets all options with values that fit the processing of leaves.

    Parameter:
    ----------
        image: ndarray, image for which the distance map will be created, has to be single channel
            grayscale
        
    Returns:
    -------
        dst: ndarray, distance map of the image with precisely calculated distances
        
    Notes:
    -----
        The function on the openCV side only exists within cv and not cv2 so internally
        the ndarray will be converted to a CvMat and converted back to a ndarray at the end
    
    '''
    image = image.astype(np.uint8)
    image = cv.fromarray(image)
    dst = cv.CreateMat(image.rows,image.cols,cv.CV_32FC1)
    cv.DistTransform(image,dst,distance_type=cv.CV_DIST_L2,mask_size=cv.CV_DIST_MASK_PRECISE)
    dst = np.asarray(dst)
    return dst

def createGraph(adjacency_matrix,all_triangles,height):
    '''
    Creates a graph from the adjacency matrix and the radii and euclidean distances
    stored in the distance map and list of all triangles.
    
    Parameter:
    ----------
        adjacency_matrix: lil2 matrix, matrix containing triangle neighborhood
             relations
        all_triangles: list of triangle objects, list containing all triangles
             still present in the network
        distance_map: 2d-array, matrix containing the distance to the nearest
             background pixel for every foreground pixel.
             
    Returns:
    ---------
        G: networkx graph object, graph of the network created from the
            triangle adjacency matrix
    '''
    
    #create basic graph from neighborhood relations
    G = nx.Graph(adjacency_matrix)
    
    #extract and set x-coordinates of nodes from triangle centroids
    x = [triangle.get_centroid().get_x() for triangle in all_triangles]
    attr = dict(zip(np.arange(len(x)),x))
    nx.set_node_attributes(G,'x',attr)
    
    #extract and set y-coordinates of nodes from triangle get_centroid()s
    y = [triangle.get_centroid().get_y() for triangle in all_triangles]
    attr = dict(zip(np.arange(len(y)),y))
    nx.set_node_attributes(G,'y',attr)
    
    #extract triangle radii and set them as thickness for nodes
    radius_node = [triangle.get_radius() for triangle in all_triangles]
    attr = dict(zip(np.arange(len(radius_node)),radius_node))
    nx.set_node_attributes(G,'conductivity',attr)
    
    #set thickness of edges as mean over the thickness of the nodes it connects
    radius_edge = [(G.node[edge[0]]['conductivity'] + \
                    G.node[edge[1]]['conductivity'])/2.0 for edge in G.edges()]                 
    attr = dict(zip(G.edges(),radius_edge))
    nx.set_edge_attributes(G,'conductivity',attr)
    
    #set length of the edges
    length_edge = [math.sqrt((G.node[edge[0]]['x'] - G.node[edge[1]]['x'])**2 + \
                             (G.node[edge[0]]['y'] - G.node[edge[1]]['y'])**2 ) \
                             for edge in G.edges()]
    attr = dict(zip(G.edges(), length_edge))
    nx.set_edge_attributes(G,'weight',attr)
    
    y = [triangle.get_centroid ().get_y() for triangle in all_triangles]
    y = np.array(y)
    y = height - y
    y = list(y)
    attr = dict(zip(np.arange(len(y)),y))
    nx.set_node_attributes(G,'y',attr)
    
    return nx.connected_component_subgraphs(G)[0]
  
def _drawGraph(G,verbose):
    '''
        Draws the leaf network using the thickness-data stored in the edges
        edges of the graph and the coordinates stored in the nodes.
        
        Parameter:
        ----------
            G: networkx graph object, graph to be drawn
        
    '''
    start = time.clock()
    plt.figure()
    pos = {}
    for k in G.node.keys():
        pos[k] = (G.node[k]['x'], G.node[k]['y'])
    
    widths = np.array([G[e[0]][e[1]]['conductivity'] for e in G.edges()])
    widths = 15./(np.amax(widths)*2)*widths

    nx.draw_networkx_edges(G, pos=pos, width=widths,edgecolor='DarkSlateGray',
                           alpha=0.3)
    
    colors = {3:"orange",2:"purple",1:"red"}
    for node in G.nodes(data=True):
        x = node[1]['x']
        y = node[1]['y']
        typ = len(nx.neighbors(G,node[0]))
        if typ > 3:
            typ = 3
        c = colors[typ]
        plt.plot(x,y,'o',color=c,markersize=8,mec=c,alpha=0.8,mew=1)
        
    
    if verbose:
        print "\t from _drawGraph: drawing took %1.2f sec"%(time.clock()-start)
    
    
def removeRedundantNodes(G,verbose,mode):
    print "mode: ",mode
    '''
        Removes a specified number of redundant nodes from the graph. Nodes should
        be removed in place but the graph is returned nevertheless.
        
        Parameter:
        ---------
            G: networkx graph object, graph from which the nodes are removed
            mode: string, can be "all" or "half" which will remove either all
                nodes of degree 2 or half of the nodes of degree 2. If mode is
                something else, the function will return before the first
                iteration. Defaults to "all".
                
        Returns:
        ---------
            G: networkx graph object, reduced graph
    '''   
    if (type(G) == scipy.sparse.lil.lil_matrix):
        G = nx.Graph(G)
    order = G.order()
    new_order = 0    
    i = 0 
    
    while(True):
        if mode == 1:
            if i > 2:              
                break
        if mode == 0:
            if new_order == order:break
        if mode == 2:
            break
        
        nodelist = []
        for node in G.nodes():
            if(G.degree(node)==2):
                neighbors = G.neighbors(node)
                n1 = neighbors[0]
                n2 = neighbors[1]
                w1 = G.edge[node][n1]['weight']
                w2 = G.edge[node][n2]['weight']
                length = float(w1) + float(w2)
                c1 = G.edge[node][n1]['conductivity']
                c2 = G.edge[node][n2]['conductivity']
                radius = (c1*w1+c2*w2)/length
                G.add_edge(n1,n2, weight = length, conductivity = radius )
                if n1 not in nodelist and n2 not in nodelist:
                    nodelist.append(node)
        
        #sometimes when the graph does not contain any branches (e.g. one
        #single line) the redundant node removal ends up with a triangle at
        #the end, god knows why. This is a dirty hack to remove one of the 
        #three final nodes so a straight line is left which represents the 
        #correct topology but maybe not entirely correct coordinates.        
        if len(nodelist) == len(G.nodes())-1:
            G.remove_node(nodelist[1])
        else:
            for node in nodelist:
                G.remove_node(node)
                
        length = 0
        for edge in G.edges(data=True):
            length += edge[2]['weight']
        print "total network length", length
        
        order = new_order
        new_order = G.order() 
        if verbose:
            print "\t from removeRedundantNodes: collapsing iteration ",i
        i+=1
        if order == new_order:
            break
    return G
 
def drawAndSafe(G,image_name,dest,redundancy,verbose):
    '''
    Draws a graph calling the helper function _drawGraph and saves it at
    destination "dest" with the name "image_name" + "_graph"
    
    Parameter:
    ----------
        G: networkx graph object, graph to be drawn
        image_name: string, name of the file saved plus added "_graph" extension
        dest: string, destination where the drawn graph will be saved
    '''
    leaf = False
    start = time.clock()    
    if redundancy == 0: mode = "red0"   
    elif redundancy == 1: mode = "red1"
    else: mode = "red2"   
    if not leaf:
        plt.clf()
        _drawGraph(G,verbose)          
        plt.savefig(join(dest,image_name + "_graph_" + mode + ".png"),dpi=600)
    figure_save = time.clock()
    
    nx.write_gpickle(G,join(dest,image_name + "_graph_" + mode + ".gpickle"))
    graph_save = time.clock()
    if verbose:
        print "\t from drawAndSafe: figure saving took %1.2f sec"%(figure_save-start)
        print "\t from drawAndSafe: graph saving took %1.2f sec"%(graph_save-figure_save)
    plt.close()
  

def drawTriangulation(h,triangles,image_name,dest):
    def m(h,y):
        return h-y
    '''
    Draws and saves an illustration of the triangulation. Triangles are
    already classified in end-triangles (red), normal-triangles (purple)
    and junction-triangles (orange).
    
    Parameter:
    ---------
        triangle_classes: list of lists of end-, normal and junction triangles
        image_name: string, name used for saving the plot
        dest: string, destination at which the plot is saved
    '''
    plt.clf()
    ax = plt.gca()
    ax.set_aspect('equal', 'datalim')
    plt.title("Triangulation: " + image_name)
    colors = {"junction":["orange",3],"normal":["purple",1],"end":["red",2]}
    
    #normal triangles
    for t in triangles:
            x = [t.get_p1().get_x(),t.get_p2().get_x(),t.get_p3().get_x(),\
                      t.get_p1().get_x()]
            y = [m(h,t.get_p1().get_y()),m(h,t.get_p2().get_y()),\
                      m(h,t.get_p3().get_y()),m(h,t.get_p1().get_y())]
            
            c = colors[t.get_typ()][0]
            zorder = colors[t.get_typ()][1]
            plt.plot(x,y,'o',color=c,linewidth=1.5,zorder=zorder,
                     markersize=0.1,mew=0,alpha=1.0)
            plt.fill(x,y,facecolor=c,alpha=0.6,\
                     edgecolor=c,linewidth=0.05)
            
    plt.savefig(join(dest,image_name + "_triangulation.png"),dpi=2400)
    plt.close()
   
def drawContours(height,contour_list,image_name,dest): 
    from copy import deepcopy
    contour_list = deepcopy(contour_list)
    '''
    Draws and saves the list of contours it is provided with.
    
    Parameter:
    ---------
        thresholded_contours: list of lists of integers, list of contours
        image_name: string, name used for saving the plot
        dest: string, destination at which the plot is saved
    '''
    plt.clf()
    plt.title("Contours: " + image_name)
    ax = plt.gca()
    ax.set_aspect('equal', 'datalim')
    #flag to turn on plotting of coordinates next to every contour point
    coordinates = False
    
    #contours represented as circles for every point in the contour
    #connected by lines
    for c,i in zip(contour_list,xrange(len(contour_list))):
        c.append((c[0][0],c[0][1]))
        c = np.asarray(c)
        plt.text(c[0,0],height-c[0,1],str(i))
        plt.plot(c[0:,0],height-c[0:,1],color="black",marker="o", \
                 markersize=0.2,mfc="FireBrick", mew = 0.1, alpha=0.7, \
                 linewidth = 0.1)
                 
        #optional plotting of coordinates       
        if coordinates:
            for point in c:
                coordinate_string = "(" + str(point[0]) + "," + str(point[1]) + ")"
                plt.text(point[0],point[1],coordinate_string,fontsize=0.1)
                
    plt.savefig(join(dest,image_name + "_contours.png"),dpi=2400)
    plt.close()
 