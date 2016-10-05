'''
    Copyright (C) 2015  Jana Lasser Max Planck Institute for Dynamics and
    Self Organization Goettingen

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    If you find any bugs or have any suggestions, please contact me at
    jana.lasser@ds.mpg.de
'''
#path handling
from os.path import join
import os
import sys

print sys.argv[0]
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../net'))
if not path in sys.path:
    sys.path.insert(1, path)
del path


#standard imports
from os.path import join
import time
import math

#depenencies
import numpy as np
import scipy
import networkx as nx
#import matplotlib 
#matplotlib.use('qt4agg') 
import matplotlib.pyplot as plt 
from PIL import Image
import cv2

#custom classes and functions from the cythonized helper library
from C_net_functions import CbuildTriangles, CbruteforcePruning
from C_net_functions import CcreateTriangleAdjacencyMatrix, Cpoint

#global switches
edgesize = 0.5
plt.ioff()                                                                     #turn off matplotlib interactive mode, we safe everything we plot anyways

#functions used in the vectorize.py script
def getImage(image_path):
    '''
    Helper for opening an image.

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
    Helper for converting an image to grayscale and converting it to
    a numpy array filled with floats.

    Parameters
    ----------
        image_path: string
            path to the file containing the image
        
    Returns
    -------
        image: ndarray
            array containing the image data read from the file
        
    '''
    if (image.ndim == 2): return image      
    else:
        image = image.astype(np.float32).sum(axis=2);image/=3.0
        return image
        
def getContours(image):
    '''
    Wrapper around openCV's cv2.findContours() function
    (see: http://docs.opencv.org/modules/imgproc/doc/
        structural_analysis_and_shape_descriptors.html#cv2.findContours)
    Sets convenient options and converts contours to a list of ndarrays.
    
    Parameter:
    ---------
        image: ndarray, input image
        
    Returns:
    --------
        contours: list, list of contours
    '''
    
    image = image.astype(np.uint8)

    if cv2.__version__ < '3.1.0':
        contours,hirachy = cv2.findContours(image,cv2.RETR_CCOMP,\
                                        cv2.CHAIN_APPROX_TC89_L1)
    else:
        image, contours,hirachy = cv2.findContours(image,cv2.RETR_CCOMP,\
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
     
    #check if the center of the contour already qualifies as interior point
    x = 0
    y = 0
    for point in contour:
        x += point[0]
        y += point[1]
    x /= len(contour)
    y /= len(contour)
    
    center = Point(x,y)
    if center.within(poly):
        return [x,y] 
    
    #if the center is no good, invoke a more sofisticated method
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
    #inside the polygon and if not, take 360-phi and find another point with
    #this angle
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
            
            #if nothing else helps: find interior point by sampling all points
            #on a circle with radius N with 1 degree difference
            else:
                angle = 0
                while (is_interior_point == False and angle < 361):
                    rot_seg = rotate(angle,seg2)
                    int_point = [cp.get_x() + N*rot_seg[0],
                                 cp.get_y() +N*rot_seg[1]]
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
                
            #if nothing else helps: find interior point by sampling all points
            #on a circle with radius N with 1 degree difference
            else:
                angle = 0
                while (is_interior_point == False and angle < 361):
                    rot_seg = rotate(angle,seg2)
                    int_point = [cp.get_x() + N*rot_seg[0],\
                                 cp.get_y() + N*rot_seg[1]]
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
    
def bruteforcePruning(triangles,order,verbose):
    return CbruteforcePruning(np.asarray(triangles),order,verbose)
    
def cvDistanceMap(image):
    '''
    Wrapper for openCV's distanceTransform function
    (see http://docs.opencv.org/modules/imgproc/doc/
            miscellaneous_transformations.html#cv2.distanceTransform)
    which sets all options to values that fit the processing of leaves.

    Parameter:
    ----------
        image: ndarray, image for which the distance map will be created, has
               to be single channel grayscale
        
    Returns:
    -------
        dst: ndarray, distance map of the image with precisely calculated
             distances
    '''
    image = image.astype(np.uint8)
    
    #distanceType = 2 corresponds to the old cv.CV_DIST_L2
    #maskSize = 0 corresponds to the old cv.CV_DIST_MASK_PRECISE 
    dst = cv2.distanceTransform(image,distanceType=2,\
                     maskSize=0)
    return dst

def createGraph(adjacency_matrix,all_triangles,height):
    '''
    Creates a graph from the adjacency matrix and the radii and euclidean
    distances stored in the distance map and list of all triangles.
    
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
    
    #extract and set x-coordinates of nodes from triangle centers
    x = [triangle.get_center().get_x() for triangle in all_triangles]
    attr = dict(zip(np.arange(len(x)),x))
    nx.set_node_attributes(G,'x',attr)
    
    #extract and set y-coordinates of nodes from triangle centers
    y = [triangle.get_center().get_y() for triangle in all_triangles]
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
    length_edge = [math.sqrt((G.node[edge[0]]['x']-G.node[edge[1]]['x'])**2 + \
                             (G.node[edge[0]]['y']-G.node[edge[1]]['y'])**2 ) \
                             for edge in G.edges()]
    attr = dict(zip(G.edges(), length_edge))
    nx.set_edge_attributes(G,'weight',attr)
    
    y = [triangle.get_center().get_y() for triangle in all_triangles]
    y = np.array(y)
    y = height - y
    y = list(y)
    attr = dict(zip(np.arange(len(y)),y))
    nx.set_node_attributes(G,'y',attr)
    
    return G


def removeRedundantNodes(G,verbose,mode):
    '''
        Removes a specified number of redundant nodes from the graph. Nodes
        should be removed in place but the graph is returned nevertheless.
        
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
        
        nodelist = {}
        for node in G.nodes():
            neighbors = G.neighbors(node)
            if(len(neighbors)==2):
                n1 = neighbors[0]
                n2 = neighbors[1]
                w1 = G.edge[node][n1]['weight']
                w2 = G.edge[node][n2]['weight']
                length = float(w1) + float(w2)
                c1 = G.edge[node][n1]['conductivity']
                c2 = G.edge[node][n2]['conductivity']
                #TODO: figure out why length can be zero
                if length == 0: length = 1
                radius = (c1*w1+c2*w2)/length
                G.add_edge(n1,n2, weight = length, conductivity = radius )
                if n1 not in nodelist and n2 not in nodelist:
                    nodelist.update({node:node})
        
        #sometimes when the graph does not contain any branches (e.g. one
        #single line) the redundant node removal ends up with a triangle at
        #the end, god knows why. This is a dirty hack to remove one of the 
        #three final nodes so a straight line is left which represents the 
        #correct topology but maybe not entirely correct coordinates.        
        if len(nodelist) == len(G.nodes())-1:
            G.remove_node(nodelist[1])
        else:
            for node in nodelist.keys():
                G.remove_node(node)
        
        order = new_order
        new_order = G.order() 
        if verbose:
            print "\t from removeRedundantNodes: collapsing iteration ",i
        i+=1
        if order == new_order:
            break
    return G


def drawGraphTriangulation(h,G,triangles,image_name,dest,distance_map,\
        figure_format,dpi):

    def m(h,y):
        return h - y

    plt.clf()
    ax = plt.gca()
    ax.set_aspect('equal', 'datalim')
    distance_map = np.where(distance_map > 0,(distance_map)**1.5 + 20,0)
    ax.imshow(np.abs(np.flipud(distance_map)-255),cmap='gray')
    plt.title("Triangulation: " + image_name)
    colors = {"junction":["orange",3],"normal":["purple",1],"end":["red",2]}
    
    #normal triangles
    for i,t in enumerate(triangles):
            x = [t.get_p1().get_x(),t.get_p2().get_x(),t.get_p3().get_x(),\
                      t.get_p1().get_x()]
            y = [m(h,t.get_p1().get_y()),m(h,t.get_p2().get_y()),\
                      m(h,t.get_p3().get_y()),m(h,t.get_p1().get_y())]
            
            c = colors[t.get_type()][0]
            zorder = colors[t.get_type()][1]
            ax.plot(x,y,'o',color='black',linewidth=3,zorder=zorder,
                     markersize=0.3,mew=0,alpha=1.0)
            ax.fill(x,y,facecolor=c,alpha=0.25,\
                     edgecolor=c,linewidth=0.05)
            #ax.text((x[0]+x[1]+x[2])/3.0-0.5,(y[0]+y[1]+y[2])/3.0-0.5,"%d"%i,\
            #        fontsize=1)
    scale = 1 
    pos = {}
    for k in G.node.keys():
        pos[k] = (G.node[k]['x']*scale, G.node[k]['y']*scale)
    
    widths = np.array([G[e[0]][e[1]]['conductivity'] for e in G.edges()])*scale
    widths = 15./(np.amax(widths)*13)*widths

    nx.draw_networkx_edges(G, pos=pos, width=widths,edgecolor='DimGray',
                           alpha=0.4)
    
    colors = {3:"green",2:"green",1:"green"}
    for node in G.nodes(data=True):
        x = node[1]['x']*scale
        y = node[1]['y']*scale
        typ = len(nx.neighbors(G,node[0]))
        if typ > 3:
            typ = 3
        if typ < 4:
            c = colors[typ]
            plt.plot(x,y,'+',color=c,markersize=0.3,mec=c)       
    plt.savefig(join(dest,image_name + "_graph_and_triangulation" + "." + \
                figure_format,bbox_inches='tight'), dpi=dpi)
    

  
def _drawGraph(G,verbose,n_size):
    '''
        Draws the leaf network using the thickness-data stored in the edges
        edges of the graph and the coordinates stored in the nodes.
        
        Parameter:
        ----------
            G: networkx graph object, graph to be drawn
        
    '''
    start = time.clock()
    #plt.axis('off')
    #scale = 0.3459442
    scale = 1
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.axis('off')
    pos = {}
    for k in G.node.keys():
        pos[k] = (G.node[k]['x']*scale, G.node[k]['y']*scale)
    
    
    
    edgelist = [(e[0],e[1]) for e in G.edges(data=True) if e[2]['weight']<1000]
    widths = np.array([G[e[0]][e[1]]['conductivity'] for e in edgelist])*scale
    widths = 15./(np.amax(widths)*2)*widths*edgesize

    nx.draw_networkx_edges(G, pos=pos,width=widths,edge_color='DarkSlateGray',\
        edgelist=edgelist)

    #TODO: find out why degrees can be > 3!    
    color_dict = {3:"orange",2:"purple",1:"red"}
    colors = []
    for node in G.nodes():
        degree = nx.degree(G,node)
        if degree > 3:
            degree = 3
        colors.append(color_dict[degree])
    
    nx.draw_networkx_nodes(G, pos=pos,alpha=1,node_size=n_size,\
        node_color=colors,linewidths=0,node_shape='o')            
    
    if verbose:
        print "\t from _drawGraph: drawing took %1.2f sec"%(time.clock()-start)
 
def drawAndSafe(G,image_name,dest,parameters,verbose,plot,figure_format,dpi,\
                graph_format,n_size):
    '''
    Draws a graph calling the helper function _drawGraph and saves it at
    destination "dest" with the name "image_name" + "_graph"
    
    Parameter:
    ----------
        G: networkx graph object, graph to be drawn
        image_name: string, name of the file saved plus added "_graph"
        extension dest: string, destination where the drawn graph will be saved
    '''
    start = time.clock() 
    #failsafe to remove disconnected nodes
    G = max(nx.connected_component_subgraphs(G), key=len)
    graph_name = image_name + '_graph'
    for key,value in zip(parameters.keys(),parameters.values()):
        graph_name += '_' + key + str(value)    

    if plot:
        plt.clf()
        _drawGraph(G,verbose,n_size)          
        plt.savefig(join(dest,graph_name + "." + figure_format), \
            dpi=dpi,bbox_inches='tight')
        
    figure_save = time.clock()

    save_function_dict = {'gpickle':[nx.write_gpickle, '.gpickle'],
                          'adjlist':[nx.write_adjlist,'.adjlist'],
                          'gml': [nx.write_gml,'.gml'],
                          'graphml':[nx.write_graphml,'graphml'],
                          'edgelist':[nx.write_edgelist,'.edgelist'],
                          'yaml': [nx.write_yaml,'.yaml'],
                          'weighted_edgelist': [nx.write_weighted_edgelist,'.edgelist'],
                          'multiline_adjlist': [nx.write_multiline_adjlist,'.adjlist'],
                          'gexf': [nx.write_gexf,'.gexf'],
                          'pajek': [nx.write_pajek,'.net']}

    writefunc = save_function_dict[graph_format][0]
    writeformat = save_function_dict[graph_format][1]
    if graph_format in save_function_dict:
        if graph_format == 'graphml' or graph_format == 'gexf':
            G = _convertNumbers(G)
        writefunc(G, join(dest, graph_name + writeformat))    
    else:
        print "unknown graph format!"
    
    graph_save = time.clock()
    if verbose:
        print "\t from drawAndSafe: figure saving took %1.2f sec"\
              %(figure_save-start)
        print "\t from drawAndSafe: graph saving took %1.2f sec"\
              %(graph_save-figure_save)
    plt.close()
    
def _convertNumbers(G):
    for n in G.nodes(data=True):
        n[1]['x'] = float(n[1]['x'])
        n[1]['y'] = float(n[1]['y'])
        n[1]['conductivity'] = float(n[1]['conductivity'])
    for e in G.edges(data=True):
        e[2]['weight'] = float(e[2]['weight'])
        e[2]['conductivity'] = float(e[2]['conductivity'])
    return G
    
  

def drawTriangulation(h,triangles,image_name,dest,distance_map, \
            figure_format,dpi):
    def m(h,y):
        return y
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
    #ax.imshow(distance_map,cmap='gray')
    plt.title("Triangulation: " + image_name)
    colors = {"junction":["orange",3],"normal":["purple",1],"end":["red",2]}
    
    #normal triangles
    for i,t in enumerate(triangles):
            x = [t.get_p1().get_x(),t.get_p2().get_x(),t.get_p3().get_x(),\
                      t.get_p1().get_x()]
            y = [m(h,t.get_p1().get_y()),m(h,t.get_p2().get_y()),\
                      m(h,t.get_p3().get_y()),m(h,t.get_p1().get_y())]
            
            c = colors[t.get_type()][0]
            zorder = colors[t.get_type()][1]
            ax.plot(x,y,'o',color=c,linewidth=3,zorder=zorder,
                     markersize=0.1,mew=0,alpha=1.0)
            ax.fill(x,y,facecolor=c,alpha=0.45,\
                     edgecolor=c,linewidth=0.05)
            #ax.text((x[0]+x[1]+x[2])/3.0-0.5,(y[0]+y[1]+y[2])/3.0-0.5,"%d"%i,\
            #        fontsize=1)
            
    plt.savefig(join(dest,image_name + "_triangulation" + '.pdf'),\
                     dpi=dpi)
    plt.close()
   
def drawContours(height,contour_list,image_name,dest,figure_format,dpi): 
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
        plt.plot(c[0:,0],height-c[0:,1],color="black",marker="o", \
                 markersize=2,mfc="FireBrick", mew = 0.1, alpha=0.7, \
                 linewidth = 1)
                 
        #optional plotting of coordinates       
        if coordinates:
            for point in c:
                coordinate_string = "(" + str(point[0]) + "," + str(point[1]) + ")"
                plt.text(point[0],point[1],coordinate_string,fontsize=0.1)
                
    plt.savefig(join(dest,image_name + "_contours" + '.' + figure_format),\
                dpi=dpi)
    plt.close()
 