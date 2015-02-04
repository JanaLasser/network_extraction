# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 13:34:10 2015

@author: jana
"""

### imports ###
# standard imports
import time
from os.path import join
import copy	
import ntpath
import argparse

# dependencies
import numpy as np	
import meshpy.triangle as triangle	
#import triangle_wrapper as triangle
import scipy.misc
from skimage.morphology import binary_opening, binary_closing
	
# helper functions
import vectorize_helpers2 as vh




"""
Argument handling:
Required: source
        The program will load the image specified in "source".
        The image needs to be an 8-bit image.
        If the image is not binary or even grayscale, the program will first
        convert it to grayscale and then perform basic thresholding to create
        a binary image.
        
Optional: destination
        The program will save all results in "destination" if it is specified.
        Otherwise results will get saved at destination of the source image.
        
Optional: redundancy
        When redundancy = 0 is specified, a graph-object with no redundant
        nodes (i.e. nodes with degree 2) will be safed.
        When redundancy = 1 is specified, a graph-object with half the 
        redundant nodes as well as one without redundant nodes will be safed.
        When redundancy = 2 is specified, a graph-object with all redundant
        nodes as well as one with half the redundant nodes and one with no 
        redundant nodes will be safed.
        
Optional: pruning
        If pruning is specified, a number of nodes equal to pruning will be
        removed at the edges of the graph. This is done to reduce the number
        of surplus branches due to noise in the contours of the network.
        Five has proven to be a reasonable number of nodes to prune away (also
        specified as default for this option).

Optional: contour_threshold
        If contour_threshold is specified, contours with a length (number of
        points) shorter than contour_threshold will be omitted. This is useful
        to get rid of noisy structures such as small holes in the features or
        small islands which were not removed by previous image processing.
        A reasonable value for contour_threshold is strongly dependent on the
        size and delicacy of the features we want to resolve. contour_threshold
        defaults to 3 which barely removes any contours and is feasible for
        rather low resolution images.

Optional: verbose
        When specified turns on verbose behaviour. The program will then print
        its current processing step as well as the time it needed for the last
        step. For some steps (usually the ones that take longer) information
        about the current state of the program is given in more detail to be 
        able to trace the program's progress. Defaults to True.

Optional: debug
        When specified turns on additional debug output. Also saves additional
        visualizations of the processing like a plot of the contours and the
        triangulation. May consume a considerable additional amount of time
        because the additional figures take quite long to plot and save.
        Defaults to False.
"""				

parser = argparse.ArgumentParser(description='Vectorize: a program that ' +\
    'extracts network information from a binary image containing ribbon-like,'\
    + ' connected shapes.')
parser.add_argument('source',type=str,help='Complete path to the image')
parser.add_argument('-dest', type=str, help='Complete path to the folder '\
                + 'results will be saved to if different than source folder')
parser.add_argument('-v','--verbose',action="store_true",\
                help='Turns on program verbosity', default=True)
parser.add_argument('-d','--debug',action="store_true",\
                help='Turns on debugging output',default=False)
parser.add_argument('-p','--pruning', type=int, help='Number of triangles that '\
                + 'will be pruned away at the ends of the graph', default=5)
parser.add_argument('-r','--redundancy', type=int, help='Sets the desired '\
                + 'number of redundant nodes in the output graph',\
                default=0,choices=[0,1,2])
parser.add_argument('-t','--contour_threshold', type=int, \
                help='Lenght up to which contours are discarded', default=3)
                
args = parser.parse_args()
verbose = args.verbose                                                         #verbosity switch turns on progress output
debug = args.debug                                                             #debugging switch enables debugging output 
image_source = args.source                                                     #full path to the image file
dest = args.dest                                                               #full path to the folder the results will be stored in
order = args.pruning                                                           #Order parameter for pruning: cut off triangles up to order at the ends of the graph
redundancy = args.redundancy                                                   #parameter for the number of redundant nodes in the final graph
contour_threshold = args.contour_threshold                                     #Contours shorter than threshold are discarded.

image_name = ntpath.basename(image_source).split('.')[0]
if dest == None:
    dest = image_source.split(image_name)[0]
if verbose:
    print "\n*** Starting to vectorize image %s ***"%image_name
    print "\nCurrent step: Image preparations"
start = time.clock()                                                           #start timer
previous_step = time.clock()                                                               
                                                     
image = vh.getImage(image_source).astype(np.uint8)                             #Load the image.
if image.ndim > 2:
    image = vh.RGBtoGray(image)                                                #In case it is not yet grayscale, convert to grayscale.  
image = np.where(image < 127,0,1)                                              #In case it is not yet binary, perform thresholding  

kernel = np.ones((3,3),np.int)                                                 #standard binary image noise-removal with opening followed by closing
image = binary_opening(image,kernel)                                           #maybe remove this processing step if depicted structures are really tiny
image = binary_closing(image,kernel)

distance_map = vh.cvDistanceMap(image).astype(np.int)                          #Create distance map
height,width = distance_map.shape
scipy.misc.toimage(distance_map, cmin=0, cmax=255)\
          .save(join(dest,image_name + "_dm.png"))                             #save the distance map in case we need it later

if debug:   
    scipy.misc.imsave(join(dest,image_name + "_processed.png"),image)          #debug output   
if verbose:													
    step = time.clock()                                                        #progress output
    print "Done in %1.2f sec."%(step-previous_step)
    print "\nCurrent step: Contour extraction and thresholding."
    previous_step = step
 
""" 
Contours:
        - Find the contours of the features present in the image.
          The contours are approximated using the Teh-Chin-dominant-point
          detection-algorithm (see Teh, C.H. and Chin, R.T., On the Detection
          of Dominant Pointson Digital Curve. PAMI 11 8, pp 859-872 (1989))
        - Find the longest contour within the set of contours
"""
raw_contours = vh.getContours(image)							     #Extract raw contours.   
flattened_contours = vh.flattenContours(raw_contours,height)			     #Flatten nested contour list

if debug:                                                                      #debug output
    print "\tContours converted, we have %i contours."\
           %(len(flattened_contours))
													
thresholded_contours = vh.thresholdContours(flattened_contours,\
    contour_threshold)                                                         #contours shorter than "threshold" are discarded.

if debug:                                                                      #debug output
    print "\tContours thresholded, %i contours left."\
           %(len(thresholded_contours)) 
    vh.drawContours(height,thresholded_contours,image_name,dest)

longest_index = 0											     #Position of longest contour.
longest_length = 0										     #Length of longest contour.
for c in xrange(len(thresholded_contours)):						     #Find index of longest contour.
    if(len(thresholded_contours[c])>longest_length):
        longest_length = len(thresholded_contours[c])
        longest_index = c

if verbose:       
    step = time.clock()                                                        #progress output
    print "Done in %1.2f sec."\
          %(step-previous_step)
    print "\nCurrent step: Mesh setup."
    previous_step = step
     
"""
Mesh Creation:
        - The mesh is created of points and facets where every facet is the
          plane spanned by one contour.

""" 
thresholded_contours = np.asarray(thresholded_contours)                        #add a bit of noise to increase stability of triangulation algorithm
for c in thresholded_contours:
    for p in c:
        p[0] = p[0] + 0.1*np.random.rand()
        p[1] = p[1] + 0.1*np.random.rand()
   
#mesh_points = copy.copy(thresholded_contours[longest_index])			     #First add longest contour to mesh.
mesh_points = thresholded_contours[longest_index]
mesh_facets = vh.roundTripConnect(0,len(mesh_points)-1)				     #Create facets from the longest contour.
hole_points = []  										     #Every contour other than the longest needs an interiour point.
for i in xrange(len(thresholded_contours)):						     #Traverse all contours. 
    curr_length = len(mesh_points)									
    if(i == longest_index):                                                    #Ignore longest contour.
        pass
    else:					                                             #Find a point that lies within the contour.
        contour = thresholded_contours[i]
        interior_point = vh.getInteriorPoint(contour)        										
        hole_points.append((interior_point[0],interior_point[1]))		     #Add point to list of interior points.
        mesh_points.extend(contour)							     #Add contours identified by their interior points to the mesh.
        mesh_facets.extend(vh.roundTripConnect(curr_length,len(mesh_points)-1))#Add facets to the mesh

if verbose:
    step = time.clock()                                                        #progress output
    print "Done in %1.2f sec."%(step-previous_step)
    print "\nCurrent step: Triangulation."       
    previous_step = step

"""
Triangulation:
        - set the points we want to triangulate
        - mark the holes we want to ignore by their interior points
        - triangulation: no interior steiner points, we want triangles to fill
          the whole space between two boundaries. Allowing for quality meshing
          would also mess with the triangulation we want.
"""
info = triangle.MeshInfo()									     #Create triangulation object.
info.set_points(mesh_points)									     #Set points to be triangulated.
if(len(hole_points) > 0):									     
	info.set_holes(hole_points)                                              #Set holes (contours) to be ignored.
	info.set_facets(mesh_facets)       						     #Set facets.
triangulation = triangle.build(info,verbose=False,allow_boundary_steiner=False,#Build Triangulation.
       allow_volume_steiner=False,quality_meshing=False)

if verbose:
    step = time.clock()                                                        #progress output
    print "Done in %1.2f sec."%(step-previous_step)
    print "\nCurrent step: Setup of triangles and neighborhood relations."
    previous_step = step

"""
Triangle classification:
        - build triangle-objects from the triangulation
        - set the type of each triangle (junction, normal, end or isolated)
          depending on how many neighbors it has
        - set the diameter of each triangle by looking up its "midpoint"
          in the distance map
        - get rid of isolated triangles
"""

triangles = vh.buildTriangles(triangulation)	                                 #Build triangles                                                                 
junction = 0
normal = 0
end = 0
isolated_indices = []
default_triangles = 0
for i in range(len(triangles)):                                                
    t = triangles[i]
    default_triangles += t.set_typ(distance_map)                               #set the triangle's type, midpoint and diameter
    if t.get_typ() == "junction":                                              #count the number of each triangle type for debugging
        junction += 1
    elif t.get_typ() == "normal":
        normal += 1
    elif t.get_typ() == "end":
        end += 1
    elif t.get_typ() == "isolated":
        isolated_indices.append(i)    
triangles = list(np.delete(np.asarray(triangles), isolated_indices))           #remove isolated triangles from the list of triangles

if debug:                                                                      #debug output
    vh.drawTriangulation(height,triangles,image_name,dest)
    print "\tTriangle types:"
    print "\tjunction: %d, normal: %d, end: %d, isolated: %d"\
                %(junction,normal,end,len(isolated_indices))
    print "\ttriangle diameters defaulted to 1.0: ",default_triangles
if verbose:	           
    step = time.clock()                                                        #progress output
    print "Done in %1.2f sec." %(step-previous_step)
    print "\nCurrent step: Creation of triangle adjacency matrix." 
    previous_step = step

"""
Triangle adjacency matrix:
        - create a matrix of triangle neighborhoods with distances between
          triangles (euclidean) as weights
        - create a copy of the adjacency matrix for the pruning process

"""
adjacency_matrix = vh.createTriangleAdjacencyMatrix(triangles)       	     #Create an adjacency matrix of all triangles with euclidean distances between triangles as link weights 

if verbose:                        									    					     
    step = time.clock()                                                        #progress output
    print "Done in %1.2f sec."%(step-previous_step)
    print "\nCurrent step: Bruteforce Pruning."
    previous_step = step																					

"""
Graph creation and improvement
        - prune away the outermost branches to avoid surplus branches due
          to noisy contours
        - create a graph object from the neighborhood relations, coordinates
          and diameter stored in the adjacency matrix and
          the list of triangles.
"""
adjacency_matrix,triangles = vh.bruteforcePruning(adjacency_matrix,\
                             triangles,order,verbose)                          #prune away the "order" number of triangles at the ends of the network
if verbose:
    step = time.clock()                                                        #progress output
    print "Done in %1.2f sec."%(step-previous_step)
    print "\nCurrent step: Graph creation."
    previous_step = step

G = vh.createGraph(adjacency_matrix,triangles,height)                          #creation of graph object 

if verbose:
    step = time.clock()                                                        #progress output
    print "Done in %1.2f sec."%(step-previous_step)
    print "\nCurrent step: Removal of redundant nodes,"\
          + " drawing and saving of the graph."
    previous_step = step

"""
Redundant node removal
        - if so specified, remove half the redundant nodes (i.e. nodes with
          degree 2), draw and save the graph
        - if so specified, remove all the redundant nodes, draw and save the 
          graph
"""
if redundancy == 2: 
    vh.drawAndSafe(G,image_name,dest,redundancy,verbose)                       #draw and safe graph with redundant nodes                         

if redundancy == 1:                                                            #draw and safe graph with half redundant nodes
    G = vh.removeRedundantNodes(G,verbose, redundancy)
    vh.drawAndSafe(G,image_name,dest,redundancy,verbose)
    
G = vh.removeRedundantNodes(G,verbose)                                         #draw and safe graph without redundant nodes
vh.drawAndSafe(G,image_name,dest,redundancy,verbose)										

if verbose:
    step = time.clock()                                                        #progress output
    print "Done in %1.2f sec."%(step-previous_step)
    previous_step = step	
    end = time.clock()                                                             
    print "\nALL DONE! Total time: %1.2f sec"%(end-start)

