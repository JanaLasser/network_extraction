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

### imports ###
# standard imports
import os
import sys
import time
from os.path import join
import ntpath
import argparse

# dependencies
import numpy as np	
import meshpy.triangle as triangle	
import scipy.misc
from skimage.morphology import binary_opening, binary_closing, disk
from skimage.morphology import remove_small_objects

#path handling
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../net'))
if not path in sys.path:
    sys.path.insert(1, path)
del path
	
#custom helper functions
import net_helpers as nh

"""
Argument handling:
------------------

Required: source
        The program will load the image specified in "source".
        The image needs to be an 8-bit image.
        If the image is not binary or even grayscale, the program will first
        convert it to grayscale and then perform basic thresholding to create
        a binary image.
        
Optional: destination -dest
        The script will save all results in "destination" if it is specified.
        Otherwise results are saved at the destination of the source image.
        
Optional: redundancy -r
        When redundancy = 0 is specified, a graph-object with no redundant
        nodes (i.e. nodes with degree 2) will be safed.
        When redundancy = 1 is specified, a graph-object with half the 
        redundant nodes as well as one without redundant nodes will be safed.
        When redundancy = 2 is specified, a graph-object with all redundant
        nodes as well as one with half the redundant nodes and one with no 
        redundant nodes will be safed.
        
Optional: pruning -p
        If pruning is specified, branches shorter than the threshold specified
        with pruning will be removed from the network. This is done to reduce
        the number of surplus branches due to noisy contours of the network.
        Defaults to 5 which has proven to be a reasonable number of nodes 
        to prune away.
        
Optional: minimum_feature_size -m
        If minimum_feature_size in pixels is specified, features with a size of
        up to minimum_feature_size will be removed from the binary image
        previous to processing. This serves to a) reduce the number of contours
        which will be processed and not needed in the end and b) helps the
        triangulation algorithm to not get confused when there are islands
        within holes.
        Defaults to 3000 pixels.

Optional: smoothing -s
        If smoothing is enabled, the binary image will be smoothed
        using binary opening and closing operations. The kernel size of said
        operations can be controlled by passing an integer along with the
        smoothing option. The right kernel size is highly dependant on
        the resolution of the image. As a rule of thumb choose a kernel size
        below the width of the smallest feature in the image (in pixels).
        Defaults to 3 pixels.

Optional: verbose -v
        When specified turns on verbose behaviour. The program will then print
        its current processing step as well as the time it needed for the last
        step. For some steps (usually the ones that take longer) information
        about the current state of the program is given in more detail to be 
        able to trace the program's progress. Defaults to True.

Optional: debug -d
        When specified turns on additional debug output. Also saves additional
        visualizations of the processing like a plot of the contours and the
        triangulation. May consume a considerable additional amount of time
        because the additional figures take quite long to plot and save.
        Defaults to False.
        
Optional: plot -plt
        When specified enables visualization (plotting) of the created graphs.
        The visualization is done using matplotlib and in general gets very
        slow when dealing with large networks. As a rule of thumb it is 
        encouraged to only plot networks when they contain fewer than 10^5
        nodes. The format the plot will be saved in as well as its resolution
        (if not saved as vector graphic) can be specified using the -format
        and -dpi commands.
        If you want to visualize really large networks, saving them as .png
        with a high dpi (>2000 so features are still recognizeable) might work.
        Per default, visualization is disabled to safe time.

Optional: figure_format -fformat
        Sets the output format of plots (networks or debugging plots) to the
        desired format. Supported formats include: pdf, png, jpg, tif
        (look at matplotlib's safefig-function for a complete list).
        In general if the plot is needed for a report/publication, pdf is
        recommended as they can easily be modified and beautified using a
        vector graphics editor. Saving as pdf is relatively time-inefficient.
        If one wants to visualize large plots, I would recommend saving them
        as png with a high resolution (use dpi option to crank up resolution).
        This saves time and yields acceptable results.
        Defaults to pdf.
        
Optional graph_format -gformat:
        Sets the output format of graphs to the desired formad. Supported 
        formats are [gpickle, adjlist, gml, graphml, edgelist, yaml,
        weighted_edgelist, multiline_adjlist, gexf, pajek]. Have a look at the
        different networkx.write_ ... functions for an in detail explanation.
        Default is .gpickle wich is a dump of a python object which can easily
        be read again using networkx.read_gpickle.

Optional: resolution -dpi
        Sets the resolution (dots per inch) for plots (networks or debugging
        plots). High resolutions take longer to save and create larger image
        files. Low resolutions sometimes make it hard to recognize smaller
        details in the networks. 
        Defaults to 500 dpi which is a good compromise most of the times.

Optional: node_size -n
        Controls the size of the nodes displayed in the visualization of the
        graph. If node_size = 0, only edges and no nodes are plotted.
        Defaults to 4 which turned out to be a good value for nice plots of
        tracheoles.

Optional: distance_map -dm
        Enables saving of the euclidean distance map created during the
        network extraction process. This is useful if the network will later
        be loaded and manipulated using GeGUI as the GUI needs the distance
        map to determine radii of network edges.
        Defaults to False.

Optional: gegui -gegui
        Enables gegui mode. This is especially helpful if you plan on using the
        extracted networks within gegui later on. With this option enabled NET
        will create a separate folder for the extracted graph and save the graph
        and its distance map in this folder (this is the file-structure gegui
        requires to run). NOTE that you still have to put the original file into
        the newly created folder if you want geguis overlay functionality to
        work properly!
"""				

parser = argparse.ArgumentParser(description='Net (Network Extraction '\
    + 'Tool): a program that ' +\
    'extracts network information from a binary image containing ribbon-like,'\
    + ' connected shapes.\nCopyright (C) 2015 Jana Lasser')

parser.add_argument('source',type=str,help='Complete path to the image')

parser.add_argument('-dest', type=str, help='Complete path to the folder '\
                + 'results will be saved to if different than source folder')

parser.add_argument('-v','--verbose',action="store_true",\
                help='Turns on program verbosity', default=False)

parser.add_argument('-d','--debug',action="store_true",\
                help='Turns on debugging output',default=False)

parser.add_argument('-p','--pruning', type=int, help='Number of triangles that '\
                + 'will be pruned away at the ends of the graph', default=5)

parser.add_argument('-r','--redundancy', type=int, help='Sets the desired '\
                + 'number of redundant nodes in the output graph',\
                default=0,choices=[0,1,2])

parser.add_argument('-m','--minimum_feature_size', type=int, \
                help='Minimum size (pixels) up to which features will be ' + \
                'discarded.', default = 3000)

parser.add_argument('-s','--smoothing',type=int,\
                    help='Turns on image smoothing via binary opening and '+\
                    'closing. Sets kernel size of the morphology operators '+\
                    'to argument value.', default=False)

parser.add_argument('-plt','--plot',action='store_true',default=False,\
                    help='Enables plotting')

parser.add_argument('-fformat','--figure_format',type=str,\
                    help='Specifies the file format of plots.',\
                    default='pdf')
                    
parser.add_argument('-gformat','--graph_format',type=str,\
                    help='Specifies the file format of graphs.',\
                    default='gpickle',choices=['gpickle','adjlist','edgelist',\
                    'dot','gml','graphml','gexf','shp','yaml','pajek',\
                    'weighted_edgelist','multiline_adjlist'])

parser.add_argument('-dpi', '--resolution', type=int, help='Sets dpi' +\
         '(dots per inch) for plots.', default = 500)

parser.add_argument('-n', '--node_size', type=float, help='Sets size of nodes' +\
         ' for plots.', default = 4)

parser.add_argument('-dm', '--distance_map',action="store_true", \
            help = "Saves the distance map.",default=False)

parser.add_argument('-gegui', '--gegui', action="store_true", \
            help= "Enable this when planning to use gegui with the extracted" +
            "graphs: it will create a separate folder for the result in the" +
            "target directory and save all the files gegui needs correclty" +
            "named in the new folder. NOTE: you will still need to copy the" +
            "original (non-binary) image to the newly created folder if you"
            + "want to superimpose the graph onto it!")
                
args = parser.parse_args()
verbose = args.verbose                                                         #verbosity switch enables printing of progress output
debug = args.debug                                                             #debugging switch enables debugging output 
image_source = args.source                                                     #full path to the image file
dest = args.dest                                                               #full path to the folder the results will be stored in
order = args.pruning                                                           #Order parameter for pruning: remove branches shorter than specified threshold
redundancy = args.redundancy                                                   #parameter for the number of redundant nodes in the final graph
contour_threshold = 5                                                          #Contours shorter than threshold are discarded.
minimum_feature_size = args.minimum_feature_size                               #features smaller than minimum_feature_size will be discarded.
smoothing = args.smoothing                                                     #enables smoothing via binary opening and closing on and off
plot = args.plot                                                               #enables visualization of extracted networks
figure_format = args.figure_format                                             #specifies desired format to save plots
graph_format = args.graph_format
dpi = args.resolution                                                          #specifies resolution of plots (will be ignored if figure format is pdf)               
n_size = args.node_size
save_distance_map = args.distance_map                                          #enables saving of the euclidean distance map
gegui = args.gegui

parameters = {'r':redundancy, 'p':order}

pa = "NET> "                                                                   #preamble

image_name = ntpath.basename(image_source).split('.')[0]
image_name = image_name.split('_binary')[0]

if dest == None:
    dest = image_source.split(image_name)[0]

if gegui:
    save_distance_map = True
    dest = join(dest,image_name)
    if os.path.exists(dest):
        pass
    else:
        os.mkdir(dest)

if verbose:
    print "\n" + pa + "*** Starting to vectorize image %s ***"%image_name
    print "\n" + pa + "Current step: Image preparations"
start = time.clock()                                                           #start timer
previous_step = time.clock()                                                               
                                                     
image = nh.getImage(image_source).astype(np.uint8)                             #Load the image.
if image.ndim > 2:
    image = nh.RGBtoGray(image)                                                #In case it is not yet grayscale, convert to grayscale.  
image = np.where(image < 127,0,1)                                              #In case it is not yet binary, perform thresholding  

image = remove_small_objects(image.astype(bool),\
                                 min_size=minimum_feature_size,connectivity=1)
if smoothing:                                                                  #standard binary image noise-removal with opening followed by closing
    image = binary_opening(image,disk(smoothing))                              #maybe remove this processing step if depicted structures are really tiny
    image = binary_closing(image,disk(smoothing))

if minimum_feature_size:
    image = remove_small_objects(image.astype(bool),\
                                 min_size=minimum_feature_size,connectivity=1)

distance_map = nh.cvDistanceMap(image).astype(np.int)                          #Create distance map
height,width = distance_map.shape
if save_distance_map:
    scipy.misc.toimage(distance_map, cmin=0, cmax=255)\
              .save(join(dest,image_name + "_dm.png"))                         #save the distance map in case we need it later

if debug:   
    scipy.misc.imsave(join(dest,image_name + "_processed.png"),image)          #debug output   
if verbose:													
    step = time.clock()                                                        #progress output
    print pa + "Done in %1.2f sec."%(step-previous_step)
    print "\n" + pa + "Current step: Contour extraction and thresholding."
    previous_step = step
 
""" 
Contours:
        - Find the contours of the features present in the image.
          The contours are approximated using the Teh-Chin-dominant-point
          detection-algorithm (see Teh, C.H. and Chin, R.T., On the Detection
          of Dominant Pointson Digital Curve. PAMI 11 8, pp 859-872 (1989))
        - Find the longest contour within the set of contours
"""
raw_contours = nh.getContours(image)							               #Extract raw contours.   
flattened_contours = nh.flattenContours(raw_contours)       			       #Flatten nested contour list

if debug:                                                                      #debug output
    print pa + "\tContours converted, we have %i contour(s)."\
           %(len(flattened_contours))
													                                                        #contours shorter than "threshold" are discarded.
flattened_contours = nh.thresholdContours(flattened_contours,3)                #filter out contours smaller than 3 in case there are any left

if debug:                                                                      #debug output
    nh.drawContours(height,flattened_contours,image_name,dest,\
            figure_format,dpi)

longest_index = 0											                   #Position of longest contour.
longest_length = 0										                       #Length of longest contour.
for c in xrange(len(flattened_contours)):						               #Find index of longest contour.
    if(len(flattened_contours[c])>longest_length):
        longest_length = len(flattened_contours[c])
        longest_index = c

if verbose:       
    step = time.clock()                                                        #progress output
    print pa + "Done in %1.2f sec."\
          %(step-previous_step)
    print "\n" + pa + "Current step: Mesh setup."
    previous_step = step
     
"""
Mesh Creation:
        - The mesh is created of points and facets where every facet is the
          plane spanned by one contour.
""" 
flattened_contours = np.asarray(flattened_contours)                            #add a bit of noise to increase stability of triangulation algorithm
for c in flattened_contours:
    for p in c:
        p[0] = p[0] + 0.1*np.random.rand()
        p[1] = p[1] + 0.1*np.random.rand()
   			     
mesh_points = flattened_contours[longest_index]                                #First add longest contour to mesh.
mesh_facets = nh.roundTripConnect(0,len(mesh_points)-1)				           #Create facets from the longest contour.

hole_points = []  										                       #Every contour other than the longest needs an interiour point.
for i in xrange(len(flattened_contours)):						               #Traverse all contours.   
    curr_length = len(mesh_points)									
    if(i == longest_index):                                                    #Ignore longest contour.
        pass
    else:					                                                   #Find a point that lies within the contour.
        contour = flattened_contours[i]
        interior_point = nh.getInteriorPoint(contour)        										
        hole_points.append((interior_point[0],interior_point[1]))		       #Add point to list of interior points.
        mesh_points.extend(contour)							                   #Add contours identified by their interior points to the mesh.
        mesh_facets.extend(nh.roundTripConnect(curr_length,len(mesh_points)-1))#Add facets to the mesh

if verbose:
    step = time.clock()                                                        #progress output
    print pa + "Done in %1.2f sec."%(step-previous_step)
    print"\n" + pa + "Current step: Triangulation."       
    previous_step = step

"""
Triangulation:
        - set the points we want to triangulate
        - mark the holes we want to ignore by their interior points
        - triangulation: no interior steiner points, we want triangles to fill
          the whole space between two boundaries. Allowing for quality meshing
          would also mess with the triangulation we want.
"""
info = triangle.MeshInfo()									                   #Create triangulation object.
info.set_points(mesh_points)									               #Set points to be triangulated.
if(len(hole_points) > 0):									     
	info.set_holes(hole_points)                                                #Set holes (contours) to be ignored.
info.set_facets(mesh_facets)       						                       #Set facets.
triangulation = triangle.build(info,verbose=False,allow_boundary_steiner=False,#Build Triangulation.
       allow_volume_steiner=False,quality_meshing=False)

if verbose:
    step = time.clock()                                                        #progress output
    print pa + "Done in %1.2f sec."%(step-previous_step)
    print "\n" + pa + "Current step: Setup of triangles and neighborhood relations."
    previous_step = step

"""
Triangle classification:
        - build triangle-objects from the triangulation
        - set the type of each triangle (junction, normal, end or isolated)
          depending on how many neighbors it has
        - set the radius of each triangle by looking up its "midpoint"
          in the distance map
        - get rid of isolated triangles
"""
triangles = nh.buildTriangles(triangulation)	                                 #Build triangles                                                                 
junction = 0
normal = 0
end = 0
isolated_indices = []
default_triangles = 0
for i in range(len(triangles)):                                                
    t = triangles[i]                                                           #set the triangle's type
    t.init_triangle_mesh()    
    if t.get_type() == "junction":                                             #count the number of each triangle type for debugging
        junction += 1
    elif t.get_type() == "normal":
        normal += 1
    elif t.get_type() == "end":
        end += 1
    elif t.get_type() == "isolated":
        isolated_indices.append(i)
triangles = list(np.delete(np.asarray(triangles), isolated_indices))           #remove isolated triangles from the list of triangles

if debug:                                                                      #debug output
    print pa + "\tTriangle types:"
    print pa + "\tjunction: %d, normal: %d, end: %d, isolated: %d"\
                %(junction,normal,end,len(isolated_indices))
                
if verbose:	           
    step = time.clock()                                                        #progress output
    print pa + "Done in %1.2f sec." %(step-previous_step)
    print "\n" + pa + "Current step: Creation of triangle adjacency matrix." 
    previous_step = step

"""
Triangle adjacency matrix:
        - create a matrix of triangle neighborhoods with distances between
          triangles (euclidean) as weights
        - create a copy of the adjacency matrix for the pruning process

"""
if verbose:                        									    					     
    step = time.clock()                                                        #progress output
    print pa + "Done in %1.2f sec."%(step-previous_step)
    print "\n" + pa + "Current step: Bruteforce Pruning."
    previous_step = step																					

"""
Graph creation and improvement
        - prune away the outermost branches to avoid surplus branches due
          to noisy contours
        - create a graph object from the neighborhood relations, coordinates
          and radius stored in the adjacency matrix and
          the list of triangles.
"""
triangles = nh.bruteforcePruning(triangles,order,verbose)                      #prune away the "order" number of triangles at the ends of the network

default_triangles = 0
junction = 0
normal = 0
end = 0
isolated = 0
for t in triangles:
    t.init_triangle_mesh()
    default_triangles += t.set_center(distance_map)
    if t.get_type() == "junction":                                             #count the number of each triangle type for debugging
        junction += 1
    elif t.get_type() == "normal":
        normal += 1
    elif t.get_type() == "end":
        end += 1
    elif t.get_type() == "isolated":
        isolated += 1
if debug:
    nh.drawTriangulation(height,triangles,image_name,dest,distance_map, \
            figure_format,dpi)                                           #debug output
    print pa + "\tTriangles defaulted to zero: %d"%default_triangles
    print pa + "\tTriangle types:"
    print pa + "\tjunction: %d, normal: %d, end: %d, isolated: %d"\
                %(junction,normal,end,len(isolated_indices))

adjacency_matrix = nh.createTriangleAdjacencyMatrix(triangles)


if verbose:
    step = time.clock()                                                        #progress output
    print pa + "Done in %1.2f sec."%(step-previous_step)
    print "\n" + pa + "Current step: Graph creation."
    previous_step = step

G = nh.createGraph(adjacency_matrix,triangles,height)                          #creation of graph object 
print "number of nodes in the graph: ",len(G.nodes())

if verbose:
    step = time.clock()                                                        #progress output
    print pa + "Done in %1.2f sec."%(step-previous_step)
    print  "\n" + pa + "Current step: Removal of redundant nodes,"\
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
    nh.drawAndSafe(G,image_name,dest,parameters,verbose,plot,figure_format,dpi,\
        graph_format,n_size)                                                   #draw and safe graph with redundant nodes
if debug:
    nh.drawGraphTriangulation(height,G,triangles,image_name,dest,\
        distance_map,figure_format, dpi)

if redundancy == 1:                                                            #draw and safe graph with half redundant nodes
    G = nh.removeRedundantNodes(G,verbose,1)
    nh.drawAndSafe(G,image_name,dest,parameters,verbose,plot,figure_format,dpi,\
        graph_format,n_size)
 
if redundancy == 0:   
    G = nh.removeRedundantNodes(G,verbose,0)                                   #draw and safe graph without redundant nodes
    nh.drawAndSafe(G,image_name,dest,parameters,verbose,plot,figure_format,dpi,\
            graph_format,n_size)										

if verbose:
    step = time.clock()                                                        #progress output
    print pa + "Done in %1.2f sec."%(step-previous_step)
    previous_step = step	
    end = time.clock()                                                             
    print "\n" + pa + "ALL DONE! Total time: %1.2f sec"%(end-start)

