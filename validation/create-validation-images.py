#standard imports
from os import getcwd, listdir, remove
from os.path import join, isfile

#dependencies
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand, uniform, choice
from noise import snoise2
from PIL import Image
from skimage.filters import gaussian
from scipy.misc import toimage, imsave 
import sys

def format_number(i):
	if i < 10:
		return '00' + str(i)
	elif i < 100:
		return '0' + str(i)
	else:
		return str(i)

def create_background(h,w,octaves):
	'''
	Creates a background image of dimensions (h,w) with spatially correlated
	noise mimicking differences in background illumination in microscopy images.
	See https://github.com/caseman/noise for specifics of the noise
	implementation.
	'''
	freq = 16.0 * octaves
	img = np.zeros((h,w))
	for y in range(h):
		for x in range(w):
			 img[y,x] = int(snoise2(x / freq, y / freq, octaves) * 127.0 + 128.0)

	return img

def create_random_grid(N,h,w):
	'''
	Creates a 'random' grid with N nodes. Node positions are shuffled a bit in 
	space but without making edges cross. Excess edges are removed so no node 
	has a degree > 3. Assigns a random weight to edges. All node positions and
	edge weights are rescaled so they fit into the dimensions h and w of the
	background.
	'''

	#initial grid graph
	N = int(np.sqrt(N))
	G = nx.grid_2d_graph(N,N)

	for node in G.nodes():
		#scales all positions to be in the [0,1] interval because its easier
		#to think in the unit interval ;)
		scale_y = float(N - 1)
		scale_x = float(N - 1)
		#the 'wiggle range' determines the maximum distance, a node can move
		#around in space. A wiggle range of 0.4 ensures, that in a quadratic
		#grid edges will never overlap
		wiggle_range_y = 0.4/(N - 1)
		wiggle_range_x = 0.4/(N - 1)
		#offset ensures that the outer nodes are not too close to the edges of
		#the background and therefore image segmentation does not run into
		#troubles there
		offset_y = h*0.4/(N - 1)
		offset_x = h*0.4/(N - 1)
		#caluclate the new node position by scaling to [0,1], wiggling, 
		#offsetting and then scaling to the dimensions of the background
		new_y = (node[0]/scale_y + uniform(-wiggle_range_y,wiggle_range_y)) \
			*(h - offset_y*3) + 1.5*offset_y
		new_x = (node[1]/scale_x + uniform(-wiggle_range_x,wiggle_range_x)) \
			*(w - offset_x*3) + 1.5*offset_y
		G.node[node].update({'y':new_y,'x':new_x})

	#by default, node-labels created by the grid_2d_graph function are tuples
	#like (0,0) for the first node. I find it easier to deal with a single
	#number as label so I sort the default labels and relabel the nodes
	nodes = G.nodes()
	nodes.sort()
	node_mapping = {node:i for i, node in enumerate(nodes)}
	nx.relabel_nodes(G,node_mapping,copy=False)

	#NET by default only deals with graphs with a maximum node degree of 3 like
	#the networks in the trachea images. To mimick these graphs, we remove 
	#escess edges: for every node with a degree > 3 we choose a random neighbor
	#and remove the edge between them
	for node in G.nodes():
		if G.degree()[node] > 3:
			neighbor = choice(G.neighbors(node),1)[0]
			G.remove_edge(node, neighbor)		

	#last but not least we assign a random weight to each edge. The weights are
	#chosen uniformly from the interval [0.2,1) and then scaled to the
	#background dimensions. We bound the interval at 0.2 because super-thin
	#edges are problematic to render in the final plot.	
	for edge in G.edges():
		G[edge[0]][edge[1]].update({'w':uniform(0.3,1.0)*h/100.})	

	return G

def largest_connected_component(G):
	'''
	Detects the largest connected component CC in a graph G. Removes all nodes 
	not belonging to CC from G and returns the remaining graph.
	'''
	CC = max(nx.connected_components(G), key=len)
	for n in G.nodes():
		if n not in CC:
			G.remove_node(n)
	return G	

def plot_graph(G,background,h,w,dpi):
	'''
	Plots the graph superimposed over the background image. We try to make
	validation images close to the size of the trachea images, which is about
	2100 x 2100 pixels. As matplotlib deas with physical sizes rather than 
	pixels, we need to set dpi as scale factor to control the number of
	pixels in the plot.
	'''
	#create a figure with appropriate dimensions
	figure = plt.figure(figsize=(w/float(dpi),h/float(dpi)))
	ax = figure.add_axes([0,0,1,1])
	ax.axis('off')
	#display the background image. We constrain the gray values to 0.8*254
	#as background noise usually has less intensity than foregorund features.
	ax.imshow(background*0.5,cmap=plt.get_cmap('gray'),vmin = 50,vmax=254*0.8, \
		interpolation='nearest')
	#find (x,y) coordinates and width for each edge in the graph and plot it
	for edge in G.edges(data=True):
		x1 = G.node[edge[0]]['x']
		x2 = G.node[edge[1]]['x']
		y1 = G.node[edge[0]]['y']
		y2 = G.node[edge[1]]['y']
		width = G[edge[0]][edge[1]]['conductivity']*2
		#to mimick noise in the image even further and make segmentation harder,
		#we assign a randomly chosen gray value from the interval [0.7*254, 254)
		#to each edge. Note that intensity intervals for the background and the
		#edges overlap.
		color = uniform(0.7,0.9)
		ln, = ax.plot([x1,x2],[y1,y2],color=[color,color,color],linewidth=width)
		#make edges of plottet lines 'round' to help avoid artifacts during
		#vectorization
		ln.set_solid_capstyle('round')

	ax.set(xlim=[0,w],ylim=[0,h],aspect=1)
	return figure
	
### validation image parameters ###
mode = 'tracheoles'
h, w = (2100, 2100) #dimensions of the background in pixels
octaves = 40		#parameter controlling the spatial correlation of the noise
dpi = 80			#screen dpi

#In the publication we validate with real-world graphs extracted
#from drosophila tracheoles -> mode = 'tracheoles'. The known graphs
#as well as their binaries are stored in the 'validation-graphs-tracheoles'
#folder. We uploaded a random selection of 20 graphs from the 531 graphs
#used in the full validation so you can retrace our steps if you like.
if mode == 'tracheoles':
	dest = join(getcwd(),'validation-graphs-tracheoles')
	GRAPHS = listdir(dest) #number of graphs
	GRAPHS = [g for g in GRAPHS if g.endswith('.gpickle')]
	GRAPHS.sort() #sort just so progress is easier to track

#We can also validate with atrificially created random graphs
# -> mode == 'randomgrid'. Set GRAPHS to determine how many
#random graphs you want to create and plot. Be aware that each
#graph will take about 5 sec to create as writing images to the
#harddrive is quite a lengthy process.
#Be aware that the path to the tracheoles folder is hardcoded
#in the analysis scripts. If you want to try the validation
#with randomgrids, you will have to change it.
elif mode == 'randomgrid':
	dest = join(getcwd(),'validation-graphs-grids')
	GRAPHS = 100 #number of graphs to be created
	GRAPHS = range(GRAPHS)
	NODES = 50	#maximum number of nodes in each graph
else:
	print('mode not recognized, aborting ...')
	sys.exit()

#for each graph - tracheole or random grid - we create an artificial
#noisy background and then plot the graph on it and save the resulting image
#for later re-extraction using NET.
for graph,i in zip(GRAPHS,range(len(GRAPHS))):
	print('creating validation image {}/{}'.format(i,len(GRAPHS)))
	#create noisy background image
	background = create_background(h,w,octaves)
	if mode == 'tracheoles':
		save_dest = join(dest,graph.split('_')[0] + '_validation_image')
		#load original tracheole graph
		G = nx.read_gpickle(join(dest,graph))
	else:
		save_dest = join(dest,'{}_original_graph'.format(format_number(graph)))
		#create random grid graph
		G = create_random_grid(NODES,h,w)
		#remove disconnected parts, therefore actual number of nodes can be < NODES
		G = largest_connected_component(G)
		#save the newly created graph
		nx.write_gpickle(G,save_dest + '.gpickle')
		save_dest = join(dest,'{}_validation_image'.format(format_number(graph)))
	
	#plot the graph on the new background image
	figure = plot_graph(G,background,h,w,dpi)
	#save the plot as .png, reload it to get a bitmap, blur it a bit and save it again
	plt.savefig(save_dest + '.png', dpi=dpi, transparent=True)
	img = np.asarray(Image.open(save_dest + '.png'))[0:,0:,0]
	img = gaussian(img, 2)
	imsave(save_dest + '.png', img)
	plt.close()
