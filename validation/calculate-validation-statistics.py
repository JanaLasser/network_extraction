import networkx as nx
from os import getcwd, listdir
from os.path import join
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from PIL import Image

def count_nodes(G):
	'''
	Simply sums over all nodes in the graph G
	'''
	degrees = list(G.degree().values())
	nodes = len([deg for deg in degrees if deg != 2])
	return nodes

def measure_length(G):
	'''
	Calculates the distance between every two nodes
	of an edge in a graph G and sums up all the distances
	'''
	length = 0
	for edge in G.edges():
		x1 = G.node[edge[0]]['x']
		x2 = G.node[edge[1]]['x']
		y1 = G.node[edge[0]]['y']
		y2 = G.node[edge[1]]['y']
		distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
		length += distance
	return length

def measure_weight(G):
	'''
	Extracts all the weights from the edges in the graph G.
	Caution: NET labels them als 'conductivity', the label
	'weight' is actually reserved for the length of the edge.
	Calculates the average as well as the ratio of the
	smallest to the largest weight.
	'''
	weights = nx.get_edge_attributes(G,'conductivity').values()
	weights = np.asarray([w for w in weights if w != 0])
	ratio = weights.min() / weights.max()
	return (weights.mean(), ratio)

def get_convex_hull(G):
	'''
	Calculates the convex hull from the set of points made up
	by the nodes of the graph G.
	'''
	points = []
	for node in G.nodes(data=True):
		points.append([node[1]['y'],node[1]['x']])

	points = np.array(points)
	hull = ConvexHull(points)
	vertices = points[hull.vertices]
	vertices = np.vstack([vertices,vertices[0,0:]])
	lines = np.hstack([vertices,np.roll(vertices,-1,axis=0)])
	area = 0.5*abs(sum(x1*y2-x2*y1 for x1,y1,x2,y2 in lines))
	return (vertices,area)

def measure_convex_hull_overlap(G,H):
	'''
	Calculates the logical XOR between the two convex hulls of
	the graphs G and H to see how different they look in space
	'''
	points1, area1 = get_convex_hull(G)
	points2, area2 = get_convex_hull(H)
	polygon1 = Polygon(points1)
	polygon2 = Polygon(points2)
	union = cascaded_union([polygon1, polygon2])
	intersection = union.area - polygon1.intersection(polygon2).area
	return (area1, area2, intersection)

def plot_graph(G,dpi):
	'''
	Plots the graph G. Node positions are taken from the 'x' and 'y'
	attributes of G's nodes, edge widths are taken from the 'conductivity'
	attribute of G's edges.
	As the maximum size of the original images is 2100 x 2100 pixels,
	we plot each graph on a canvas of this size. Returns the figure
	for saving. Make sure you close the figure later on!
	'''
	h,w = (2100,2100)
	figure = plt.figure(figsize=(w/float(dpi),h/float(dpi)))
	ax = figure.add_axes([0,0,1,1])
	ax.axis('off')
	#find (x,y) coordinates and width for each edge in the graph and plot it
	for edge in G.edges(data=True):
		x1 = G.node[edge[0]]['x']
		x2 = G.node[edge[1]]['x']
		y1 = G.node[edge[0]]['y']
		y2 = G.node[edge[1]]['y']
		width = G[edge[0]][edge[1]]['conductivity']*2
		ln, = ax.plot([x1,x2],[y1,y2],color='b',linewidth=width)
		#make edges of plottet lines 'round' to help avoid artifacts during
		#vectorization
		ln.set_solid_capstyle('round')

	ax.set(xlim=[0,w],ylim=[0,h],aspect=1)
	return figure

#I/O handling
#change the path to the src if validating with randomgrids
src = join(getcwd(),'validation-graphs-tracheoles')
labels = [filename for filename in listdir(src) if \
	filename.endswith('original_binary.png')]
labels = [label.split('_')[0] for label in labels]

#format handling for the result file
header_format = '{:<40}{:^12}{:^12}{:^12}{:^12}{:^12}{:^12}' + \
			'{:^12}{:^12}{:^12}{:^12}{:^12}{:^12}'
header_values = ['#label','#N_o','#N_v','#L_o','#L_v','#R_o','#r_o',\
		'#R_v','#r_v','#A_o','#A_v','#A_i','#D_px']
entry_format = '{:<40}{:^12d}{:^12d}{:^12.2f}{:^12.2f}{:^12.3f}' + \
			'{:^12.3f}{:^12.2f}{:^12.2f}{:^12.2f}{:^12.2f}' + \
			'{:^12.2f}{:^12.5f}'

#resolution for the pixel-wise difference plots
dpi = 80

#open resultfile and write header
stats = open('validation-statistics.txt','w+')
print >> stats, header_format.format(*header_values)

#iterate through all extracted tracheole networks
for label,i in zip(labels[0:],range(len(labels))):
	print('calculating statistics for cell {}/{}'.format(i,len(labels)))
	#read original and re-extracted graph
	original = nx.read_gpickle(join(src,label + '_original_graph_p2_r1.gpickle'))
	validation = nx.read_gpickle(join(src,label + '_validation_image_graph_p2_r1.gpickle'))

	#plot the original graph
	save_dest = join(src,label + '_original_figure')
	original_figure = plot_graph(original,dpi)
	plt.savefig(save_dest + '.png', dpi=dpi, transparent=True)
	plt.close()
	original_figure = np.asarray(Image.open(save_dest + '.png'))[0:,0:,0]

	#plot the re-extracted graph
	save_dest = join(src,label + '_validation_figure')
	validation_figure = plot_graph(validation,dpi)
	plt.savefig(save_dest + '.png', dpi=dpi, transparent=True)
	plt.close()
	validation_figure = np.asarray(Image.open(save_dest + '.png'))[0:,0:,0]

	#do a pixel-wise comparision of the two plots to check, whether the
	#geometry of the network is sustained
	D_px = np.abs(original_figure - validation_figure).sum()/float(original_figure.sum())

	#measure all the relevant statistics in the networks using
	#the measurement functions defined above

	#count graph nodes
	N_o = count_nodes(original)
	N_v = count_nodes(validation)

	#measure network length
	L_o = measure_length(original)
	L_v = measure_length(validation)

	#measure mean conductivity (edge width) 
	#and ratio of smallest to biggest conductivity
	R_o, r_o = measure_weight(original)
	R_v, r_v = measure_weight(validation)

	#measure the area of the convex hull and XOR of the two convex hulls
	A_o, A_v, A_i = measure_convex_hull_overlap(original, validation)

	#dump everything into the resultfile, analysis is done in a different script
	entry_values = [label,N_o,N_v,L_o,L_v,R_o,r_o, \
				R_v,r_v,A_o,A_v,A_i,D_px]
	print >> stats, entry_format.format(*entry_values)

stats.close()