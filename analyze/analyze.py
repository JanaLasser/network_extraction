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
import argparse
import networkx as nx
import numpy as np
import scipy.spatial
import ntpath
from os.path import join

parser = argparse.ArgumentParser(description='Analyze: a program that ' +\
    'calculates basic statistics of a graph.\nCopyright (C) 2015 Jana Lasser')

parser.add_argument('source',type=str,help='Complete path to the graph')

parser.add_argument('-dest', type=str, help='Complete path to the folder '\
                + 'results will be saved to if different than source folder')
                
args = parser.parse_args()
source = args.source
dest = args.dest

graph_name = image_name = ntpath.basename(source).split('.')[0]
if dest == None:
    dest = source.split(graph_name)[0]
graph_name = graph_name.split('_graph')[0]

G = nx.read_gpickle(source)

def NumberOfJunctions(G):
    junctions = 0    
    for n in G.nodes():
        if type(G) == type(nx.DiGraph):
            if len(G.neighbors(n)) >= 2:
                junctions += 1
        else:
            if len(G.neighbors(n)) >= 3:
                junctions += 1 
                
    return junctions
    
def NumberOfTips(G):
    tips = 0  
    for n in G.nodes():
        if type(G) == type(nx.DiGraph):
            if len(G.neighbors(n)) == 0:
                tips += 1
        else:
            if len(G.neighbors(n)) == 1:
                tips += 1 
                
    return tips
    
def TotalLength(G):
    return np.asarray([e[2]['weight'] for e in G.edges_iter(data=True)]).sum()
    
def AverageEdgeLength(G):
    return np.asarray([e[2]['weight'] for e in G.edges_iter(data=True)]).mean()
    
def AverageEdgeRadius(G):
    return np.asarray([e[2]['conductivity'] for e in G.edges_iter(data=True)]).mean()
    
def TotalNetworkArea(G):
    return np.asarray([e[2]['weight']*e[2]['conductivity'] \
        for e in G.edges_iter(data=True)]).sum()
    
def AreaOfConvexHull(G):
    points = np.asarray([[n[1]['y'],n[1]['x']] for n in G.nodes(data=True)])   
    hull = scipy.spatial.ConvexHull(points)
    vertices = points[hull.vertices]
    vertices = np.vstack([vertices,vertices[0,0:]])
    lines = np.hstack([vertices,np.roll(vertices,-1,axis=0)])
    area = 0.5*abs(sum(x1*y2-x2*y1 for x1,y1,x2,y2 in lines))
    return area

def NumberOfCycles(G):
    return len(nx.cycle_basis(G))

f = open(join(dest,graph_name + '_network_statistics' + '.txt'), 'w')
f.write('*** Statistics for ' + graph_name + ' ***\n\n')
f.write('Number of junctions:\t %d\n'%NumberOfJunctions(G))
f.write('Number of tips:\t\t %d\n'%NumberOfTips(G))
f.write('Total length:\t\t %f px\n'%TotalLength(G))
f.write('Average edge length:\t %f px\n'%AverageEdgeLength(G))
f.write('Average edge radius:\t %f px\n'%AverageEdgeRadius(G))
f.write('Total network area:\t %f px^2\n'%TotalNetworkArea(G))
f.write('Area of convex hull:\t %f px^2\n'%AreaOfConvexHull(G))
f.write('Number of cycles:\t %d\n'%NumberOfCycles(G)) 
f.close()



