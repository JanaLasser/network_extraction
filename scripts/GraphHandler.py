# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 11:20:41 2014

@author: Ista
"""

import matplotlib.pyplot as plt
import os
#import argparse
import networkx as nx
#import vectorize_helpers2 as vh
import numpy as np
#import itertools
import PlotHandler
#import numpy as np
from os.path import join
from PIL import Image



#####################################################
#               Internal Graph Class                #
#####################################################

def getImage(image_path):    
    image = Image.open(image_path)
    image = image.convert('L')
    image = np.asarray(image,dtype=np.uint8)
    return image

def load(filename):
    if filename.endswith(".gpickle"): return nx.read_gpickle(filename)
    elif filename.endswith("dm.png"): return getImage(filename)
    else:
        raise Exception("Unknown filename ending")
        return 

class GraphHandler(object):
    
    def __init__(self,figure,name_dict,state):
        self.figure = figure
        self.name_dict = name_dict
        self.graph = state.graph
        self.selected_nodes = state.selected_nodes
        self.distance_map = np.flipud(load(join(name_dict['source_path'],\
                                                     name_dict['dm_name'])))
        self.init_PlotHandler()
        self.nodemax = np.amax(np.asarray(self.graph.nodes()))
     
    def init_PlotHandler(self):
        self.PH = PlotHandler.PlotHandler(self.figure,self.name_dict)
        
        print "initializing plot handler with %d nodes and %d edges"\
                    %(len(self.graph.nodes()),len(self.graph.edges()))
                    
        self.PH.plot_graph(self.graph.nodes(data=True),\
                    self.graph.edges(data=True))  
                    
        for node in self.selected_nodes:
            x = self.graph.node[node]['x']
            y = self.graph.node[node]['y']
            self.PH.mark_node(node,x,y)
            
    
    def clear_selection(self):
        self.selected_nodes = {}
        self.PH.clear_selection()
        
    def get_node_from_xy(self,x,y):
        x_disp, y_disp = self.PH.figure.axes[0].transData.transform((x,y))
        for n, (x_s, y_s) in [(n, (d['x'], d['y']))\
                for n, d in self.graph.nodes_iter(data=True)]:
                    
            xs_disp, ys_disp = \
                self.PH.figure.axes[0].transData.transform((x_s, y_s))
                
            dist = np.sqrt((xs_disp - x_disp)**2 + (ys_disp - y_disp)**2)
            
            if dist < 8: return (n,x_s,y_s)                
            
    def measure_radius(self,x,y):
        radius = self.distance_map[y,x]
        return radius
   
    def delete_node(self,n):
        try:
            neighbors = self.graph.neighbors(n)
            self.graph.remove_node(n)
            self.PH.undraw_node(n)
            for neighbor in neighbors:
                self.PH.undraw_edge((n,neighbor))
                self.PH.undraw_edge((neighbor,n))
            self.PH.unmark_node(n)
        except nx.NetworkXError:
            print "tried to remove a node which is not there"
        
    def create_node(self,x,y):
        node = self.nodemax+1
        radius = self.distance_map[y,x]
        self.graph.add_node(node, x=x,y=y,conductivity=radius)
        print "node created, number ",node
        self.PH.draw_node(node,x=x,y=y)
        self.nodemax += 1            
                  
    def create_edge(self,n1,n2):
        print self.graph[n1]
        r1 = self.graph.node[n1]['conductivity']
        r2 = self.graph.node[n1]['conductivity']
        x1 = self.graph.node[n1]['x']
        y1 = self.graph.node[n1]['y']
        x2 = self.graph.node[n2]['x']
        y2 = self.graph.node[n2]['y']
        radius = (r1 + r2)/2.0
        length = np.sqrt( (x1 - x2)**2 + (y1 - y2)**2)
        self.graph.add_edge(n1,n2,conductivity=radius,weight=length)
        print "edge created: n1 radius %f, n2 radius %f, edge radius %f"\
              %(r1,r2,radius)
        self.PH.draw_edge(n1,n2,x1,y1,x2,y2,radius)
    
    def detect_cycles(self):
        cycles_raw = nx.algorithms.cycle_basis(self.graph)
        number_of_cycles = len(cycles_raw)
        if number_of_cycles == 0:
            return None
        else:
            cycles = {}
            nodes = dict(self.graph.nodes(data=True))
            for cycle in cycles_raw:
                for node in cycle:
                    x = nodes[node]['x']
                    y = nodes[node]['y']
                    cycles.update({node:(x,y)})
            for key in cycles.keys():
                self.PH.mark_node(key, cycles[key][0],cycles[key][1])
            return cycles      
      
    def streamline_graph(self):          
        self.rm_redundant_node()
        self.PH.plot_graph(self.graph.nodes(data=True),\
                           self.graph.edges(data=True))    
           
    def rm_redundant_node(self):
        order = self.graph.order()
        new_order = 0   
        i = 0
        
        while(new_order != order):
            nodelist = []
            for node in self.graph.nodes():
                if(self.graph.degree(node)==2):
                    neighbors = self.graph.neighbors(node)
                    n1 = neighbors[0]
                    n2 = neighbors[1]
                    l1 = self.graph.edge[node][n1]['weight']
                    l2 = self.graph.edge[node][n2]['weight']
                    r1 = self.graph.edge[node][n1]['conductivity']
                    r2 = self.graph.edge[node][n2]['conductivity']
                    length = l1 + l2
                    radius = (r1*l1+r2*l2)/(l1+l2)
                    self.graph.add_edge(n1,n2,weight=length,conductivity = radius)
                    nodelist.append(node)
                    
            for node in nodelist:
                self.graph.remove_node(node)
            order = new_order
            new_order = self.graph.order() 
            i += 1
            
    def create_digraph(self,root):
        T = nx.dfs_tree(self.graph,root)
        
        #manually put node attributes into new digraph
        for node in T.nodes(data=True):
            node[1]['y'] = self.graph.node[node[0]]['y']
            node[1]['x'] = self.graph.node[node[0]]['x']
            node[1]['conductivity'] = self.graph.node[node[0]]['conductivity']
            
        #manually put edge attributes into new digraph
        for edge in T.edges(data=True):
            length = self.graph[edge[0]][edge[1]]['weight'] 
            radius = self.graph[edge[0]][edge[1]]['conductivity']              
            T[edge[0]][edge[1]]['weight'] = length
            T[edge[0]][edge[1]]['conductivity'] = radius
        
        L = list(nx.bfs_edges(T,source=root))
        nodes = [val for subl in L for val in subl]
        bfs_preorder_nodes = []
        for item in nodes:
            if item not in bfs_preorder_nodes:
                bfs_preorder_nodes.append(item)
                
        indices = range(0,len(nodes))
        new_node_label_dict = dict(zip(bfs_preorder_nodes,indices))
        T = nx.relabel_nodes(T,new_node_label_dict)

        nx.write_gpickle(T,join(self.name_dict['dest'],\
                    self.name_dict['work_name'] + "_full_digraph.gpickle"))
        
        self.streamline_graph()
        nx.write_gpickle(T,join(self.name_dict['dest'],\
                    self.name_dict['work_name'] + "_full_digraph.gpickle"))
        
    def draw_tree(self,T):
        figure2 = plt.figure()
        nx.draw_graphviz(T,prog='dot')
        plt.savefig(join(self.name_dict['dest'],\
                    self.name_dict['work_name'] + "_tree.png"))
        del figure2
        plt.close()               
      
    def save_graph(self,normalized):
        i = 1
        while True:
            if normalized:
                name = self.name_dict['work_name'] + '_digraph' + str(i) + '.gpickle'
            else:
                name = self.name_dict['work_name'] + '_new' + str(i) + '.gpickle'
            dest = self.name_dict['dest_path']
            if os.path.isfile(join(dest,name)):
                i +=1
            else:      
                print name
                print dest
                nx.write_gpickle(self.graph,join(dest,name))
                break
        print "saved graph!"