# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 11:20:51 2014

@author: Jana Lasser
"""

'''
Copyright (C) 2015 Jana Lasser GPL-3.0
'''

import matplotlib.pyplot as plt
from os.path import join
import numpy as np
import networkx as nx
from PIL import Image
from matplotlib.patches import Rectangle

def getImage(image_path):    
    image = Image.open(image_path)
    image = image.convert('L')
    image = np.asarray(image,dtype=np.uint8)
    return np.flipud(image)
  
#####################################################
#               Internal plotting Class             #
#####################################################
'''
Handles all the appearing and disappearing of objects on the canvas. The class
does not know the graph itself. It is composed of methods to draw nodes and
edges and keeps track of the objects it has drawn. If nodes or edges are
removed from the graph, this class takes care of simultaneously removing the
corresponding objects on the canvas.
'''
class PlotHandler(object):
    
    def __init__(self,figure,name_dict):     
        self.figure = figure
        self.current_mode = None
        self.current_action = None
        self.ax = self.figure.gca()
        #self.ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
        #  fancybox=True, shadow=True)
        self.name_dict = name_dict
        try:
            self.background = getImage(join(self.name_dict['source_path'],
                                        self.name_dict['orig_name']))
        except IOError:
            print "No original .tif file found!"
            
        self.height, self.width = self.background.shape
        
        self.node_list = {}
        self.edge_list = {}
        self.marked_list = {}
        self.width_scale_factor = 0
        plt.draw()
            
        
    def plot_graph(self,node_collection,edge_collection):
        plt.clf()    
        node_collection = dict(node_collection)
        edge_collection = edge_collection
        
        widths = np.array([edge[2]['conductivity'] for edge in edge_collection])
        self.width_scale_factor = 1./np.amax(widths)    
     
        for edge in edge_collection:
            x1 = node_collection[edge[0]]['x']
            x2 = node_collection[edge[1]]['x']
            y1 = node_collection[edge[0]]['y']
            y2 = node_collection[edge[1]]['y']
            width = edge[2]['conductivity']
            new_edge = self.figure.gca().plot([x1,x2],[y1,y2],\
                linewidth=self.width_scale_factor*width,color='k',zorder=2)[0]
            self.edge_list.update({(edge[0],edge[1]):new_edge})
            
        for node in node_collection:
            x = node_collection[node]['x']
            y = node_collection[node]['y']
            new_node = self.figure.gca().plot(x,y,marker='o',color='b',\
                    markersize=5,zorder=9)[0]
            self.node_list.update({node:new_node})
        
        colormap = plt.get_cmap('hot')
        plt.imshow(self.background,origin='lower',alpha=0.5,cmap=colormap)
    
    def update_mode(self,text):     
        self.current_mode = self.figure.gca().set_title(text,fontsize=14)
        
    def update_action(self,text):
        self.current_action = self.figure.gca().set_xlabel(text,fontsize=14)
        
        
    def mark_node(self,n,x_s,y_s):
        try:
            self.marked_list[n]
        except KeyError:   
            new_mark = self.figure.gca().plot(x_s,y_s,marker='o',color='r',\
                                            markersize=7,zorder=10)[0]
            self.marked_list[n]=new_mark

    def unmark_node(self,n):
        self.marked_list[n].remove()
        del self.marked_list[n]
        
    def draw_node(self,node,x,y):
        new_node = self.figure.gca().plot(x,y,marker='o', color='b',\
                                            markersize=5,zorder=9)[0]
        self.node_list.update({node:new_node})
 
    def undraw_node(self,n):
        self.node_list[n].remove()
        del self.node_list[n]
        
    def clear_selection(self):
        for key in self.marked_list.keys():
            self.unmark_node(key)
        self.marked_list = {}
            
    def draw_edge(self,n1,n2,x1,y1,x2,y2,radius):
        new_edge = self.figure.gca().plot([x1,x2],[y1,y2],\
            linewidth=self.width_scale_factor*2*radius,color='k',zorder=2)[0]
        self.edge_list.update({(n1,n2):new_edge})
     
    def undraw_edge(self,nodes):
        if nodes in self.edge_list:
            self.edge_list[nodes].remove()
            del self.edge_list[nodes]
            
    def plot_and_save(self,G,name):
    
        figure = plt.figure()
        ax = figure.gca()
        pos = {}
        for k in G.node.keys():
            pos[k] = (G.node[k]['x'], G.node[k]['y'])
        
        widths = np.array([G[e[0]][e[1]]['conductivity'] for e in G.edges()])
        widths = 15./np.amax(widths)*widths

        nx.draw_networkx_edges(G, pos=pos, width=widths,arrows=False)
        root_x = G.node[0]['x']
        root_y = G.node[0]['y']
        ax.plot([0,root_x],[0,root_y], 'ro')
        figure.savefig(join(self.name_dict['dest_path'],name + ".png"),dpi=600)
        plt.close(figure)
            
        
    
    
    