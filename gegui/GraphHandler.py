# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 11:20:41 2014

@author: Jana Lasser
"""
'''
Copyright (C) 2015 Jana Lasser GPL-3.0
'''

import matplotlib.pyplot as plt
import os
import networkx as nx
import numpy as np
import PlotHandler
from os.path import join
import InterActor as IA

#####################################################
#               Internal Graph Class                #
#####################################################



class GraphHandler(object):
    
    def __init__(self,figure,name_dict,state):
        self.name_dict = name_dict
        self.graph = state.graph
        self.selected_nodes = state.selected_nodes
        self.distance_map = None
        for f in os.listdir(name_dict['source_path']):
            if f.endswith('_dm.png'):
                self.distance_map = IA.load(join(name_dict['source_path'],f))
        if self.distance_map == None:
            print 'No corresponding distance map found (looking for "_dm.png")'
            print 'GUI still functional but trying to create new nodes will '
            print 'result in an error!'
            IA.printHelp()
            
        self.init_PlotHandler(figure)
        self.nodemax = np.amax(np.asarray(self.graph.nodes()))
     
    def init_PlotHandler(self,figure):
        self.PH = PlotHandler.PlotHandler(figure,self.name_dict)             
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
            pass
        
    def create_node(self,x,y):
        node = self.nodemax+1
        radius = self.distance_map[y,x]
        self.graph.add_node(node, x=x,y=y,conductivity=radius)
        self.PH.draw_node(node,x=x,y=y)
        self.PH.update_action("Node created at (%1.2f,%1.2f), "%(x,y)\
            + "radius = %1.1f"%(radius))                                        #display the last action at the bottom of the figure
        self.nodemax += 1            
                  
    def create_edge(self,n1,n2):
        r1 = self.graph.node[n1]['conductivity']
        r2 = self.graph.node[n1]['conductivity']
        x1 = self.graph.node[n1]['x']
        y1 = self.graph.node[n1]['y']
        x2 = self.graph.node[n2]['x']
        y2 = self.graph.node[n2]['y']
        radius = (r1 + r2)/2.0
        if radius < 1.0:
            radius = 1
        length = np.sqrt( (x1 - x2)**2 + (y1 - y2)**2)
        self.graph.add_edge(n1,n2,conductivity=radius,weight=length)
        self.PH.draw_edge(n1,n2,x1,y1,x2,y2,radius)
        self.PH.update_action("-e: Connected n %d (r = %1.1f) to "%(n1,r1) \
            + "n %d (r = %1.1f), edge radius = %1.1f "%(n2,r2,radius))
    
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
        
        digraph = False
        if type(self.graph) == type(nx.DiGraph()):
            digraph = True
        
        while(new_order != order):
            nodelist = []
            for node in self.graph.nodes():
                if digraph:
                    pred = self.graph.predecessors(node)
                    succ = self.graph.successors(node)
                    if (len(pred) == 1 and len(succ) == 1):
                        n1 = pred[0]
                        n2 = succ[0]
                        l1 = self.graph.edge[n1][node]['weight']
                        l2 = self.graph.edge[node][n2]['weight']
                        r1 = self.graph.edge[n1][node]['conductivity']
                        r2 = self.graph.edge[node][n2]['conductivity']
                        length = l1 + l2
                        radius = (r1*l1+r2*l2)/(l1+l2)
                        self.graph.add_edge(n1,n2,weight=length,\
                                conductivity = radius)
                        if n1 not in nodelist and n2 not in nodelist:
                            nodelist.append(node)
                    
                else:
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
                        self.graph.add_edge(n1,n2,weight=length,\
                                conductivity = radius)
                        if n1 not in nodelist and n2 not in nodelist:
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
        
        name = 'fullDigraph'
        nx.write_gpickle(T,join(self.name_dict['dest_path'],name+".gpickle"))
        self.PH.plot_and_save(T,name)
        
        self.graph = T
        self.streamline_graph()
        name = 'digraph'
        nx.write_gpickle(self.graph,join(self.name_dict['dest_path'],\
            name + ".gpickle"))
        self.PH.plot_and_save(self.graph,name)
      
    def draw_tree(self,T):
        figure2 = plt.figure()
        nx.draw_graphviz(T,prog='dot')
        plt.savefig(join(self.name_dict['dest_path'],'tree.png'))
        del figure2
        plt.close()               
      
    def save_graph(self):
        i = 1
        while True:
            name = self.name_dict['work_name'] + '_new' + str(i)
            dest = self.name_dict['dest_path']
            if os.path.isfile(join(dest,name)):
                i +=1
            else:      
                nx.write_gpickle(self.graph,join(dest,name + '.gpickle'))
                self.PH.plot_and_save(self.graph,name)
                break