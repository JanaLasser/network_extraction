# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 11:19:04 2014

@author: Ista
"""

import matplotlib.pyplot as plt
import os
from os.path import join
#import argparse
#import networkx as nx
#import tile_functions as tf
#import numpy as np
import GraphHandler
from copy import deepcopy
from PIL import Image
import networkx as nx
import numpy as np


#####################################################
#               Interaction Class                   #
#####################################################

class StateSnapshot(object):
    def __init__(self,graph,selected_nodes):
        self.graph = graph
        self.selected_nodes = selected_nodes
        
def load(filename):
    if filename.endswith(".gpickle"): return nx.read_gpickle(filename)
    elif filename.endswith("dm.png"): return getImage(filename)
    else:
        raise Exception("Unknown filename ending")
        return 
        
def getImage(image_path):    
    image = Image.open(image_path)
    image = image.convert('L')
    image = np.asarray(image,dtype=np.uint8)
    return image

class InterActor(object):
    
    
    def __init__(self,source,dest_path):
        self.state_stack = []
        self.max_stored_states = 10
        
        #handling of file IO names
        work_name, original_image_format = os.path.basename(source).split('.')
        source_path = source.split(work_name)[0]
        if dest_path == None: dest_path = source_path
        graph_name = work_name + '_graph_red1.gpickle'
        dm_name = work_name + '_dm.png'
        self.name_dict = {'work_name':work_name,
                               'orig_name':work_name + '.' + original_image_format,
                               'source_path':source_path,
                               'dest_path':dest_path,
                               'graph_name':graph_name,
                               'dm_name':dm_name}
                               
        self.graph = load(join(self.name_dict['source_path'],\
                                    self.name_dict['graph_name']))
        self.selected_nodes = {}
        self.mode_list = []
        
        #state flags
        self.select_on = False
        self.measure_on = False
        self.node_create_on = False
        self.normalized = False
        self.shift_on = False
        
        #figure switches                       
        self.dpi = 100
        self.init_window()
        self.CurrentStateSnapshot = StateSnapshot(self.graph,self.selected_nodes)
        self.GH = GraphHandler.GraphHandler(self.figure,self.name_dict,\
                                            self.CurrentStateSnapshot)
        self.update_state_stack()
        self.update_mode(self.name_dict['work_name'],'add')
        self.edit_loop()      

    def clear_selection(self):
        self.selected_nodes = {}
        self.GH.clear_selection()

    def processing_step(self):
        self.figure.canvas.draw()
        self.set_lim()
        self.update_state_stack()
        
    def update_mode(self,text,action):
        if action == 'rm':
            self.mode_list.remove(text)
        elif action == 'add':
            self.mode_list.append(text)
        else:
            print "action not recognized!"
            
        s = ""
        for item in self.mode_list:
            s += item + "\n"
        s.rstrip('\n')
        self.GH.PH.update_mode(s)

    def set_lim(self):
        #plt.xlim((0,self.GH.PH.width))
        #plt.ylim((0,self.GH.PH.height))
        pass

    def init_window(self):
        print "initializing figure"
        self.figure = plt.figure(dpi=self.dpi,figsize=(12,12)) 
        
    def update_state_stack(self):
        self.CurrentStateSnapshot = StateSnapshot(self.graph,self.selected_nodes)
        if len(self.state_stack) > self.max_stored_states:
            del self.state_stack[0]
            self.state_stack.append(deepcopy(self.CurrentStateSnapshot))
        else:
            self.state_stack.append(deepcopy(self.CurrentStateSnapshot))
            
    def undo(self):
        if len(self.state_stack) >= 1:
            self.state_stack.pop()
            try:
                self.CurrentStateSnapshot = self.state_stack[-1] 
            except IndexError:
                print "no more undo-operations stored!"                  
            self.figure.clf()
            self.graph = self.CurrentStateSnapshot.graph
            self.selected_nodes = self.CurrentStateSnapshot.selected_nodes
            self.GH = GraphHandler.GraphHandler(self.figure, self.name_dict,\
                                                self.CurrentStateSnapshot)
            print self.selected_nodes
        else:
            print "no more undo-operations stored!"
        
    def edit_loop(self):
        self.print_help_message()
        while True:
            cmd = raw_input("GE> ")

            #lave and maybe save before leaving
            if cmd == 'x':                
                while True:
                    cmd2 = raw_input("Want to save before leaving? 'y/n' ")
                    if cmd2 == 'y':
                        self.GH.save_graph(self.normalized)
                        return
                    elif cmd2 == 'n':
                        return
                    else:
                        print 'input not recognized!'
             
            #save current progress
            elif cmd == 's':
                self.GH.save_graph(self.normalized)
            
            #enter manual graph manipulation mode
            elif cmd == 'm':
                self.manipulation_mode()
            
            #streamline graph if not already done
            elif cmd == 'norm':
                if not self.normalized:
                    self.GH.streamline_graph()
                    self.normalized = True
                else:
                    print "already streamlined!"
            
            #create digraph with the currently selected node as root
            elif cmd == 'digraph':
                if len(self.selected_nodes) != 1:
                    print "Select exactly one node for di-Graph creation!"
                else:
                    self.GH.create_digraph(list(self.selected_nodes)[0])             
            
            #enter trace mode (not yet implemented!)
            elif cmd == 't':
                self.trace_mode()
            
            #redraw the current figure (not sure if functional / needed)
            elif cmd == 'r':
                self.init_window()
            
            #display help again
            elif cmd == 'h':
                self.print_help_message()
            
            #unknown user input
            else:
                print "Command not recognized."
                
    def print_help_message(self):
        print """ GraphEdit - edits graphs in networkx format.
        Available commands:
        
        s       - Save current graph
        r       - Show window again
        m       - Enter manipulation mode
        norm    - Streamline graph
        digraph - Create and save digraph
        t       - Enter trace mode (not yet implemented)
        h       - Print help message
        x       - Exit
        """
    
    def manipulation_mode(self):
        print """
        Press 'b' to enter node manipulation mode:
                  Select and deselect nodes by clicking on them.
                  Create new nodes by shift-clicking on the figure.
        Press 'm' to show cycles
        Press 'a' to clear the current node selection
        Press 'd' to delete selected node(s) and adjacent edges
        Press 'e' to connect all selected nodes
        Press 'z' to undo the last action
        
        """
        
        self.update_mode("Manipulation Mode","add") 
        self.figure.canvas.draw()
        
        cid_press = self.figure.canvas.mpl_connect('key_press_event',
                 self.manipulation_key_press)
                 
        raw_input()
        
        if self.select_on:
            self.update_mode("-b: Node Manipulation",'rm')
            self.figure.canvas.mpl_disconnect(self.cid_click)
            self.select_on = False
        self.update_mode("Manipulation Mode",'rm')
        self.figure.canvas.draw()
        self.figure.canvas.mpl_disconnect(cid_press)
        
                 
                 
                 
    def manipulation_key_press(self,event):
        
        #node selection mode
        if event.key == 'b':
            if self.select_on:
                self.update_mode("-b: Node Manipulation",'rm')
                self.figure.canvas.draw()
                self.figure.canvas.mpl_disconnect(self.cid_click)
                self.select_on = False
            else:
                self.update_mode("-b: Node Manipulation",'add')
                self.figure.canvas.draw()
                self.cid_mod1 = self.figure.canvas.mpl_connect('key_press_event',
                                self.modifier_key_on)
                self.cid_mod1 = self.figure.canvas.mpl_connect('key_release_event',
                                self.modifier_key_off)
                self.cid_click = self.figure.canvas.mpl_connect('button_press_event',
                    self.on_click_select)
                self.select_on = True
            
        if event.key == 'm':
            print "looking for cycles"
            cycles = self.GH.detect_cycles()
            if cycles != None:
                self.GH.PH.update_action("-m: Cycles marked")
                self.selected_nodes.update(cycles) 
                self.processing_step()
            else:
                self.GH.PH.update_action("-m: No Cycles detected")
                self.figure.canvas.draw()
                print "no cycles detected!"
               
        if event.key == 'a':
            print "clearing current selection:"
            self.GH.PH.update_action("-a: Selection cleared")
            self.clear_selection()
            self.processing_step()
            
        if event.key == 'd':
            print "deleting selected nodes:"
            self.GH.PH.update_action("-d: Selected nodes deleted")
            self.processing_step()
            for n in self.selected_nodes:
                self.GH.delete_node(n)
            self.clear_selection()
            self.processing_step()
                          
        #create a new edge between two selected nodes    
        if event.key == 'e':
            print "connecting selected nodes"
            if len(self.selected_nodes) > 2:
                self.GH.PH.update_action("-e: Too many nodes selected!")
                print('more than two nodes selected, aborting...')
                return
            self.GH.PH.update_action("-e: Connected selected nodes")
            n1 = self.selected_nodes.keys()[0]
            n2 = self.selected_nodes.keys()[1]
            self.GH.create_edge(n1,n2)
            self.processing_step()
        
        #undo the last action
        if event.key == 'z':
            self.GH.PH.update_action("-z: Undo...")
            print "undoing last action"
            self.undo()
            self.figure.canvas.draw()
            self.set_lim()
            
        #does not work! diameters screwed!
        #if event.key == 'q':
        #    print "entered vein measuring mode"
        #    if self.measure_on:
        #        self.figure.canvas.mpl_disconnect(self.cid_measure)
        #        print "exited measuring mode"
        #    else:   
        #        self.cid_measure = self.figure.canvas.mpl_connect('button_press_event',
        #            self.on_click_measure)
        #        self.measure_on = True
        #        print "entered measuring mode"
            
    
    def modifier_key_on(self,event):
        if event.key == 'shift':
            self.shift_on = True
    def modifier_key_off(self,event):
        if event.key == 'shift':
            self.shift_on = False
    
    def on_click_select(self,event):
        if self.shift_on:
            self.GH.create_node(event.xdata, event.ydata)
            self.GH.PH.update_action("Node created at (%1.2f,%1.2f)"\
                    %(event.xdata,event.ydata))
            self.processing_step()
        else:
            tmp = self.GH.get_node_from_xy(event.xdata, event.ydata)
            if tmp != None:
                n,x_s,y_s = tmp
                x = self.graph.node[n]['x']
                y = self.graph.node[n]['y']
                if n in self.selected_nodes:
                    self.GH.PH.unmark_node(n)
                    del self.selected_nodes[n]
                    self.GH.PH.update_action("Node unmarked at (%1.2f,%1.2f)"\
                        %(x,y))
                else:
                    self.GH.PH.mark_node(n,x_s,y_s)
                    self.selected_nodes.update({n:(x_s,y_s)})
                    self.GH.PH.update_action("Node marked at (%1.2f,%1.2f)"\
                        %(x,y))
                self.figure.canvas.draw()
                self.set_lim()
            else:
                self.GH.PH.update_action("Click nearer!")

    #def on_click_measure(self,event):
    #    self.GH.measure_diameter(event.xdata, event.ydata)
        
    
    #def trace_mode(self):
    #    print """
    #    Create a new part of the network by either clicking
    #    on an existing node or on free space to create a new node.
    #    Each click will create a new node wich will be connected to
    #    the previously placed node.
    #    Press 't' to start tracing
    #    Press 'e' to end tracing
    #    Press ENTER when done.
    #    """
         
    #def trace_key_press(self,event):
    #    return
        
        
        