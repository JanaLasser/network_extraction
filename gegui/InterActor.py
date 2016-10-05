# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 11:19:04 2014

@author: Jana Lasser
"""
'''
Copyright (C) 2015 Jana Lasser GPL-3.0
'''

#import matplotlib 
#matplotlib.use('qt4agg') 
import matplotlib.pyplot as plt 
plt.ion()
import os
from os.path import join
import GraphHandler
from copy import deepcopy
from PIL import Image
import networkx as nx
import numpy as np
import sys


#####################################################
#               Interaction Class                   #
#####################################################

#helper class to store the current state of the InterActor. Instances of 
#StateSnapshot will be put on a stack and loaded if an "undo" action is
#performed
class StateSnapshot(object):
    def __init__(self,graph,selected_nodes,digraph_on,normalized):
        self.graph = graph
        self.selected_nodes = selected_nodes
        self.digraph_on = digraph_on
        self.normalized = normalized

#helper to load different kinds of files        
def load(filename):
    if filename.endswith(".gpickle"): return nx.read_gpickle(filename)
    elif filename.endswith("dm.png"): return getImage(filename)
    else:
        raise Exception("Unknown filename ending")
        return 

#helper to load images        
def getImage(image_path):    
    image = Image.open(image_path)
    image = image.convert('L')
    image = np.asarray(image,dtype=np.uint8)
    return np.flipud(image)

#function to print a hopefully helpful description of what the GUI needs to run
#correctly. It will be printed every time the GUI cannot find one of the files
#it requires.    
def printHelp():
    print('''\nThe script will look for three different files it needs to load
             within the directory it is passed at start and will fail if these
             files can not be found. The files are:
             
                 - a file with the format ".gpickle",
                   this is the graph-object we will be working on
      
                 - a file with the ending "_orig" (format does not matter)
                  this is the original microscopy image which will be
                  superimposed onto the visualization of the graph to assist
                  with correct recognition of junctions
      
                 - a file with the ending "_dm" (format does not matter)
                   this is the distance-map of the binary used to create the
                   graph object it is needed in case we want to create new
                   nodes so the thickness of the nodes can automatically be
                   determined\n.
''')

#Class that handles interaction with the user. It will create instances of
#a class for handling of a graph object (the GraphHandler) and handling of
#the plotting i.e. the output visible to the user on the screen (PlotHandler).
#The class supports several features like the selection, deletion and creation
#of nodes, display of all cycles in the graph, undo of the last action,
#creation of a digraph, removal of all redundant nodes and saving of the graph.
class InterActor(object):
      
    def __init__(self,source_path,dest_path):
        self.state_stack = []                                                   #stores snapshots of current states
        self.max_stored_states = 10                                             #defines the maximum number of possible undo actions

        work_name = os.path.basename(source_path)                               #name of the directory we are working in
        self.source_path = source_path                                          #path to the working directory
        if dest_path == None: dest_path = source_path                           #if specified: path to the directory files will be saved to
            
        self.name_dict = {'work_name':work_name,                                #store names in dictionary to make things cleaner
                          'source_path':source_path,
                          'dest_path':dest_path}
                          
        self.graph = None                                                       #try to load the graph object
        for f in os.listdir(self.name_dict['source_path']):                     #look for a file that ends with '_red1.gpickle'
            if f.endswith('.gpickle'):                                          #this corresponds to a graph which has some redundant nodes left
                self.graph = load(join(self.name_dict['source_path'],f))
        if self.graph == None:                                                  #if no suitable file is found, print an error message
            print 'gegui> No corresponding .gpickle-file found!'\
            +'\ngegui> Closing the GUI.'
            printHelp()                                                         #print the help message and exit the GUI
            sys.exit()
        if len(self.graph.nodes()) > 1000:
            print '\n'
            print 'gegui> *** WARNING: Loaded graph is quite large (> 1000 nodes).'
            print 'gegui> Large graphs slow down or even freeze gegui!. ***'
            print '\n'
            
        self.selected_nodes = {}                                                #stores the keys of the currently selected nodes
        self.mode_list = []                                                     #list of currently switched on modes which will be displayed at the top of the figure
        
        #state flags
        self.select_on = False                                                  #node selection mode on?
        self.digraph_on = False                                                 #is the graph already a digraph?
        self.normalized = False                                                 #has the graph already been streamlined? (redundant nodes removed)
        self.shift_on = False                                                   #is the shift key pressed? (ugly but working modifier handling...)
        
        #figure initialization                       
        self.dpi = 100                                                          #resolution, increase for super crowded graphs
        self.figsize = (12,12)                                                  #size of the figure, tinker with this if the displayed figure does not fit your monitor
        self.figure = plt.figure(dpi=self.dpi,figsize=self.figsize) 
      
        #initialize state stack and mode
        self.CurrentStateSnapshot = StateSnapshot(self.graph,                   #take an initial snapshot and store it
            self.selected_nodes, self.digraph_on,self.normalized)
        self.update_state_stack()
        
        self.GH = GraphHandler.GraphHandler(self.figure,self.name_dict,         #initialize the graph handler
                                            self.CurrentStateSnapshot)

        self.update_mode("Image: " + self.name_dict['work_name'],'add')         #update the display at the top of the figure   
        self.edit_loop()                                                        #start the main interaction loop     
        
    #function to clear the current selection of nodes    
    def clear_selection(self):
        self.selected_nodes = {}
        self.GH.clear_selection()

    #helper which is called after every processing step. It redraws the
    #canvas to update the displayed figure and updates the state stack.
    def processing_step(self):
        self.figure.canvas.draw()
        self.set_lim()
        self.update_state_stack()
        
    #updates the mode displayed at the top of the figure
    def update_mode(self,text,action):
        if action == 'rm':
            self.mode_list.remove(text)
        elif action == 'add':
            self.mode_list.append(text)
        else:
            print "gegui> action not recognized!"          
        s = ""
        for item in self.mode_list:
            s += item + "\n"
        s.rstrip('\n')
        self.GH.PH.update_mode(s)

    #intendet to fix the size of the figure but disables zoom so not functional
    #at the moment.
    def set_lim(self):
        #plt.xlim((0,self.GH.PH.width))
        #plt.ylim((0,self.GH.PH.height))
        pass

    #creates a new snapshot of the current state and saves it in state stack    
    def update_state_stack(self):
        self.CurrentStateSnapshot=StateSnapshot(self.graph,self.selected_nodes,
            self.digraph_on,self.normalized)
        if len(self.state_stack) > self.max_stored_states:
            del self.state_stack[0]
            self.state_stack.append(deepcopy(self.CurrentStateSnapshot))
        else:
            self.state_stack.append(deepcopy(self.CurrentStateSnapshot))
    
    #exposes the "undo" functionality of the GUI       
    def undo(self):
        if len(self.state_stack) > 1:
            self.state_stack.pop()
            try:
                self.CurrentStateSnapshot = self.state_stack[-1] 
            except IndexError:
                print "gegui> no more undo-operations stored!"                  
            self.figure.clf()
            self.graph = self.CurrentStateSnapshot.graph
            self.selected_nodes = self.CurrentStateSnapshot.selected_nodes
            self.digraph_on = self.CurrentStateSnapshot.digraph_on
            self.normalized = self.CurrentStateSnapshot.normalized
            self.GH = GraphHandler.GraphHandler(self.figure, self.name_dict,\
                                                self.CurrentStateSnapshot)
        else:
            print "gegui> no more undo-operations stored!"
    
    #main interaction loop    
    def edit_loop(self):
        self.print_help_message()
        while True:
            cmd = raw_input("gegui> ")

            #leave and ask if we want to save before leaving
            if cmd == 'x':                
                while True:
                    cmd2 = raw_input("gegui> Want to save before leaving? 'y/n' ")
                    if cmd2 == 'y':
                        self.GH.save_graph()
                        return
                    elif cmd2 == 'n':
                        return
                    else:
                        print 'gegui> input not recognized!'
             
            #save current progress
            elif cmd == 's':
                self.GH.save_graph()
            
            #enter manual graph manipulation mode
            elif cmd == 'm':
                self.manipulation_mode()
            
            #streamline graph if not already done
            elif cmd == 'norm':
                if not self.normalized:
                    self.GH.streamline_graph()
                    self.normalized = True
                    self.update_state_stack()
                else:
                    print "gegui> already streamlined!"
            
            #create digraph with the currently selected node as root
            elif cmd == 'digraph':
                if len(self.selected_nodes) != 1:
                    print "Select exactly one node for di-Graph creation!"
                elif self.digraph_on == True:
                    print "gegui> Already a digraph!"
                else:
                    self.GH.create_digraph(list(self.selected_nodes)[0]) 
                    self.digraph_on = True
                    self.update_state_stack()
            
            #display help again
            elif cmd == 'h':
                self.print_help_message()
            
            #unknown user input
            else:
                print "gegui> Command not recognized."
                
    def print_help_message(self):
        print """
        Available commands:
        
        s       - Save current graph
        m       - Enter manipulation mode
        norm    - Streamline graph (remove all redundant nodes)
        digraph - Create and save digraph
        h       - Print help message
        x       - Exit
        """
    
    def manipulation_mode(self):
        print """
        b       - activate node manipulation:
                  Select and deselect nodes by clicking on them.
                  Create new nodes by shift-clicking on the canvas.
        m       - show cycles in the graph
        a       - clear the current node selection
        d       - delete selected node(s) and adjacent edges
        e       - connect two selected nodes
        z       - undo the last action
        ENTER   - leave manipulation mode
       
        """     
        self.update_mode("-m: Manipulation Mode","add") 
        self.figure.canvas.draw()   
        cid_press = self.figure.canvas.mpl_connect('key_press_event',
                 self.manipulation_key_press)                 
        raw_input()
        
        if self.select_on:
            self.update_mode("-b: Node Manipulation activated",'rm')
            self.figure.canvas.mpl_disconnect(self.cid_click)
            self.select_on = False
        self.update_mode("-m: Manipulation Mode",'rm')
        self.figure.canvas.draw()
        self.figure.canvas.mpl_disconnect(cid_press)
    
    #function to detect key presses in manipulation mode                
    def manipulation_key_press(self,event):      
        #activate node selection mode
        if event.key == 'b':
            if self.select_on:
                self.update_mode("-b: Node Manipulation activated",'rm')
                self.figure.canvas.draw()
                self.figure.canvas.mpl_disconnect(self.cid_click)
                self.select_on = False
            else:
                self.update_mode("-b: Node Manipulation activated",'add')
                self.figure.canvas.draw()
                self.cid_mod1 = self.figure.canvas.mpl_connect(\
                    'key_press_event', self.modifier_key_on)
                self.cid_mod1 = self.figure.canvas.mpl_connect(\
                    'key_release_event', self.modifier_key_off)
                self.cid_click = self.figure.canvas.mpl_connect(\
                    'button_press_event', self.on_click_select)
                self.select_on = True
        
        #show cylces in the graph
        if event.key == 'm':
            print "gegui> looking for cycles"
            cycles = self.GH.detect_cycles()
            if cycles != None:
                self.GH.PH.update_action("-m: Cycles marked")
                self.selected_nodes.update(cycles) 
                self.processing_step()
            else:
                self.GH.PH.update_action("-m: No Cycles detected")
                self.figure.canvas.draw()
                print "gegui> no cycles detected!"
        
        #clear the current node selection
        if event.key == 'a':
            print "gegui> clearing current selection:"
            self.GH.PH.update_action("-a: Selection cleared")
            self.clear_selection()
            self.processing_step()
         
        #delete currently selected nodes
        if event.key == 'd':
            print "gegui> deleting selected nodes:"
            self.GH.PH.update_action("-d: Selected nodes deleted")
            self.processing_step()
            for n in self.selected_nodes:
                self.GH.delete_node(n)
            self.clear_selection()
            self.processing_step()
                          
        #create a new edge between two selected nodes    
        if event.key == 'e':
            print "gegui> connecting selected nodes"
            if len(self.selected_nodes) != 2:
                self.GH.PH.update_action("-e: Not exactly two nodes selected!")
                print('gegui> more than two nodes selected, aborting...')
                return
            n1 = self.selected_nodes.keys()[0]
            n2 = self.selected_nodes.keys()[1]
            self.GH.create_edge(n1,n2)
            self.processing_step()
        
        #undo the last action
        if event.key == 'z':
            self.GH.PH.update_action("-z: Undo...")
            print "gegui> undoing last action"
            self.undo()
            self.figure.canvas.draw()
            self.set_lim()            
 
    #handling of "shift" key modifier   
    def modifier_key_on(self,event):
        if event.key == 'shift':
            self.shift_on = True
    def modifier_key_off(self,event):
        if event.key == 'shift':
            self.shift_on = False
    
    #handling of node selection via clicks on the figure
    def on_click_select(self,event):
        if self.shift_on:                                                       #detect if shift is currently pressed
            self.GH.create_node(event.xdata, event.ydata)                       #if yes, create a new node
            self.processing_step()
        else:
            tmp = self.GH.get_node_from_xy(event.xdata, event.ydata)            #if shift is not pressed, try to mark the node nearest to the click
            if tmp != None:                                                     #if we found a node in the vicinity, mark it
                n,x_s,y_s = tmp
                x = self.graph.node[n]['x']
                y = self.graph.node[n]['y']
                if n in self.selected_nodes:                                    #if the node is already marked, unmark it
                    self.GH.PH.unmark_node(n)
                    del self.selected_nodes[n]
                    self.GH.PH.update_action("Node unmarked at (%1.2f,%1.2f)"\
                        %(x,y))
                else:                                                           #if not, mark it
                    self.GH.PH.mark_node(n,x_s,y_s)
                    self.selected_nodes.update({n:(x_s,y_s)})
                    self.GH.PH.update_action("Node marked at (%1.2f,%1.2f)"\
                        %(x,y))
                self.figure.canvas.draw()
                self.set_lim()
            else:                                                               #if we didn't find a node, complain
                self.GH.PH.update_action("Click nearer!")
        
        