# -*- coding: utf-8 -*-
"""
Created on Wed Apr 09 11:30:35 2014

@author: Ista
"""

import argparse
import InterActor
import sys

#dirty hack to turn on interactive mode in most recent matplitlib version
sys.ps1 = 'SOMETHING'


'''
Calling the graph_edit_GUI with the right arguments is easy but making sure
it will correctly load the right files is not that trivial.
To start the script, simply call it with the path to a directory.
The script will look for three different files it needs to load within that
directory and will fail if the files can not be found.
The files are:
    - a file with the ending "_red1.gpickle"
      this is the graph-object we will be working on
      
    - a file with the ending ".tif" or "_orig" 
      (format does not matter in this case)
      this is the original microscopy image which will be superimposed onto the
      visualization of the graph to assist with correct recognition of junctions
      
    - a file with the ending "_dm.png"
      this is the distance map of the binary used to create the graph object
      it is needed in case we want to create new nodes so the thickness of the
      nodes can automatically be determined
'''
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='graph_edit_GUI - a GUI ' +\
    'for manual manipulation of graphs in networkx format. Especially craeted,'\
    + ' to correct surplus junctions in 2D projetions of 3D structures.\n'\
    
    + '\nCopyright (C) 2015 Jana Lasser')
    parser.add_argument('source', help="Path to the source folder")
    parser.add_argument('-dest', help="Path to the destination folder")
    args = parser.parse_args()
    edt = InterActor.InterActor(args.source,args.dest)
    
