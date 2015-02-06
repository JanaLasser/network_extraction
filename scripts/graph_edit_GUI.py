# -*- coding: utf-8 -*-
"""
Created on Wed Apr 09 11:30:35 2014

@author: Ista
"""

import argparse
import InterActor

import sys
sys.ps1 = 'SOMETHING'
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', help="Path to the source folder")
    parser.add_argument('-dest', help="Path to the destination folder")
    args = parser.parse_args()
    edt = InterActor.InterActor(args.source,args.dest)
    
