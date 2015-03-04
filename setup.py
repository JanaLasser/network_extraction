# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 11:11:24 2015

@author: jana
"""

from distutils.core import setup, Extension

setup(name='Network Extraction',
      platforms = 'linux-x86_64',
      version='1.0',
      author='Jana Lasser',
      author_email='jana.lasser@ds.mpg.de',
      url='https://github.com/JanaLasser/network_extraction',
      license='GPL-3.0',
      description='Network Extraction Utilities',
      long_description='''
      The scripts and libraries uploaded in this project are intended
      to be a suite of tools for the extraction and manipulation of network
      data (graphs) from images.
      ''',
      
      packages=['vectorization', 'graph_manipulation'],
      package_data = {'vectorization':'data/*.png'},
      data_files = [('README', 'README.txt')],
      scripts=['scripts/vectorize', 'scripts/graph_edit_GUI'],
                    
      #ext_package = 'vectorization',
      ext_modules = [Extension('test', ['test.c']), \
      Extension('vectorization.C_vectorize_functions',['C_vectorize_functions.c'])]
)
