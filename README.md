The scripts and libraries uploaded in this project are intended to be a suite of tools for the extraction and manipulation of network data (graphs) from images.

*** Data Extraction using NEAT***
The network extraction and analysis tool (neat) is intended for the extraction of network data from images which can later be analyzed easily.
Extraction of network data is done with the neat.py script. NEAT takes the path to a binary (black and white) image as required command line argument and extracts a graph (networkx graph-object) from the largest connected component in the image. 

>> python neat.py /dir/subdir1/subdir2/binary_image.png

NOTE: the format of the input image does not have to be .png, every format supported by the python image library (PIL) can be used.
Further information about the options available to modify the behaviour and output of NEAT can be obtained by calling

>> python neat.py --help

The script relies on a collection of functions from neat_helpers.py which in turn needs functions from the C_neat_functions. The latter is written in cython and then compiled for speed. I have provided the original .pyx file as well as the .c file. As I have so far only been able to compile under and for a linux system and therefore only upladed the .so file. If you want to use the script under Windows/OSX you have to either compile the library yourself or wait until I figure out how to cross compile and upload them :-) In the meantime there is a de-cythonized version of the library available called C_neat_functions.py which is slower an very ugly for python code but contains all the functionality of the cythonized version.



*** Creating a binary image ***

To get from a grayscale to a suitable binary image can be tricky. I have provided a rudimentary script which does some contrast improvement and thresholding (binarize.py and binarize_standalone.py). The scripts do the same with the only difference that the standalone does not require the C_neat_functions library to work.
The script takes the path to the image it will process as well as an optinal modifier to the value used for thresholding the image as command line arguments

>> python binarize_standalone.py /dir/subdir1/subdir2/grayscale_image.png -t 10

As with NEAT, other formats as .png as supported as well.



*** Manipulating the extracted graph using the GeGUI ***

I have been dealing with networks extracted from pseudo 2D structures a lot. These images are projections onto a plane and therefore might contain network "crossings" which arent real but just created by superposition of two branches.
To correct these "false junctions" I have writtenthe GeGUI (graph edit GUI) which allows you to load the extracted graph object, superimpose it onto the original image and modify the graph's structure by manually deleting and creating nodes and edges.
The script to start the GUI is called GeGUI.py but all the functionality is placed in the three sub-scripts InterActor (dealing with user-interaction), GraphHandler (dealing with manipulations of the graph object) abd PlotHandler (dealing with the dynamic display of changes to the graph on the screen). To run the GeGUI you need to specify a folder in which the graph you want to process is located. The folder also needs to hold an euclidean distance map of the binary the graph was created from and the original image for the GeGUI to properly start. Therefore the folder should look like this:

dirofimage
	original_image.tif
	extracted_graph_red1.gpickle
	distance_map_dm.png

So far the GeGUI looks for any .tif file in dirofimage and treats it as the original (the actual name of the file does not matter). For the graph and the distance map the GeGUI looks for file names which end with "_red1.gpickle" and "dm.png" respectively and loads them.
The graph and the distance map should have been created and named in a way the geGUI can understand if you have used the NEAT to create the graph. In order to not confuse the script I strongly recommend putting each processed image and the processing results in their own folder. If you did that, simply run the GUI by calling

>> python GeGUI.py /dir/subdir1/subdir2/dirofimage/  

