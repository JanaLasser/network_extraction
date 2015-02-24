The scripts and libraries uploaded in this project are intended to be a suite of tools for the extraction and manipulation of network data (graphs) from images.

*** Data Extraction***
Extraction of network data is done with the vectorize2.py script.
The script takes the path to a binary (black and white) image as input and extracts a graph (networkx graph-object) from the largest connected component on the images.

>> python vectorize2.py /dir/subdir1/subdir2/binary_image.png

Further options to modify the behaviour and output of the script can be obtained by calling

>> python vectorize2.py --help

The script relies on a collection of functions from vectorize_helpers2.py which in turn needs functions from the C_vectorize_functions. The latter is written in cython and then compiled. I have provided the original .pyx file as well as the .c translation into pure C. As I have so far only been able to compile under and for a linux system, I have only upladed the .so file. If you want to use the script under Windows/OSX you have to either compile the library yourself or wait until I figure out how to cross compile and upload them :-)

*** Creating a binary image ***
To get from a grayscale to a suitable binary image can be tricky. I have provided a rudimentary script which does some
contrast improvement and thresholding (binarize.py and binarize_standalone.py). The scripts do the same with the only difference that the standalone does not require the C_vectorize_functions library to work.
The script takes the path to the image it will process as well as an optinal modifier to the value used for thresholding the image

>> python binarize_standalone.py /dir/subdir1/subdir2/grayscale_image.png -t 10

*** Manipulating the extracted graph ***
I have been dealing with networks extracted from pseudo 2D structures a lot. These images are projections onto a plane and therefore might contain network "crossings" which arent real but just created by superposition of two branches.
To correct these "false junctions" I have written a small GUI which allows you to load the extracted graph object, superimpose it onto the original image and modify the graph's structure by manually deleting and creating nodes and edges.
The script to start the GUI is called graph_edit_GUI.py (geGUI) but all the functionality is placed in the three sub-scripts InterActor (dealing with user-interaction), GraphHandler (dealing with manipulations of the graph object) abd PlotHandler (dealing with the dynamic display of changes to the graph on the screen). To run the geGUI you need to specify a folder in which the graph you want to process is located. The folder also needs to hold an euclidean distance map of the binary the graph was created from and the original image for the geGUI to properly start. The graph and the distance map should have been created and named in a way the geGUI can understand if you have used the vectorize2 script to create the graph. In order to not confuse the script (so far it only checks for filename endings) I strongly recommend putting each processed image and the processing results in their own folder. If you did that, simply run the GUI by calling

>> python graph_edit_GUI.py /dir/subdir1/subdir2/dirofimage/  

