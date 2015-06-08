Disclaimer: The scripts and libraries uploaded in this project are intended to be a suite of tools for the extraction, manipulation and analysis of network data (graphs) from images. The scripts uploaded here are part of a methods paper detailing the algorithms used which is soon to be published.
The original images used as examples in the paper can be found in the /data directory. The images represent a range of different use-cases for NEAt both from different research projects as well as with regards to the size of the networks they contain. They are ready to be processed without any further modifications.

*** Setting up the NEAT framework ***
-------------------------------------

The neat framework is a set of python scripts that make use of the python module neat_helpers and the cythonized library C_neat_functions.
For neat to work you will need three things
1) a working installation of python 2.7 and pip
2) the third party libraries utilized within NEAT (scipy, numpy, matplotlib, cython, networkx, scikit-image, opencv, pillow, shapely)
3) a version of the C_neat_functions.pyx file compiled for your platform
In the following we detail how to get these three things on Linux, Windows and MacOS platforms.

-- Linux --
1) Python 2.7 should be installed by default, if it is not, your package manager most likely will provide it.
You can get pip by running  

wget https://bootstrap.pypa.io/ez_setup.py -O - | sudo python

(More detailed instructions on how to get pip can be found here: https://pypi.python.org/pypi/setuptools#unix-wget)
 

2) For numpy (and some other libraries) you will need the python-dev package, get it by running

sudo apt-get install python-dev 

Now you have pip, it should be easy to install most of the other libraries. For example to install networkx, simply run

pip install networkx

Repeat this process with scipy, matplotlib, cython, scikit-image, pillow and shapely. 
To install opencv, follow the instructions detailed here http://docs.opencv.org/doc/tutorials/introduction/linux_install/linux_install.html.

3) To compile your own C_neat_functions.so file, navigate to the /neat directory and run

python setup_C_neat_functions.py build_ext --inplace

This should create two files called C_neat_functions.so and C_neat_functions.c, the first you need, the latter you can safely delete.

-- Windows --
1) Download the latest python 2.7 release from https://www.python.org/downloads/ and install it.
To get pip, if you are on windows 8 or later, start Powershell and run

(Invoke-WebRequest https://bootstrap.pypa.io/ez_setup.py).Content | python -

If you have an earlier windows version, download ez_setup.py from https://bootstrap.pypa.io/ez_setup.py,
navigate to the directory you saved the file in and run

python ez_setup.py

(More detailed instructions on how to get pip can be found here: https://pypi.python.org/pypi/setuptools#unix-wget).

2) Thankfully, Christoph Gohlke maintains a site with a wide collection of python packages for windows. Go to
http://www.lfd.uci.edu/~gohlke/pythonlibs/ and download the .whl files for scipy, numpy, matplotlib, cython, networkx, scikit-image,
 opencv, pillow and shapely that fit your platform. Install the packages by navigating to the directory you saved them in and run

pip install SomePackage.whl

for all the packages you downloaded.

3) Navigate to the folder /neat and run

python setup_C_neat_functions.py build_ext --inplace

A file named "C_neat_functions.pyc" should have bean created in the \neat directory
Compilation might be abortet with the message "error: Unable to find vcvarsall.bat"
There is a relatively easy fix for that (based on this stack overflow post http://stackoverflow.com/questions/2817869/error-unable-to-find-vcvarsall-bat):
Install Visual studio 2013 and set the environment variable by running

SET VS90COMNTOOLS=%VS120COMNTOOLS% 

Rerun python setup_c_neat_functions.py build_ext --inplace

If it still does not work, good luck!

-- Mac OSX --
1) Download and install the latest python 2.7 release from https://www.python.org/downloads/ and install it.
Depending on whether you have access to easy_install or curl, run one of the following

sudo easy_install pip
curl https://bootstrap.pypa.io/ez_setup.py -o - | python

(More detailed instructions on how to get pip can be found here: https://pypi.python.org/pypi/setuptools#unix-wget)

2) For numpy (and some other libraries) you will need the python-dev package, get it by running

sudo apt-get install python-dev 

Now you have pip, it should be easy to install most of the other libraries. For example to install networkx, simply run

pip install networkx

Repeat this process with scipy, matplotlib, cython, scikit-image, pillow and shapely. 
To install opencv, follow the instructions detailed here http://tilomitra.com/opencv-on-mac-osx/.

3) Navigate to the folder /neat and run

python setup_C_neat_functions.py build_ext --inplace

This should create two files called C_neat_functions.so and C_neat_functions.c, the first you need, the latter you can safely delete.


*** Extracting network data using the NEAT framework ***
--------------------------------------------------------

The network extraction and analysis tool (NEAT) is intended for the extraction of network data from images which can later be analyzed easily.
The workflow is broken down into four steps represented by four processing scripts

1) Segment the original image into foreground and background and therefore create a binary image using binarize.py
2) Extract the network from the binary image unsing neat.py
3) Optional: manually correct errors in the network or remove artifacts using the graph-edit GUI gegui.py
4) Analyze basic characteristics of the network with analyze.py

-- binarize --
To get from a grayscale to a suitable binary image can be tricky. I have provided a rudimentary script which does some contrast improvement and thresholding (binarize.py and binarize_standalone.py). The scripts do the same with the only difference that the standalone does not require the C_neat_functions library to work.
The script takes the path to the image it will process as well as an optinal modifier to the value used for thresholding the image as command line arguments

>> python binarize_standalone.py /dir/subdir1/subdir2/grayscale_image.png -t 10

As with NEAT, other formats as .png as supported as well.


-- neat --
Extraction of network data is done with the neat.py script. NEAT takes the path to a binary (black and white) image as required command line argument and extracts a graph (networkx graph-object) from the largest connected component in the image. 

>> python neat.py /dir/subdir1/subdir2/binary_image.png

NOTE: the format of the input image does not have to be .png, every format supported by the python image library (PIL) can be used.
Further information about the options available to modify the behaviour and output of NEAT can be obtained by calling

>> python neat.py --help

The script relies on a collection of functions from neat_helpers.py which in turn needs functions from the C_neat_functions. The latter is written in cython and then compiled for speed. I have provided the original .pyx file as well as the .c file. As I have so far only been able to compile under and for a linux system and therefore only upladed the .so file. If you want to use the script under Windows/OSX you have to either compile the library yourself or wait until I figure out how to cross compile and upload them :-) In the meantime there is a de-cythonized version of the library available called C_neat_functions.py which is slower an very ugly for python code but contains all the functionality of the cythonized version.


-- gegui --
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


-- analyze --







