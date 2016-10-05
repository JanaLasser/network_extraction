#extracts a graph from every 'original_binary' file at the destination with r=1 and p=2
#saves the extracted graph and a pdf-plot of it at the destination
for f in $(find validation-graphs-tracheoles/ -name '*original_binary.png')
do 
 echo "Processing $f"
 python ../net/net.py "$f" -r 1 -p 2 -dest '/media/jlasser/LabDataJana/publications/2016_NET-SCBM-revised/clean-net-repo/validation/validation-graphs-tracheoles' -plt
done

#creates a new and artificial image with the known graph blurred and plotted on top
#of a noisy background for each graph at the destination.
python create-validation-images.py

#creates a binary for every 'validation_image' at the destination using the binarize_adaptive
#script with t = 51, g = 3, s = 3 and m = 30000
#saves the extracted graph and a pdf-plot of it at the destination
for f in $(find validation-graphs-tracheoles/ -name '*validation_image.png')
do 
 echo "Processing $f"
 python ../binarize/binarize_adaptive.py "$f" -t 51 -g 3 -s 3 -m 30000 -c 80 -dest '/media/jlasser/LabDataJana/publications/2016_NET-SCBM-revised/clean-net-repo/validation/validation-graphs-tracheoles'
done

#extracts a graph from every 'validation_binary' file at the destination with r=1 and p=2
#saves the extracted graph and a pdf-plot of it at the destination
for f in $(find validation-graphs-tracheoles/ -name '*validation_image_binary.png')
do 
 echo "Processing $f"
 python ../net/net.py "$f" -r 1 -p 2 -dest '/media/jlasser/LabDataJana/publications/2016_NET-SCBM-revised/clean-net-repo/validation/validation-graphs-tracheoles' -plt
done

#calculates statistics for both the original and the newly created validation graph, saves
#the statistics in the file 'validation-statistics.txt'
python calculate-validation-statistics.py

#compares the number of nodes, length of network, area of convex hull, edge weight and pixel
#distance of the original and the validation graph, saves the deviations in the file 'deviations.txt'
#and prints a summary to the command line
python analyze-validation-statistics.py

