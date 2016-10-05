import numpy as np
from os import getcwd
from os.path import join
import matplotlib.pyplot as plt

#helper function to calculate the deviation normalized by the original
#network's value
def deviation(o,v):
	return np.abs(o - v)/o

#load network statistics as str because the file contains mixed datatypes
cwd = getcwd()
data = np.loadtxt(join(cwd,'validation-statistics.txt'),dtype=str)

#extract data and convert to float
labels = data[0:,0]
N_o = data[0:,1].astype(np.float)
N_v = data[0:,2].astype(np.float)
L_o = data[0:,3].astype(np.float)
L_v = data[0:,4].astype(np.float)
Wm_o = data[0:,5].astype(np.float)
Wr_o = data[0:,6].astype(np.float)
Wm_v = data[0:,7].astype(np.float)
Wr_v = data[0:,8].astype(np.float)
A_o = data[0:,9].astype(np.float)
A_v = data[0:,10].astype(np.float)
A_i = data[0:,11].astype(np.float)
D_px = data[0:,12].astype(np.float)

#calculate deviations
N_deviation = deviation(N_o,N_v)
L_deviation = deviation(L_o,L_v)
Wm_deviation = deviation(Wm_o, Wm_v)
Wr_deviation = deviation(Wr_o, Wr_v)
A_deviation = A_i / A_o

#format handling for the result file
header_format = '{:<40}{:^12}{:^12}{:^12}{:^12}{:^12}{:^12}'
header_values = ['#label','#S_N','#S_L','#S_R','#S_r','#S_A','#D']
entry_format = '{:<40}{:^12.3f}{:^12.3f}{:^12.3f}{:^12.3f}{:^12.3f}{:^12.3f}'

#dump everything into a result file
deviations = open(join(cwd,'deviations.txt'),'w+')
print >> deviations, header_format.format(*header_values)
for d in data:
	n_o = float(d[1])
	n_v = float(d[2])
	l_o = float(d[3])
	l_v = float(d[4])
	wm_o = float(d[5])
	wm_v = float(d[7])
	wr_o = float(d[6])
	wr_v = float(d[8])
	a_o = float(d[9])
	a_v = float(d[10])
	a_i = float(d[11])
	d_px = float(d[12])
	entry = [d[0], deviation(n_o,n_v),deviation(l_o,l_v),deviation(wm_o, wm_v),\
			deviation(wr_o, wr_v),a_i/a_o, d_px]
	print >> deviations, entry_format.format(*entry)

deviations.close()

#plot deviation distributions
plt.hist(N_deviation,bins=20)
plt.xlabel('$\\sigma_N [\%]$')
plt.ylabel('counts')
plt.savefig(join(cwd,'N_deviation.pdf'),transparent=True)
plt.close()

plt.hist(L_deviation,bins=20)
plt.xlabel('$\\sigma_L [\%]$')
plt.ylabel('counts')
plt.savefig(join(cwd,'L_deviation.pdf'),transparent=True)
plt.close()

plt.hist(Wm_deviation,bins=20)
plt.xlabel('$\\sigma_R [\%]$')
plt.ylabel('counts')
plt.savefig(join(cwd,'Wm_deviation.pdf'),transparent=True)
plt.close()

plt.hist(Wr_deviation,bins=20)
plt.xlabel('$\\sigma_r [\%]$')
plt.ylabel('counts')
plt.savefig(join(cwd,'Wr_deviation.pdf'),transparent=True)
plt.close()

plt.hist(A_deviation,bins=20)
plt.xlabel('$\\sigma_A [\%]$')
plt.ylabel('counts')
plt.savefig(join(cwd,'A_deviation.pdf'),transparent=True)
plt.close()

plt.hist(D_px,bins=20)
plt.xlabel('$D [\%]$')
plt.ylabel('counts')
plt.savefig(join(cwd,'D_px.pdf'),transparent=True)
plt.close()

#output a summary of the deviations to the command line
print('Deviations:')
print('Number of nodes: {} pm {}'.format(N_deviation.mean(),N_deviation.std()))
print('Network length: {} pm {}'.format(L_deviation.mean(),L_deviation.std()))
print('Mean edge weight: {} pm {}'.format(Wm_deviation.mean(),Wm_deviation.std()))
print('Edge weight ratio: {} pm {}'.format(Wr_deviation.mean(),Wr_deviation.std()))
print('Convex hull deviation: {} pm {}'.format(A_deviation.mean(),A_deviation.std()))
print('Pixel difference: {} pm {}'.format(D_px.mean(),D_px.std()))