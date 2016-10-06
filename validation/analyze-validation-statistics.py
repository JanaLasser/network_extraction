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
R_o = data[0:,5].astype(np.float)
r_o = data[0:,6].astype(np.float)
R_v = data[0:,7].astype(np.float)
r_v = data[0:,8].astype(np.float)
A_o = data[0:,9].astype(np.float)
A_v = data[0:,10].astype(np.float)
A_i = data[0:,11].astype(np.float)
D_px = data[0:,12].astype(np.float)

#calculate deviations
N_deviation = deviation(N_o,N_v)
L_deviation = deviation(L_o,L_v)
R_deviation = deviation(R_o, R_v)
r_deviation = deviation(r_o, r_v)
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
fig, ax = plt.subplots()
ax.grid('on')
ax.hist(N_deviation,bins=50,color='FireBrick',rwidth=0.8, linewidth=0)
ax.set_title('Node number error $\\sigma_N$')
ax.set_xlabel('$\\sigma_N = \\frac{|N_o - N_v|}{N_o}$')
ax.set_ylabel('counts')
plt.savefig(join(cwd,'N_error_distribution.pdf'),transparent=True)
plt.close()

fig, ax = plt.subplots()
ax.grid('on')
ax.hist(L_deviation,bins=50,color='FireBrick',rwidth=0.8, linewidth=0)
ax.set_title('Network length error $\\sigma_L$')
ax.set_xlabel('$\\sigma_L = \\frac{|L_o - L_v|}{L_o}$')
ax.set_ylabel('counts')
plt.savefig(join(cwd,'L_error_distribution.pdf'),transparent=True)
plt.close()

fig, ax = plt.subplots()
ax.grid('on')
ax.hist(R_deviation,bins=50,color='FireBrick',rwidth=0.8, linewidth=0)
ax.set_title('Mean weight error $\\sigma_{\\bar{R}}$')
ax.set_xlabel('$\sigma_{\\bar{R}} = \\frac{|\\bar{R}_o - \\bar{R}_v|}{\\bar{R}_o}$')
ax.set_ylabel('counts')
plt.savefig(join(cwd,'R_error_distribution.pdf'),transparent=True)
plt.close()

fig, ax = plt.subplots()
ax.grid('on')
ax.hist(r_deviation,bins=50,color='FireBrick',rwidth=0.8, linewidth=0)
ax.set_title('Weight ratio error $\\sigma_r$')
ax.set_xlabel('$\sigma_r = \\frac{|r_o - r_v|}{r_o}$')
ax.set_ylabel('counts')
plt.savefig(join(cwd,'r_error_distribution.pdf'),transparent=True)
plt.close()

fig, ax = plt.subplots()
ax.grid('on')
ax.hist(D_px,bins=50,color='FireBrick',rwidth=0.8, linewidth=0)
ax.set_title('Pixel-wise difference $\\sigma_D$')
ax.set_xlabel('$\\sigma_D = \\frac{|D_o - D_v|}{D_o}$')
ax.set_ylabel('counts')
plt.savefig(join(cwd,'D_error_distribution.pdf'),transparent=True)
plt.close()

#output a summary of the deviations to the command line
print('Deviations:')
print('Number of nodes: {} pm {}'.format(N_deviation.mean(),N_deviation.std()))
print('Network length: {} pm {}'.format(L_deviation.mean(),L_deviation.std()))
print('Mean edge weight: {} pm {}'.format(R_deviation.mean(),R_deviation.std()))
print('Edge weight ratio: {} pm {}'.format(r_deviation.mean(),r_deviation.std()))
print('Convex hull deviation: {} pm {}'.format(A_deviation.mean(),A_deviation.std()))
print('Pixel difference: {} pm {}'.format(D_px.mean(),D_px.std()))