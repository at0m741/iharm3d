################################################################################
#                                                                              #
#  PLOT ONE PRIMITIVE                                                          #
#                                                                              #
################################################################################

import hdf5_to_dict as io
import plot as bplt
from analysis_fns import *

import matplotlib
import matplotlib.pyplot as plt

import sys
import numpy as np

USEARRSPACE=False
UNITS=False

SIZE = 500
#window=[-SIZE,SIZE,-SIZE,SIZE]
#FIGX = 10
#FIGY = 10
window=[0,SIZE,0,SIZE]
FIGX = 12
FIGY = 12

dumpfile = sys.argv[1]
gridfile = sys.argv[2]
var = sys.argv[3]
if UNITS:
  M_unit = float(sys.argv[4])

hdr = io.load_hdr(dumpfile)
geom = io.load_geom(hdr, gridfile)
dump = io.load_dump(dumpfile, hdr, geom)

nplotsx = 2
nplotsy = 2

fig = plt.figure(figsize=(FIGX, FIGY))

# If we're plotting a derived variable, calculate + add it
if var in ['jcov', 'jsq']:
  dump['jcov'] = np.einsum("...i,...ij->...j", dump['jcon'], geom['gcov'][:,:,None,:,n]
  for n in range(hdr['n_dim']):
    dump['jcov'][:,:,:,n] = np.sum(dump['jcon']*geom['gcov'][:,:,None,:,n], axis=3)
  dump['jsq'] = np.sum(dump['jcon']*dump['jcov'], axis=-1)
if var in ['sigma', 'Trt', 'TEMrt']:
  dump['sigma'] = dump['bsq']/dump['RHO']
if var in ['bernoulli', 'Trt', 'TEMrt']:
  dump['bernoulli'] = -T_mixed(dump,0,0) /(dump['RHO']*dump['ucon'][:,:,:,0]) - 1
if var == 'B':
  dump['B'] = np.sqrt(dump['bsq'])
if var in ['gamma', 'Trt', 'TEMrt']:
  dump['gamma'] = get_gamma(geom, dump)
if var == 'Trt':
  dump['Trt'] = T_mixed(dump, 1, 0)
if var == 'TEMrt':
  dump['TEMrt'] = TEM_mixed(dump, 1, 0)

# Add units after all calculations, manually
if UNITS:
  unit = units.get_units_M87(M_unit, tp_over_te=3)

  if var in ['bsq']:
    dump[var] *= unit['B_unit']**2
  elif var in ['B']:
    dump[var] *= unit['B_unit']
  elif var in ['Ne']:
    dump[var] = dump['RHO'] * unit['Ne_unit']
  elif var in ['Te']:
    dump[var] = ref['ME'] * ref['CL']**2 * unit['Thetae_unit'] * dump['UU']/dump['RHO']
  elif var in ['Thetae']:
    # TODO non-const te
    dump[var] = unit['Thetae_unit'] * dump['UU']/dump['RHO']
  # TODO the others

# Plot XY differently for vectors, scalars
if var in ['jcon','ucon','ucov','bcon','bcov']:
  for n in range(4):
    ax = plt.subplot(nplotsy, nplotsx, n+1)
    bplt.plot_xy(ax, geom, np.log10(np.abs(dump[var][:,:,:,n])), arrayspace=USEARRSPACE, window=window)
elif var in ['sigma']:
  ax = plt.subplot(1, 1, 1)
  bplt.plot_xy(ax, geom, dump[var], vmin=0, vmax=10, arrayspace=USEARRSPACE, window=window)
elif var in ['gamma']:
  ax = plt.subplot(1, 1, 1)
  bplt.plot_xy(ax, geom, dump[var], vmin=0, vmax=5, arrayspace=USEARRSPACE, average=True, window=window)
else:
  ax = plt.subplot(1, 1, 1)
  bplt.plot_xy(ax, geom, dump[var], arrayspace=USEARRSPACE, window=window)

plt.tight_layout()

plt.savefig(var+"_xy.png", dpi=100)
plt.close(fig)

fig = plt.figure(figsize=(FIGX, FIGY))

# Plot XZ
if var in ['jcon','ucon','ucov','bcon','bcov']:
  for n in range(4):
    ax = plt.subplot(nplotsx, nplotsy, n+1)
    bplt.plot_xz(ax, geom, dump[var][:,:,:,n], arrayspace=USEARRSPACE, window=window)
elif var in ['sigma']:
  ax = plt.subplot(1, 1, 1)
  bplt.plot_xz(ax, geom, dump[var], vmin=0, vmax=10, arrayspace=USEARRSPACE, window=window)
  bplt.overlay_contours(ax, geom, dump[var], [1])
elif var in ['gamma']:
  ax = plt.subplot(1, 1, 1)
  bplt.plot_xz(ax, geom, dump[var], vmin=0, vmax=5, arrayspace=USEARRSPACE, average=True, window=window)
  bplt.overlay_contours(ax, geom, dump[var], [2.0])
elif var in ['bernoulli']:
  ax = plt.subplot(1, 1, 1)
  bplt.plot_xz(ax, geom, dump[var], arrayspace=USEARRSPACE, average=True, window=window)
  bplt.overlay_contours(ax, geom, dump[var], [0.05])
elif var in ['Trt']:
  ax = plt.subplot(1, 1, 1)
  bplt.plot_xz(ax, geom, np.log10(-dump[var] - dump['RHO']*dump['ucon'][:,:,:,1]), arrayspace=USEARRSPACE, average=True, window=window)
  bplt.overlay_contours(ax, geom, dump['ucon'][:,:,:,1], [0.0], color='r')
  bplt.overlay_contours(ax, geom, dump['sigma'], [1.0], color='b')
  bplt.overlay_contours(ax, geom, dump['gamma'], [1.5], color='tab:purple')
elif var in ['TEMrt']:
  ax = plt.subplot(1, 1, 1)
  bplt.plot_xz(ax, geom, np.log10(-dump[var]), arrayspace=USEARRSPACE, average=True, window=window)
  bplt.overlay_contours(ax, geom, dump['ucon'][:,:,:,1], [0.0], color='r')
  bplt.overlay_contours(ax, geom, geom['r']*dump['ucon'][:,:,:,1], [1.0], color='k')
  bplt.overlay_contours(ax, geom, dump['sigma'], [1.0], color='b')
  bplt.overlay_contours(ax, geom, dump['gamma'], [1.5], color='tab:purple')
  bplt.overlay_contours(ax, geom, dump['bernoulli'], [1.02], color='g')
else:
  ax = plt.subplot(1, 1, 1)
  bplt.plot_xz(ax, geom, dump[var], arrayspace=USEARRSPACE, window=window)

plt.tight_layout()

plt.savefig(var+"_xz.png", dpi=100)
plt.close(fig)
