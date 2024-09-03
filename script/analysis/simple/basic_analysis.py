import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os, psutil, sys
import h5py
from mpl_toolkits.axes_grid1 import make_axes_locatable
import multiprocessing as mp

# Parallelize analysis by spawning several processes using multiprocessing's Pool object
def run_parallel(function, dlist, nthreads, args):
    pool = mp.Pool(nthreads)
    pool.map_async(function, [(dumpval, *args) for dumpval in dlist]).get(720000)
    pool.close()
    pool.join()

# Function to generate poloidal (x,z) slice
def xz_slice(var, grid, patch_pole=False, average=False):
    xz_var = np.zeros((2 * grid['n1'], grid['n2']))
    if average:
        var = np.mean(var, axis=2)
        for i in range(grid['n1']):
            xz_var[i,:] = var[grid['n1'] - 1 - i,:]
            xz_var[i + grid['n1'],:] = var[i,:]
    else:
        angle = 0.; ind = 0
        for i in range(grid['n1']):
            xz_var[i,:] = var[grid['n1'] - 1 - i,:, ind + grid['n3'] // 2]
            xz_var[i + grid['n1'],:] = var[i,:, ind]
    if patch_pole:
        xz_var[:,0] = xz_var[:,-1] = 0
    return xz_var

# Function to overlay field lines
def plotting_bfield_lines(ax, B1, B2, grid, nlines=20):
    xp = xz_slice(grid['x'], grid, patch_pole=True)
    zp = xz_slice(grid['z'], grid)
    B1_phi_avg = B1.mean(axis=-1) 
    B2_phi_avg = B2.mean(axis=-1)
    AJ_phi = np.zeros([2 * grid['n1'], grid['n2']]) 
    for j in range(grid['n2']):
        for i in range(grid['n1']):
            AJ_phi[grid['n1'] - 1 - i, j] = AJ_phi[i + grid['n1'], j] = (
                np.trapz(grid['gdet'][:i,j,0] * B2_phi_avg[:i,j], dx=grid['dx1']) - 
                np.trapz(grid['gdet'][i,:j,0] * B1_phi_avg[i,:j], dx=grid['dx2'])
            )
    AJ_phi -= AJ_phi.min()
    levels = np.linspace(0, AJ_phi.max(), nlines * 2)
    ax.contour(xp, zp, AJ_phi, levels=levels, colors='k')

# Analysis function for torus2d
def analysis_torus2d(args):
    dumpval, dumpsdir, grid = args
    plt.clf()
    print(f"Analyzing {dumpval:04d} dump")
    dfile = h5py.File(os.path.join(dumpsdir, f'dump_0000{dumpval:04d}.h5'), 'r')
    rho = dfile['prims'][()][Ellipsis, 0]
    uu = np.array(dfile['prims'][()][Ellipsis, 1])
    u = np.array(dfile['prims'][()][Ellipsis, 2:5])
    B = np.array(dfile['prims'][()][Ellipsis, 5:8])
    gam = np.array(dfile['header/gam'][()])
    t = dfile['t'][()]
    dfile.close()
    t = f"{t:.3f}"
    logrho = np.log10(rho)
    pg = (gam - 1) * uu

    xp = xz_slice(grid['x'], grid, patch_pole=True)
    zp = xz_slice(grid['z'], grid)
    rhop = xz_slice(logrho, grid)

    fig = plt.figure(figsize=(16, 9))
    heights = [1, 5]
    gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=heights, figure=fig)

    ax0 = fig.add_subplot(gs[0, :])
    ax0.annotate(f't= {t}', xy=(0.5, 0.5), xycoords='axes fraction', va='center', ha='center', fontsize='x-large')
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[1, 0])
    rhopolplot = ax1.pcolormesh(xp, zp, rhop, cmap='hot', vmin=-5, vmax=0, shading='gouraud')
    plotting_bfield_lines(ax1, B[Ellipsis, 0], B[Ellipsis, 1], grid, nlines=10)
    ax1.set_xlabel('$x (GM/c^2)$')
    ax1.set_ylabel('$z (GM/c^2)$')
    ax1.set_xlim([-50, 0])
    ax1.set_ylim([-50, 50])
    ax1.set_title('Log($\\rho$)', fontsize='large')
    if grid.get('rEH'):
        circle = plt.Circle((0, 0), grid['rEH'], color='k')
        ax1.add_artist(circle)
    ax1.set_aspect('equal')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(rhopolplot, cax=cax)

    plt.tight_layout()
    plt.savefig(os.path.join(grid['PLOTSDIR'], f'torus_basic_plot_{dumpval:04d}.png'))
    plt.close()

# main(): Reads param file, writes grid dict and calls analysis function
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '-p':
        fparams_name = sys.argv[2]
    else:
        sys.exit('No param file provided')

    # Reading the param file
    globalvars = {}
    globalvars_keys = ['PROB', 'NDIMS', 'DUMPSDIR', 'PLOTSDIR']
    with open(fparams_name, 'r') as fparams:
        lines = fparams.readlines()
        for line in lines:
            if line[0] == '#' or line.isspace(): 
                pass
            elif line.split()[0] in globalvars_keys: 
                globalvars[line.split()[0]] = line.split()[-1]

    # Creating the output directory if it doesn't exist
    if not os.path.exists(globalvars['PLOTSDIR']):
        os.makedirs(globalvars['PLOTSDIR'])

    # Calculating total dump files
    dstart = int(sorted(os.listdir(globalvars['DUMPSDIR']))[0][-7:-3])
    dend = int(sorted(list(filter(lambda dump: 'dump' in dump, os.listdir(globalvars['DUMPSDIR']))))[-1][-7:-3])
    dlist = range(dstart, dend + 1)
    Ndumps = dend - dstart + 1

    # Setting grid dict
    grid = {}
    gfile = h5py.File(os.path.join(globalvars['DUMPSDIR'], 'grid.h5'), 'r')
    dfile = h5py.File(os.path.join(globalvars['DUMPSDIR'], f'dump_0000{dstart:04d}.h5'), 'r')
    grid['n1'] = dfile['/header/n1'][()]
    grid['n2'] = dfile['/header/n2'][()]
    grid['n3'] = dfile['/header/n3'][()]
    grid['dx1'] = dfile['/header/geom/dx1'][()]
    grid['dx2'] = dfile['/header/geom/dx2'][()]
    grid['dx3'] = dfile['/header/geom/dx3'][()]
    grid['startx1'] = dfile['header/geom/startx1'][()]
    grid['startx2'] = dfile['header/geom/startx2'][()]
    grid['startx3'] = dfile['header/geom/startx3'][()]
    grid['metric'] = dfile['header/metric'][()].decode('UTF-8')
    grid['x1'] = gfile['X1'][()]
    grid['x2'] = gfile['X2'][()]
    grid['x3'] = gfile['X3'][()]
    grid['r'] = gfile['r'][()]
    grid['th'] = gfile['th'][()]
    grid['phi'] = gfile['phi'][()]
    grid['x'] = gfile['X'][()]
    grid['y'] = gfile['Y'][()]
    grid['z'] = gfile['Z'][()]
    grid['gcov'] = gfile['gcov'][()]
    grid['gcon'] = gfile['gcon'][()]
    grid['gdet'] = gfile['gdet'][()]
    grid['lapse'] = gfile['lapse'][()]
    grid['PLOTSDIR'] = globalvars['PLOTSDIR']

    if grid['metric'] in ['MKS', 'MMKS']:
        try:
            grid['a'] = dfile['header/geom/mks/a'][()]
        except KeyError:
            grid['a'] = dfile['header/geom/mmks/a'][()]
        try:
            grid['rEH'] = dfile['header/geom/mks/Reh'][()]
        except KeyError:
            pass
        try:
            grid['rEH'] = dfile['header/geom/mks/r_eh'][()]
        except KeyError:
            pass
        try:
            grid['rEH'] = dfile['header/geom/mmks/Reh'][()]
        except KeyError:
            pass
        try:
            grid['rEH'] = dfile['header/geom/mmks/r_eh'][()]
        except KeyError:
            pass
        try:
            grid['hslope'] = dfile['header/geom/mks/hslope'][()]
        except KeyError:
            grid['hslope'] = dfile['header/geom/mmks/hslope'][()]
    if grid['metric'] == 'MMKS':
        grid['mks_smooth'] = dfile['header/geom/mmks/mks_smooth'][()]
        grid['poly_alpha'] = dfile['header/geom/mmks/poly_alpha'][()]
        grid['poly_xt'] = dfile['header/geom/mmks/poly_xt'][()]
        grid['D'] = (np.pi * grid['poly_xt'] ** grid['poly_alpha']) / (2 * grid['poly_xt'] ** grid['poly_alpha'] + (2 / (1 + grid['poly_alpha'])))

    dfile.close()
    gfile.close()

    ncores = psutil.cpu_count(logical=True)
    pad = 0.25
    nthreads = int(ncores * pad)
    print(f"Number of threads: {nthreads:03d}")

    # Running analysis for torus2d
    if globalvars['PROB'] == 'torus' and globalvars['NDIMS'] == '2':
        run_parallel(analysis_torus2d, dlist, nthreads, [globalvars['DUMPSDIR'], grid])
