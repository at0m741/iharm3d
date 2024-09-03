/******************************************************************************
 *                                                                            *
 * FLUXES.C                                                                   *
 *                                                                            *
 * CALCULATES FLUID FLUXES                                                    *
 *                                                                            *
 ******************************************************************************/


/*
 * This file contains functions for calculating fluxes and time step limits 
 * in a magnetohydrodynamics (MHD) simulation. These functions are essential 
 * for advancing the fluid state in time, ensuring that numerical stability 
 * and physical accuracy are maintained across the simulation grid.
 *
 * Key functions in this file include:
 *
 * - `get_flux`: This function orchestrates the reconstruction of left and 
 *   right states at cell interfaces, computes fluxes in all three spatial 
 *   directions (X1, X2, X3), and determines the minimum time step based on 
 *   the fastest signal speed (ctop) across the grid.
 *
 * - `lr_to_flux`: Given the reconstructed left and right states at a cell 
 *   interface, this function calculates the fluxes based on the Riemann 
 *   solver approach. It also computes the characteristic speeds (`cmax` 
 *   and `cmin`) used to determine the stability of the time step.
 *
 * - `ndt_min`: This function computes the minimum time step (`ndt_min`) 
 *   across all zones in the simulation grid, ensuring that the Courant 
 *   condition is satisfied to maintain stability.
 *
 * - `flux_ct`: This function applies the constrained transport (CT) 
 *   method to maintain the divergence-free condition of the magnetic 
 *   field (`∇·B = 0`) throughout the simulation. It calculates the 
 *   electromotive forces (EMFs) from the fluxes and updates the magnetic 
 *   field accordingly.
 *
 * The file also includes extensive use of OpenMP to parallelize computations, 
 * enhancing performance on large grids. The functions make use of static 
 * variables and conditional initialization to minimize memory allocation 
 * overhead during repeated calls.
 *
 * Debugging features are included to provide insights into the calculation 
 * of the timestep and fluxes, aiding in the diagnosis of potential numerical 
 * issues in the simulation.
 */

#include "decs.h"

void lr_to_flux(struct GridGeom *G, struct FluidState *Sl, struct FluidState *Sr, int dir, int loc, GridPrim *flux, GridVector *ctop);
double ndt_min(GridVector *ctop);

double ndt_min(GridVector *ctop)
{
	timer_start(TIMER_CMAX);
	double ndt_min = 1e20;

#if DEBUG
	int min_x1, min_x2, min_x3;
#endif

	#pragma omp parallel for collapse(3) reduction(min : ndt_min)
	ZLOOP
	{
		double ndt_zone = 0;
		for (int mu = 1; mu < NDIM; mu++)
		{
			ndt_zone += 1 / (cour * dx[mu] / (*ctop)[mu][k][j][i]);
		}
		ndt_zone = 1 / ndt_zone;

		if (ndt_zone < ndt_min)
		{
			ndt_min = ndt_zone;
#if DEBUG
			min_x1 = i;
			min_x2 = j;
			min_x3 = k;
#endif
		}
	}

#if DEBUG
	fprintf(stderr, "Timestep set by %d %d %d\n", min_x1, min_x2, min_x3);
#endif

	timer_stop(TIMER_CMAX);
	return ndt_min;
}

double get_flux(struct GridGeom *G, struct FluidState *S, struct FluidFlux *F)
{
	static struct FluidState *Sl, *Sr;
	static GridVector        *ctop;
	double                    cmax[NDIM], ndts[NDIM];
	memset(cmax, 0, NDIM * sizeof(double));
	memset(ndts, 0, NDIM * sizeof(double));

	static int firstc = 1;

	if (firstc)
	{
		Sl = calloc(1, sizeof(struct FluidState));
		Sr = calloc(1, sizeof(struct FluidState));
		ctop = calloc(1, sizeof(GridVector));

		firstc = 0;
	}

	// FLAG("First get_flux");

	// reconstruct X1
	reconstruct(S, Sl->P, Sr->P, 1);

	// FLAG("After Reconstruct");

	// lr_to_flux X1
	lr_to_flux(G, Sl, Sr, 1, FACE1, &(F->X1), ctop);

	// reconstruct X2
	reconstruct(S, Sl->P, Sr->P, 2);

	// lr_to_flux X2
	lr_to_flux(G, Sl, Sr, 2, FACE2, &(F->X2), ctop);

	// reconstruct X3
	reconstruct(S, Sl->P, Sr->P, 3);

	// lr_to_flux X3
	lr_to_flux(G, Sl, Sr, 3, FACE3, &(F->X3), ctop);

	// TODO don't call for static timestep
	return ndt_min(ctop);
}

// Note that the sense of L/R flips from zone to interface during function call
void lr_to_flux(struct GridGeom *G, struct FluidState *Sr, struct FluidState *Sl, int dir, int loc, GridPrim *flux, GridVector *ctop)
{
	timer_start(TIMER_LR_TO_F);

	static GridPrim   *fluxL, *fluxR;
	static GridDouble *cmaxL, *cmaxR, *cminL, *cminR, *cmax, *cmin;

	static int firstc = 1;

	if (firstc)
	{
		fluxL = calloc(1, sizeof(GridPrim));
		fluxR = calloc(1, sizeof(GridPrim));
		cmaxL = calloc(1, sizeof(GridDouble));
		cmaxR = calloc(1, sizeof(GridDouble));
		cminL = calloc(1, sizeof(GridDouble));
		cminR = calloc(1, sizeof(GridDouble));
		cmax = calloc(1, sizeof(GridDouble));
		cmin = calloc(1, sizeof(GridDouble));

		firstc = 0;
	}

	// FLAG("First LR");

	// Properly offset left face
	// These are un-macro'd to bundle OpenMP thread tasks rather than memory accesses
	PLOOP
	{
		if (dir == 1)
		{
			#pragma omp parallel for collapse(2)
			ZSLOOP_REVERSE(-1, N3, -1, N2, -1, N1)
			Sl->P[ip][k][j][i] = Sl->P[ip][k][j][i - 1];
		}
		else if (dir == 2)
		{
			#pragma omp parallel for collapse(2)
			for (int k = (N3) + NG; k >= (-1) + NG; k--)
			{
				for (int i = (N1) + NG; i >= (-1) + NG; i--)
				{
					for (int j = (N2) + NG; j >= (-1) + NG; j--)
						Sl->P[ip][k][j][i] = Sl->P[ip][k][j - 1][i];
				}
			}
		}
		else if (dir == 3)
		{
			#pragma omp parallel for collapse(2)
			for (int j = (N2) + NG; j >= (-1) + NG; j--)
			{
				for (int i = (N1) + NG; i >= (-1) + NG; i--)
				{
					for (int k = (N3) + NG; k >= (-1) + NG; k--)
						Sl->P[ip][k][j][i] = Sl->P[ip][k - 1][j][i];
				}
			}
		}
	}

	// FLAG("Left Face Offset");

	timer_start(TIMER_LR_STATE);

	get_state_vec(G, Sl, loc, -1, N3, -1, N2, -1, N1);
	get_state_vec(G, Sr, loc, -1, N3, -1, N2, -1, N1);

	timer_stop(TIMER_LR_STATE);

	// FLAG("Left Face Offset");

	timer_start(TIMER_LR_PTOF);

	prim_to_flux_vec(G, Sl, 0, loc, -1, N3, -1, N2, -1, N1, Sl->U);
	prim_to_flux_vec(G, Sl, dir, loc, -1, N3, -1, N2, -1, N1, *fluxL);

	prim_to_flux_vec(G, Sr, 0, loc, -1, N3, -1, N2, -1, N1, Sr->U);
	prim_to_flux_vec(G, Sr, dir, loc, -1, N3, -1, N2, -1, N1, *fluxR);

	timer_stop(TIMER_LR_PTOF);

	// FLAG("State, flux");

	timer_start(TIMER_LR_VCHAR);
	// TODO vectorizing these loops fails for some reason
	#pragma omp parallel
	{
		#pragma omp for collapse(2) nowait
		ZSLOOP(-1, N3, -1, N2, -1, N1)
		{
			mhd_vchar(G, Sl, i, j, k, loc, dir, *cmaxL, *cminL);
		}
		#pragma omp for collapse(2)
		ZSLOOP(-1, N3, -1, N2, -1, N1)
		{
			mhd_vchar(G, Sr, i, j, k, loc, dir, *cmaxR, *cminR);
		}
	}
	timer_stop(TIMER_LR_VCHAR);

	timer_start(TIMER_LR_CMAX);
	#pragma omp parallel for collapse(2)
	ZSLOOP(-1, N3, -1, N2, -1, N1)
	{
		(*cmax)[k][j][i] = fabs(MY_MAX(MY_MAX(0., (*cmaxL)[k][j][i]), (*cmaxR)[k][j][i]));
		(*cmin)[k][j][i] = fabs(MY_MAX(MY_MAX(0., -(*cminL)[k][j][i]), -(*cminR)[k][j][i]));
		(*ctop)[dir][k][j][i] = MY_MAX((*cmax)[k][j][i], (*cmin)[k][j][i]);
		if (isnan(1. / (*ctop)[dir][k][j][i]))
		{
			printf("ctop is 0 or NaN at zone: %i %i %i (%i) ", i, j, k, dir);
#if METRIC == MKS
			double X[NDIM];
			double r, th;
			coord(i, j, k, CENT, X);
			bl_coord(X, &r, &th);
			printf("(r,th,phi = %f %f %f)\n", r, th, X[3]);
#endif
			printf("\n");
			exit(-1);
		}
	}
	timer_stop(TIMER_LR_CMAX);

	timer_start(TIMER_LR_FLUX);
	#pragma omp parallel for simd collapse(3)
	PLOOP
	{
		ZSLOOP(-1, N3, -1, N2, -1, N1)
		{
			(*flux)[ip][k][j][i] = 0.5 * ((*fluxL)[ip][k][j][i] + 
			(*fluxR)[ip][k][j][i] - (*ctop)[dir][k][j][i] * 
			(Sr->U[ip][k][j][i] - Sl->U[ip][k][j][i]));
		}
	}
	timer_stop(TIMER_LR_FLUX);
	timer_stop(TIMER_LR_TO_F);
}


/*
 * The `flux_ct` function is responsible for enforcing the divergence-free 
 * condition of the magnetic field in a magnetohydrodynamics (MHD) simulation. 
 * This condition, known mathematically as `∇·B = 0`, is crucial for ensuring 
 * that the simulated magnetic field remains physically realistic, as any 
 * non-zero divergence would imply the presence of magnetic monopoles, which 
 * do not exist in nature.
 *
 * The function operates in several key steps:
 *
 * 1. **Electromotive Force (EMF) Calculation:**
 *    - The function begins by calculating the electromotive forces (EMFs) 
 *      at the cell edges using the fluxes provided in the `FluidFlux` structure.
 *      These EMFs are computed using central differences of the magnetic flux 
 *      components, following a method proposed by Tóth to maintain the 
 *      divergence-free condition.
 *    - Specifically, the EMF components `X1`, `X2`, and `X3` are calculated 
 *      as weighted sums of the magnetic fluxes from adjacent cells. For example, 
 *      `emf->X3[k][j][i]` is computed using the fluxes `F->X1[B2]`, `F->X2[B1]` 
 *      at neighboring cells.
 *
 * 2. **Conversion of EMFs to Fluxes:**
 *    - Once the EMFs are calculated, they are used to update the magnetic fluxes 
 *      at the cell faces. This step involves rewriting the EMFs as fluxes using 
 *      a specific averaging technique that ensures the divergence-free condition 
 *      is maintained. For instance, the flux `F->X1[B2]` is updated using the 
 *      average of the EMFs `emf->X3` at adjacent grid points.
 *
 * 3. **Parallelization:**
 *    - The function is parallelized using OpenMP to efficiently handle large 
 *      simulation grids. The `#pragma omp parallel` directive is used to 
 *      distribute the workload across multiple threads, with loops collapsed 
 *      where necessary to optimize performance. The `simd` directive further 
 *      accelerates the computation by enabling vectorization of the loop 
 *      operations, allowing multiple data points to be processed simultaneously.
 *
 * 4. **Ensuring Numerical Stability:**
 *    - The `flux_ct` function plays a crucial role in the overall stability 
 *      of the MHD simulation. By accurately computing and applying the EMFs, 
 *      it ensures that the magnetic field evolution is consistent with the 
 *      physical laws governing magnetized fluids. This, in turn, prevents 
 *      the buildup of numerical errors that could lead to instability or 
 *      non-physical results.
 *
 * The `flux_ct` function is typically called after the fluxes have been computed 
 * for all directions (X1, X2, X3) by the `get_flux` function. By enforcing the 
 * divergence-free condition, it ensures that the magnetic field remains well-behaved 
 * throughout the simulation, which is particularly important in high-resolution 
 * or long-duration simulations where small errors can accumulate over time.
 *
 * Overall, `flux_ct` is an essential component of the MHD solver, enabling the 
 * accurate simulation of astrophysical plasmas, accretion disks, and other 
 * magnetized fluid systems.
 */

void flux_ct(struct FluidFlux *F)
{
	timer_start(TIMER_FLUX_CT);

	static struct FluidEMF *emf;

	static int firstc = 1;
	if (firstc)
	{
		emf = calloc(1, sizeof(struct FluidEMF));
		firstc = 0;
	}

	#pragma omp parallel
	{
		// This and the following are /not/ just ZLOOPs
		#pragma omp for simd collapse(2)
		ZSLOOP(0, N3, 0, N2, 0, N1)
		{
			emf->X3[k][j][i] = 0.25 * (F->X1[B2][k][j][i] +
				F->X1[B2][k][j - 1][i] - 
				F->X2[B1][k][j][i] - 
				F->X2[B1][k][j][i - 1]);
			emf->X2[k][j][i] = -0.25 * (F->X1[B3][k][j][i] + 
				F->X1[B3][k - 1][j][i] - F->X3[B1][k][j][i] - 
				F->X3[B1][k][j][i - 1]);
			emf->X1[k][j][i] = 0.25 * (F->X2[B3][k][j][i] + 
				F->X2[B3][k - 1][j][i] - F->X3[B2][k][j][i] - 
				F->X3[B2][k][j - 1][i]);
		}

		// Rewrite EMFs as fluxes, after Toth
		#pragma omp for simd collapse(2) nowait
		ZSLOOP(0, N3 - 1, 0, N2 - 1, 0, N1)
		{
			F->X1[B1][k][j][i] = 0.;
			F->X1[B2][k][j][i] = 0.5 * (emf->X3[k][j][i] + emf->X3[k][j + 1][i]);
			F->X1[B3][k][j][i] = -0.5 * (emf->X2[k][j][i] + emf->X2[k + 1][j][i]);
		}
		#pragma omp for simd collapse(2) nowait
		ZSLOOP(0, N3 - 1, 0, N2, 0, N1 - 1)
		{
			F->X2[B1][k][j][i] = -0.5 * (emf->X3[k][j][i] + emf->X3[k][j][i + 1]);
			F->X2[B2][k][j][i] = 0.;
			F->X2[B3][k][j][i] = 0.5 * (emf->X1[k][j][i] + emf->X1[k + 1][j][i]);
		}
		#pragma omp for simd collapse(2)
		ZSLOOP(0, N3, 0, N2 - 1, 0, N1 - 1)
		{
			F->X3[B1][k][j][i] = 0.5 * (emf->X2[k][j][i] + emf->X2[k][j][i + 1]);
			F->X3[B2][k][j][i] = -0.5 * (emf->X1[k][j][i] + emf->X1[k][j + 1][i]);
			F->X3[B3][k][j][i] = 0.;
		}
	} // omp parallel
	timer_stop(TIMER_FLUX_CT);
}
