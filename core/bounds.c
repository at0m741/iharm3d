/******************************************************************************
 *                                                                            *
 * BOUNDS.C                                                                   *
 *                                                                            *
 * PHYSICAL BOUNDARY CONDITIONS                                               *
 *                                                                            *
 ******************************************************************************/

#include "decs.h"


/*
* Boundary conditions are applied in the following order:
* 1. X1L_BOUND
* 2. X1R_BOUND
* 3. X2L_BOUND
* 4. X2R_BOUND
* 5. X3L_BOUND
* 6. X3R_BOUND
*/


#if N2 > 1 && N2 < NG
	#error "N2 must be >= NG"
#elif N3 > 1 && N3 < NG
	#error "N3 must be >= NG"
#endif

#if X1L_BOUND != PERIODIC && X1L_BOUND != OUTFLOW
	#error "Unsupported X1L_BOUND"
#endif

#if X1R_BOUND != PERIODIC && X1R_BOUND != OUTFLOW && X1R_BOUND != USER
	#error "Unsupported X1R_BOUND"
#endif

#if X2L_BOUND != PERIODIC && X2L_BOUND != OUTFLOW && X2L_BOUND != POLAR
	#error "Unsupported X2L_BOUND"
#endif

#if X2R_BOUND != PERIODIC && X2R_BOUND != OUTFLOW && X2R_BOUND != POLAR
	#error "Unsupported X2R_BOUND"
#endif

#if X3L_BOUND != PERIODIC && X3L_BOUND != OUTFLOW
	#error "Unsupported X3L_BOUND"
#endif

#if X3R_BOUND != PERIODIC && X3R_BOUND != OUTFLOW
	#error "Unsupported X3R_BOUND"
#endif

void inflow_check(struct GridGeom *G, struct FluidState *S, int i, int j, int k, int type);

/*
 * Apply the outflow boundary condition to the fluid state variables.
 * 
 * This function ensures that the physical quantities at the boundary
 * cells are consistent with an outflow condition. Specifically, it copies
 * the values from the inner cell (at index `iz`) to the boundary cell
 * (at index `i`) for all fluid state variables. Additionally, it rescales
 * the magnetic field components (`B1`, `B2`, `B3`) according to the ratio
 * of the metric determinant (`gdet`) between the inner and boundary cells.
 * 
 * Parameters:
 * - G: Grid geometry structure containing metric information.
 * - S: Fluid state structure containing the state variables.
 * - i, j, k: Indices of the boundary cell to which the condition is applied.
 * - iz: Index of the inner cell from which values are copied.
 */

void apply_outflow_boundary_condition(struct GridGeom *G, struct FluidState *S, int i, int j, int k, int iz) {
    PLOOP S->P[ip][k][j][i] = S->P[ip][k][j][iz];
    pflag[k][j][i] = pflag[k][j][iz];

    double rescale = G->gdet[CENT][j][iz] / G->gdet[CENT][j][i];
    S->P[B1][k][j][i] *= rescale;
    S->P[B2][k][j][i] *= rescale;
    S->P[B3][k][j][i] *= rescale;
}


/*
 * Handle the X1 boundary conditions for the fluid state variables.
 * 
 * This function applies the appropriate boundary conditions on the X1 direction
 * (typically corresponding to the radial or x-direction in the grid). Depending 
 * on the specified boundary condition type (`OUTFLOW`, `USER`, etc.), it applies 
 * the necessary adjustments to the fluid variables in the ghost zones at the 
 * beginning and end of the grid in the X1 direction.
 * 
 * If `OUTFLOW` is specified, the function calls `apply_outflow_boundary_condition` 
 * to copy the values from the adjacent inner cells and rescale them according to 
 * the metric. For the `USER` boundary condition, the function calls 
 * `bound_gas_prob_x1r` to handle custom user-defined boundaries.
 * 
 * Parameters:
 * - G: Grid geometry structure containing metric information.
 * - S: Fluid state structure containing the state variables.
 * - i, j, k: Indices used for looping through the grid cells.
 * - start: Starting index for applying the boundary conditions.
 * - stop: Ending index for applying the boundary conditions.
 */

void handle_x1_boundary(struct GridGeom *G, struct FluidState *S, int i, int j, int k, int start, int stop) {
    if (start == 0) 
	{
		#if !INTEL_WORKAROUND
			#pragma omp parallel for collapse(2)
		#endif
        KLOOP 
            JLOOP
                ISLOOP(-NG, -1) 
				{
                    #if X1L_BOUND == OUTFLOW
                    apply_outflow_boundary_condition(G, S, i, j, k, NG);
                    #endif
                }
        }

#if METRIC == MKS

        if (X1L_INFLOW == 0) 
		{
			#if !INTEL_WORKAROUND
				#pragma omp parallel for collapse(2)
			#endif
            KLOOP
                JLOOP 
                    ISLOOP(-NG, -1) {
                        inflow_check(G, S, i, j, k, 0);
        }
#endif
    }

    if (stop == N1TOT) 
	{
		#if !INTEL_WORKAROUND
			#pragma omp parallel for collapse(2)
		#endif
        KLOOP 
		{
            JLOOP 
			{
                ISLOOP(N1, N1 - 1 + NG)
				{
                    #if X1R_BOUND == OUTFLOW
						apply_outflow_boundary_condition(G, S, i, j, k, N1 - 1 + NG);
                    #elif X1R_BOUND == USER
						bound_gas_prob_x1r(i, j, k, S->P, G);
                    #endif
                }
            }
        }

#if METRIC == MKS

        if (X1R_INFLOW == 0) 
		{
			#if !INTEL_WORKAROUND
				#pragma omp parallel for collapse(2)
			#endif
            KLOOP 
                JLOOP
                    ISLOOP(N1, N1 - 1 + NG) 
                        inflow_check(G, S, i, j, k, 1);
        }
#endif
    }
}

void handle_x2_boundary(struct GridGeom *G, struct FluidState *S, int i, int j, int k, int start, int stop) {
    if (start == 0) 
	{
		#if !INTEL_WORKAROUND
			#pragma omp parallel for collapse(2)
		#endif
        KLOOP
		{
            ILOOPALL
			{
                JSLOOP(-NG, -1)
				{
                    #if X2L_BOUND == OUTFLOW
						apply_outflow_boundary_condition(G, S, i, j, k, NG);
                    #elif X2L_BOUND == POLAR
						int jrefl = NG + (NG - j) - 1;
				        
						PLOOP
							S->P[ip][k][j][i] = S->P[ip][k][jrefl][i];
					   
						pflag[k][j][i] = pflag[k][jrefl][i];
						S->P[U2][k][j][i] *= -1.;
					    S->P[B2][k][j][i] *= -1.;
                    #endif
                }
            }
        }
    }

    if (stop == N2TOT) 
	{
		#if !INTEL_WORKAROUND
			#pragma omp parallel for collapse(2)
		#endif
        KLOOP
		{
            ILOOPALL 
			{	
                JSLOOP(N2, N2 - 1 + NG) {
                    #if X2R_BOUND == OUTFLOW
                    apply_outflow_boundary_condition(G, S, i, j, k, N2 - 1 + NG);
                    #elif X2R_BOUND == POLAR
                    int jrefl = (N2 + NG) + (N2 + NG - j) - 1;
                    PLOOP S->P[ip][k][j][i] = S->P[ip][k][jrefl][i];
                    pflag[k][j][i] = pflag[k][jrefl][i];
                    S->P[U2][k][j][i] *= -1.;
                    S->P[B2][k][j][i] *= -1.;
                    #endif
                }
			}
		}
    }
}

void handle_x3_boundary(struct GridGeom *G, struct FluidState *S, int i, int j, int k, int start, int stop) {
    if (start == 0) 
	{
		#if !INTEL_WORKAROUND
			#pragma omp parallel for collapse(2)
		#endif
        JLOOPALL
        {
			ILOOPALL 
			{
                KSLOOP(-NG, -1) 
				{
                    #if X3L_BOUND == OUTFLOW
						apply_outflow_boundary_condition(G, S, i, j, k, NG);
                    #endif
                }
			}
		}
    }

    if (stop == N3TOT) 
	{
		#if !INTEL_WORKAROUND
			#pragma omp parallel for collapse(2)
		#endif
        JLOOPALL
		{
            ILOOPALL
			{	
				KSLOOP(N3, N3 - 1 + NG) 
				{
                    #if X3R_BOUND == OUTFLOW
						apply_outflow_boundary_condition(G, S, i, j, k, N3 - 1 + NG);
                    #endif
                }
			}
		}
    }
}

void set_bounds(struct GridGeom *G, struct FluidState *S) {
    timer_start(TIMER_BOUND);

    handle_x1_boundary(G, S, -1, -1, -1, global_start[0], global_stop[0]);

    timer_start(TIMER_BOUND_COMMS);
    sync_mpi_bound_X1(S);
    timer_stop(TIMER_BOUND_COMMS);

    handle_x2_boundary(G, S, -1, -1, -1, global_start[1], global_stop[1]);

    timer_start(TIMER_BOUND_COMMS);
    sync_mpi_bound_X2(S);
    timer_stop(TIMER_BOUND_COMMS);

    handle_x3_boundary(G, S, -1, -1, -1, global_start[2], global_stop[2]);

    timer_start(TIMER_BOUND_COMMS);
    sync_mpi_bound_X3(S);
    timer_stop(TIMER_BOUND_COMMS);

    timer_stop(TIMER_BOUND);
}

#if METRIC == MKS
void inflow_check(struct GridGeom *G, struct FluidState *S, int i, int j, int k, int type) {
    double alpha, beta1, vsq;
    double gamma = mhd_gamma_calc(G, S, i, j, k, CENT);
    
	ucon_calc(G, S, i, j, k, CENT);

    if (((S->ucon[1][k][j][i] > 0.) && (type == 0)) ||
        ((S->ucon[1][k][j][i] < 0.) && (type == 1))) 
	{
        S->P[U1][k][j][i] /= gamma;
        S->P[U2][k][j][i] /= gamma;
        S->P[U3][k][j][i] /= gamma;
        alpha = G->lapse[CENT][j][i];
        beta1 = G->gcon[CENT][0][1][j][i] * alpha * alpha;

        S->P[U1][k][j][i] = beta1 / alpha;

        vsq = 0.;

        for (int mu = 1; mu < NDIM; mu++) 
            for (int nu = 1; nu < NDIM; nu++) 
                vsq += G->gcov[CENT][mu][nu][j][i] * \
					   S->P[U1 + mu - 1][k][j][i] * \
					   S->P[U1 + nu - 1][k][j][i];

        if (fabs(vsq) < 1.e-13)
            vsq = 1.e-13;

        if (vsq >= 1.) 
            vsq = 1. - 1. / (GAMMAMAX * GAMMAMAX);

        gamma = 1. / sqrt(1. - vsq);
        S->P[U1][k][j][i] *= gamma;
        S->P[U2][k][j][i] *= gamma;
        S->P[U3][k][j][i] *= gamma;
    }
}

void fix_flux(struct FluidFlux *F) {
    if (global_start[0] == 0 && X1L_INFLOW == 0) 
	{
		#if !INTEL_WORKAROUND
			#pragma omp parallel for collapse(2)
		#endif
        KLOOPALL 
            JLOOPALL 
                F->X1[RHO][k][j][0 + NG] = MY_MIN(F->X1[RHO][k][j][0 + NG], 0.);
    }

    if (global_stop[0] == N1TOT && X1R_INFLOW == 0) 
	{
		#if !INTEL_WORKAROUND
			#pragma omp parallel for collapse(2)
		#endif
        KLOOPALL 
            JLOOPALL 
                F->X1[RHO][k][j][N1 + NG] = MY_MAX(F->X1[RHO][k][j][N1 + NG], 0.);
    }

    if (global_start[1] == 0) 
	{
		#if !INTEL_WORKAROUND
			#pragma omp parallel for collapse(2)
		#endif
        KLOOPALL
            ILOOPALL 
			{
                F->X1[B2][k][-1 + NG][i] = -F->X1[B2][k][0 + NG][i];
                F->X3[B2][k][-1 + NG][i] = -F->X3[B2][k][0 + NG][i];
                PLOOP F->X2[ip][k][0 + NG][i] = 0.;
            }
    }

    if (global_stop[1] == N2TOT) 
	{
		#if !INTEL_WORKAROUND
			#pragma omp parallel for collapse(2)
		#endif
        KLOOPALL
            ILOOPALL 
			{
                F->X1[B2][k][N2 + NG][i] = -F->X1[B2][k][N2 - 1 + NG][i];
                F->X3[B2][k][N2 + NG][i] = -F->X3[B2][k][N2 - 1 + NG][i];
                PLOOP F->X2[ip][k][N2 + NG][i] = 0.;
            }
    }
}
#endif // METRIC == MKS
