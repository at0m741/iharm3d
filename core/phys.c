/******************************************************************************
 *                                                                            *
 * PHYS.C                                                                     *
 *                                                                            *
 * COMPUTES PHYSICAL QUANTITIES: STRESS-ENERGY TENSOR, U(P), FLUXES,          *
 * 4-VECTORS, LORENTZ FACTOR, MAGNETOSONIC VELOCITY, SOURCE TERMS             *
 *                                                                            *
 ******************************************************************************/

#include "decs.h"


/*
* This file implements core functions for solving relativistic magnetohydrodynamics (MHD) 
* in a curved spacetime.
*
* The `mhd_calc` function computes the MHD stress-energy tensor, where the first index is 
* up and the second index is down. It incorporates the magnetic field vector `b`, with a 
* factor of sqrt(4 pi) absorbed into its definition. This tensor is crucial for conserving 
* energy and momentum in MHD simulations.
*
* The `prim_to_flux` function converts primitive variables (like density, pressure, and 
* magnetic field components) into fluxes in a specified direction. These fluxes are essential 
* for solving the conservation equations. The function also computes the dual of the Maxwell 
* tensor, which governs the magnetic field's evolution in the fluid.
*
* The `prim_to_flux_vec` function extends `prim_to_flux` to handle a range of grid points, 
* optimized for parallel execution with OpenMP. This function is designed for large-scale 
* simulations, where efficiency is key.
*
* The `bcon_calc` function calculates the magnetic field four-vector components, which are 
* necessary for evolving the magnetic field in the MHD framework.
*
* The `mhd_gamma_calc` function calculates the Lorentz gamma factor, representing the 
* relativistic speed of the fluid relative to a normal observer. This is derived from the 
* fluid's velocity components and the spacetime metric.
*
* The `ucon_calc` function computes the contravariant four-velocity components of the fluid, 
* which are critical for solving the MHD equations of motion.
*
* The `get_state` and `get_state_vec` functions compute both the contravariant and covariant 
* four-velocity and magnetic field components, providing all necessary state information 
* for the fluid at each grid point. The vectorized version is optimized for performance.
*
* The `mhd_vchar` function calculates the magnetosonic speeds, which are the characteristic 
* velocities of waves propagating through the magnetized fluid. This function is essential 
* for ensuring that the numerical scheme correctly captures the dynamics of fast-moving 
* waves in the fluid.
*
* The `get_fluid_source` function computes the source terms for the fluid equations of motion, 
* accounting for the MHD stress tensor and gravitational effects. An optional wind term can be 
* included to model particle injection in certain regions, such as the funnel in black hole 
* simulations.
*
* The `bsq_calc` function returns the square of the magnetic field magnitude, which is 
* proportional to twice the magnetic pressure. This value is often needed for diagnostics 
* and for computing other quantities in the simulation.
*/

/*
 * mhd_calc():
 * no instabilities, no need to check for negative pressure
 * no need to check for negative rho
*/
inline void mhd_calc(struct FluidState *S, int i, int j, int k, int dir, double *mhd)
{
    double u = S->P[UU][k][j][i];
    double rho = S->P[RHO][k][j][i];
    double pres = (gam - 1.0) * u;     
    double w = pres + rho + u;        
    double bsq = bsq_calc(S, i, j, k);
    double eta = w + bsq;            
    double ptot = pres + 0.5 * bsq; 

    #pragma omp simd
    for (int mu = 0; mu < NDIM; ++mu)
    {
        double ucon_dir = S->ucon[dir][k][j][i];
        double ucov_mu = S->ucov[mu][k][j][i];
        double bcon_dir = S->bcon[dir][k][j][i];
        double bcov_mu = S->bcov[mu][k][j][i];

        mhd[mu] = eta * ucon_dir * ucov_mu 
                  + ptot * delta(dir, mu) 
                  - bcon_dir * bcov_mu;
    }
}

// TODO OLD only used in fixup.c and even then hacked to hell

inline void prim_to_flux(struct GridGeom *G, struct FluidState *S, int i, int j, int k, int dir, int loc, GridPrim flux)
{
    double rho = S->P[RHO][k][j][i];
    double ucon_dir = S->ucon[dir][k][j][i];

    double flux_rho = rho * ucon_dir;
    flux[RHO][k][j][i] = flux_rho;

    double mhd[NDIM];
    mhd_calc(S, i, j, k, dir, mhd);

    flux[UU][k][j][i] = mhd[0] + flux_rho;
    flux[U1][k][j][i] = mhd[1];
    flux[U2][k][j][i] = mhd[2];
    flux[U3][k][j][i] = mhd[3];

    double bcon_dir = S->bcon[dir][k][j][i];
    flux[B1][k][j][i] = S->bcon[1][k][j][i] * ucon_dir - bcon_dir * S->ucon[1][k][j][i];
    flux[B2][k][j][i] = S->bcon[2][k][j][i] * ucon_dir - bcon_dir * S->ucon[2][k][j][i];
    flux[B3][k][j][i] = S->bcon[3][k][j][i] * ucon_dir - bcon_dir * S->ucon[3][k][j][i];

#if ELECTRONS
    for (int idx = KEL0; idx < NVAR; idx++)
        flux[idx][k][j][i] = flux_rho * S->P[idx][k][j][i];
    flux[KTOT][k][j][i] = flux_rho * S->P[KTOT][k][j][i];
#endif

    double gdet = G->gdet[loc][j][i];
    PLOOP flux[ip][k][j][i] *= gdet;
}


// Calculate fluxes in direction dir, over given range.
// Note backward indices convention, consistent with ZSLOOP's arguments
void prim_to_flux_vec(struct GridGeom *G, struct FluidState *S, int dir, int loc, int kstart, int kstop, int jstart, int jstop, int istart, int istop, GridPrim flux)
{
	// TODO reintroduce simd pragma to see where it messes things up
	#pragma omp parallel
	{
		#pragma omp for collapse(3) nowait
		ZSLOOP(kstart, kstop, jstart, jstop, istart, istop)
		{
			double mhd[NDIM];

			flux[RHO][k][j][i] = S->P[RHO][k][j][i] * S->ucon[dir][k][j][i] * G->gdet[loc][j][i];

			mhd_calc(S, i, j, k, dir, mhd);

			// MHD stress-energy tensor w/ first index up, second index down
			flux[UU][k][j][i] = mhd[0] * G->gdet[loc][j][i] + flux[RHO][k][j][i];
			flux[U1][k][j][i] = mhd[1] * G->gdet[loc][j][i];
			flux[U2][k][j][i] = mhd[2] * G->gdet[loc][j][i];
			flux[U3][k][j][i] = mhd[3] * G->gdet[loc][j][i];
		}

		#pragma omp for collapse(3) nowait
		ZSLOOP(kstart, kstop, jstart, jstop, istart, istop)
		{
			// Dual of Maxwell tensor
			flux[B1][k][j][i] = (S->bcon[1][k][j][i] * S->ucon[dir][k][j][i] - 
								S->bcon[dir][k][j][i] * S->ucon[1][k][j][i]) * 
								G->gdet[loc][j][i];
			flux[B2][k][j][i] = (S->bcon[2][k][j][i] * S->ucon[dir][k][j][i] - 
								S->bcon[dir][k][j][i] * S->ucon[2][k][j][i]) * 
								G->gdet[loc][j][i];
			flux[B3][k][j][i] = (S->bcon[3][k][j][i] * S->ucon[dir][k][j][i] - 
								S->bcon[dir][k][j][i] * S->ucon[3][k][j][i]) * 
								G->gdet[loc][j][i];
		}

#if ELECTRONS
		#pragma omp for collapse(3)
		ZSLOOP(kstart, kstop, jstart, jstop, istart, istop)
		{
			for (int idx = KEL0; idx < NVAR; idx++)
				flux[idx][k][j][i] = flux[RHO][k][j][i] * S->P[idx][k][j][i];
			flux[KTOT][k][j][i] = flux[RHO][k][j][i] * S->P[KTOT][k][j][i];
		}
#endif
	}
}

// calculate magnetic field four-vector

inline void bcon_calc(struct FluidState *S, int i, int j, int k)
{
    double bcon0 = S->P[B1][k][j][i] * S->ucov[1][k][j][i] + 
                   S->P[B2][k][j][i] * S->ucov[2][k][j][i] + 
                   S->P[B3][k][j][i] * S->ucov[3][k][j][i];

    S->bcon[0][k][j][i] = bcon0;

    double ucon0_inv = 1.0 / S->ucon[0][k][j][i];

    for (int mu = 1; mu < 4; mu++)
        S->bcon[mu][k][j][i] = (S->P[B1 - 1 + mu][k][j][i] + bcon0 * 
								S->ucon[mu][k][j][i]) * ucon0_inv;
}


// Find gamma-factor wrt normal observer
inline double mhd_gamma_calc(struct GridGeom *G, struct FluidState *S, int i, int j, int k, int loc)
{
    double qsq = G->gcov[loc][1][1][j][i] * S->P[U1][k][j][i] * S->P[U1][k][j][i] +
                 G->gcov[loc][2][2][j][i] * S->P[U2][k][j][i] * S->P[U2][k][j][i] +
                 G->gcov[loc][3][3][j][i] * S->P[U3][k][j][i] * S->P[U3][k][j][i] +
                 2.0 * (
                     G->gcov[loc][1][2][j][i] * S->P[U1][k][j][i] * S->P[U2][k][j][i] +
                     G->gcov[loc][1][3][j][i] * S->P[U1][k][j][i] * S->P[U3][k][j][i] +
                     G->gcov[loc][2][3][j][i] * S->P[U2][k][j][i] * S->P[U3][k][j][i]
                 );

#if DEBUG
    if (qsq < 0.0)
    {
        if (fabs(qsq) > 1.0e-10)
        {
            fprintf(stderr, "gamma_calc(): failed: [%d %d %d] qsq = %28.18e \n", i, j, k, qsq);
            fprintf(stderr, "v[1-3] = %28.18e %28.18e %28.18e  \n", S->P[U1][k][j][i], S->P[U2][k][j][i], S->P[U3][k][j][i]);
            return 1.0;
        }
        else
            qsq = 1.0e-10;
    }
#endif
    return sqrt(1.0 + qsq);
}


// Find contravariant four-velocity
inline void ucon_calc(struct GridGeom *G, struct FluidState *S, int i, int j, int k, int loc)
{
    double gamma = mhd_gamma_calc(G, S, i, j, k, loc);
    double alpha = G->lapse[loc][j][i];

    S->ucon[0][k][j][i] = gamma / alpha;

    for (int mu = 1; mu < NDIM; mu++)
        S->ucon[mu][k][j][i] = S->P[U1 + mu - 1][k][j][i] - gamma * alpha * G->gcon[loc][0][mu][j][i];
}


// Calculate ucon, ucov, bcon, bcov from primitive variables
// TODO OLD individual calculation -- use vector
inline void get_state(struct GridGeom *G, struct FluidState *S, int i, int j, int k, int loc)
{
	ucon_calc(G, S, i, j, k, loc);
	GridIndices idx = {i, j, k};
	lower_grid(S->ucon, S->ucov, G, idx, loc);
	bcon_calc(S, i, j, k);
	lower_grid(S->bcon, S->bcov, G, idx, loc);
}

// Calculate ucon, ucov, bcon, bcov from primitive variables, over given range
// Note same range convention as ZSLOOP and other *_vec functions

void get_state_vec(struct GridGeom *G, struct FluidState *S, int loc, int kstart, int kstop, int jstart, int jstop, int istart, int istop)
{
	#pragma omp parallel
    {
		#pragma omp for collapse(3)
        ZSLOOP(kstart, kstop, jstart, jstop, istart, istop)
            ucon_calc(G, S, i, j, k, loc);

		#pragma omp for collapse(3)
        ZSLOOP(kstart, kstop, jstart, jstop, istart, istop)
        {
            GridIndices idx = {i, j, k};
            lower_grid(S->ucon, S->ucov, G, idx, loc);
        }

		#pragma omp for collapse(3)
        ZSLOOP(kstart, kstop, jstart, jstop, istart, istop)
            bcon_calc(S, i, j, k);

		#pragma omp for collapse(3)
        ZSLOOP(kstart, kstop, jstart, jstop, istart, istop)
        {
            GridIndices idx = {i, j, k};
            lower_grid(S->bcon, S->bcov, G, idx, loc);
        }
    }
}


// Calculate components of magnetosonic velocity from primitive variables
// TODO this is a primary candidate for splitting/vectorizing
inline void compute_covariant_contravariant(struct GridGeom *G, int i, int j, int k, int loc, int dir, double *Acov, double *Bcov, double *Acon, double *Bcon)
{
    for (int mu = 0; mu < NDIM; mu++) {
        Acov[mu] = 0.0;
        Bcov[mu] = 0.0;
        Acon[mu] = 0.0;
        Bcon[mu] = 0.0;
    }

    Acov[dir] = 1.0;
    Bcov[0] = 1.0;

    for (int mu = 0; mu < NDIM; mu++) {
        for (int nu = 0; nu < NDIM; nu++) {
            Acon[mu] += G->gcon[loc][mu][nu][j][i] * Acov[nu];
            Bcon[mu] += G->gcon[loc][mu][nu][j][i] * Bcov[nu];
        }
    }
}


inline void mhd_vchar(struct GridGeom *G, struct FluidState *S, int i, int j, int k, int loc, int dir, GridDouble cmax, GridDouble cmin)
{
    double discr, vp, vm, bsq, ee, ef, va2, cs2, cms2, rho, u;
    double Acov[NDIM], Bcov[NDIM], Acon[NDIM], Bcon[NDIM];
    double Asq, Bsq, Au, Bu, AB, Au2, Bu2, AuBu, A, B, C;

    compute_covariant_contravariant(G, i, j, k, loc, dir, Acov, Bcov, Acon, Bcon);

    bsq = bsq_calc(S, i, j, k);
    rho = fabs(S->P[RHO][k][j][i]);
    u = fabs(S->P[UU][k][j][i]);
    ef = rho + gam * u;
    ee = bsq + ef;
    va2 = bsq / ee;
    cs2 = gam * (gam - 1.0) * u / ef;
    cms2 = cs2 + va2 - cs2 * va2;
    cms2 = fmax(SMALL, fmin(1.0, cms2));

    Asq = dot(Acon, Acov);
    Bsq = dot(Bcon, Bcov);
    Au = Bu = 0.0;

    for (int mu = 0; mu < NDIM; mu++) {
        Au += Acov[mu] * S->ucon[mu][k][j][i];
        Bu += Bcov[mu] * S->ucon[mu][k][j][i];
    }

    AB = dot(Acon, Bcov);
    Au2 = Au * Au;
    Bu2 = Bu * Bu;
    AuBu = Au * Bu;

    A = Bu2 - (Bsq + Bu2) * cms2;
    B = 2.0 * (AuBu - (AB + AuBu) * cms2);
    C = Au2 - (Asq + Au2) * cms2;

    discr = fmax(0.0, B * B - 4.0 * A * C);
    discr = sqrt(discr);

    vp = (-B + discr) / (2.0 * A);
    vm = (-B - discr) / (2.0 * A);

    cmax[k][j][i] = fmax(vp, vm);
    cmin[k][j][i] = fmin(vp, vm);
}

// Source terms for equations of motion
inline void get_fluid_source(struct GridGeom *G, struct FluidState *S, GridPrim *dU)
{
#if WIND_TERM
	static struct FluidState *dS;
	static int                firstc = 1;
	if (firstc)
	{
		dS = calloc(1, sizeof(struct FluidState));
		firstc = 0;
	}
#endif

#pragma omp parallel for collapse(3)
	ZLOOP
	{
		double mhd[NDIM][NDIM];

		DLOOP1 mhd_calc(S, i, j, k, mu, mhd[mu]); // TODO make an mhd_calc_vec?

		// Contract mhd stress tensor with connection
		// TODO this is scattered memory access.  Precompute mu,nu sums
		PLOOP(*dU)[ip][k][j][i] = 0.;
		DLOOP2
		{
			for (int gam = 0; gam < NDIM; gam++)
				(*dU)[UU + gam][k][j][i] += mhd[mu][nu] * G->conn[nu][gam][mu][j][i];
		}

		PLOOP(*dU)[ip][k][j][i] *= G->gdet[CENT][j][i];
	}

	// Add a small "wind" source term in RHO,UU
#if WIND_TERM
#pragma omp parallel for simd collapse(2)
	ZLOOP
	{
		// Stolen shamelessly from iharm2d_v3

		/* need coordinates to evaluate particle addtn rate */
		double X[NDIM];
		coord(i, j, k, CENT, X);
		double r, th;
		bl_coord(X, &r, &th);
		double cth = cos(th);

		/* here is the rate at which we're adding particles */
		/* this function is designed to concentrate effect in the
		 funnel in black hole evolutions */
		double drhopdt = 2.e-4 * cth * cth * cth * cth / pow(1. + r * r, 2);

		dS->P[RHO][k][j][i] = drhopdt;

		double Tp = 10.; /* temp, in units of c^2, of new plasma */
		dS->P[UU][k][j][i] = drhopdt * Tp * 3.;

		/* Leave P[U{1,2,3}]=0 to add in particles in normal observer frame */
		/* Likewise leave P[BN]=0 */
	}

	/* add in plasma to the T^t_a component of the stress-energy tensor */
	/* notice that U already contains a factor of sqrt{-g} */
	get_state_vec(G, dS, CENT, 0, N3 - 1, 0, N2 - 1, 0, N1 - 1);
	prim_to_flux_vec(G, dS, 0, CENT, 0, N3 - 1, 0, N2 - 1, 0, N1 - 1, dS->U);

#pragma omp parallel for simd collapse(3)
	PLOOP ZLOOP
	{
		(*dU)[ip][k][j][i] += dS->U[ip][k][j][i];
	}
#endif
}

// Returns b.b (twice magnetic pressure)
inline double bsq_calc(struct FluidState *S, int i, int j, int k)
{
	double bsq = 0.;
	DLOOP1
	{
		bsq += S->bcon[mu][k][j][i] * S->bcov[mu][k][j][i];
	}

	return bsq;
}
