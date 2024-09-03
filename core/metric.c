/******************************************************************************
 *                                                                            *
 * METRIC.C                                                                   *
 *                                                                            *
 * HELPER FUNCTIONS FOR METRIC TENSORS                                        *
 *                                                                            *
 ******************************************************************************/


/*
 * This file contains a set of functions for handling various tensor operations 
 * related to the metric and connection coefficients in a general relativistic 
 * magnetohydrodynamics (GRMHD) simulation. These operations include raising and 
 * lowering tensor indices, computing the determinant and inverse of matrices, 
 * and calculating connection coefficients (Christoffel symbols) used in the simulation.
 *
 * Key functionalities in this file include:
 *
 * - **Metric Operations:**
 *   - `gcon_func`: Computes the contravariant metric `gcon` from the covariant 
 *     metric `gcov` by inverting the metric matrix. It returns the square root 
 *     of the absolute value of the determinant of `gcov`.
 *   - `get_gcov`, `get_gcon`: Functions to extract the covariant and contravariant 
 *     metric components from the `GridGeom` structure.
 *   - `lower` and `raise`: Functions to lower and raise the indices of a rank-1 
 *     tensor using the provided covariant (`gcov`) or contravariant (`gcon`) metric.
 *
 * - **Connection Coefficients (Christoffel Symbols):**
 *   - `conn_func`: Computes the connection coefficients \(\Gamma^{i}_{j,k}\) using 
 *     finite differencing of the metric components. This function is crucial for 
 *     calculating the geodesic equation and the evolution of fluid elements in 
 *     curved spacetime.
 *
 * - **Tensor Index Manipulations:**
 *   - `lower_grid`, `raise_grid`: Functions to lower or raise the indices of rank-1 
 *     tensors (vectors) over the entire grid, using the metric stored in the `GridGeom` structure.
 *   - `lower_grid_vec`: Lowers the indices of a grid of contravariant vectors to covariant ones 
 *     across a specified range of grid points.
 *   - `dot` and `dot_grid`: Compute the dot product of a contravariant vector with a covariant 
 *     vector either for a single point (`dot`) or across the grid (`dot_grid`).
 *
 * - **Matrix Operations:**
 *   - `invert`: Computes the inverse of a 4x4 matrix and returns its determinant. 
 *     This function is essential for operations where the metric or other 4x4 matrices 
 *     need to be inverted, such as when converting between covariant and contravariant 
 *     components.
 *   - `determinant`, `adjoint`, `MINOR`: Supporting functions for matrix operations, 
 *     including calculation of the determinant, adjoint, and minors of a matrix. These 
 *     are used within the `invert` function to compute the inverse of a matrix.
 *
 * The file uses the GSL (GNU Scientific Library) for some matrix operations (commented out) 
 * as an alternative to the manual implementation of matrix inversion and determinant 
 * calculation. These functions are optimized using inline definitions for performance 
 * in a high-resolution simulation environment.
 *
 * Overall, the functions in this file are crucial for ensuring that the simulation 
 * accurately models the effects of curved spacetime on the fluid and electromagnetic 
 * fields, which is essential for realistic GRMHD simulations.
 */

#include "decs.h"


void   adjoint(double m[16], double adjOut[16]);


inline double gcon_func(double gcov[NDIM][NDIM], double gcon[NDIM][NDIM])
{
	double gdet = invert(&gcov[0][0], &gcon[0][0]);
	return sqrt(fabs(gdet));
}

inline void get_gcov(struct GridGeom *G, int i, int j, int loc, double gcov[NDIM][NDIM])
{
	DLOOP2 gcov[mu][nu] = G->gcov[loc][mu][nu][j][i];
}

inline void get_gcon(struct GridGeom *G, int i, int j, int loc, double gcon[NDIM][NDIM])
{
	DLOOP2 gcon[mu][nu] = G->gcon[loc][mu][nu][j][i];
}

inline void calculate_partial_derivatives(struct GridGeom *G, double gh[NDIM][NDIM], 
										double gl[NDIM][NDIM], int i, int j, int mu, int loc, double conn_out[NDIM][NDIM][NDIM])
{
    for (int lam = 0; lam < NDIM; lam++)
    {
        for (int nu = 0; nu < NDIM; nu++)
        {
            conn_out[lam][nu][mu] = (gh[lam][nu] - gl[lam][nu]) / (2 * DELTA);
        }
    }
}


inline void conn_func(struct GridGeom *G, GridIndices idx)
{
    double tmp[NDIM][NDIM][NDIM];
    double X[NDIM];
    double gh[NDIM][NDIM], gl[NDIM][NDIM];
    int i = idx.i, j = idx.j, k = idx.k;

    coord(i, j, k, CENT, X);

    for (int mu = 0; mu < NDIM; mu++)
    {
        double Xh[NDIM], Xl[NDIM];
        memcpy(Xh, X, sizeof(X));  // Copier X dans Xh
        memcpy(Xl, X, sizeof(X));  // Copier X dans Xl

        Xh[mu] += DELTA;
        Xl[mu] -= DELTA;

        gcov_func(Xh, gh);
        gcov_func(Xl, gl);

        for (int lam = 0; lam < NDIM; lam++)
        {
            for (int nu = 0; nu < NDIM; nu++)
            {
                double diff = (gh[lam][nu] - gl[lam][nu]) / (2 * DELTA);
                G->conn[lam][nu][mu][j][i] = diff;
            }
        }
    }

    // Réorganiser pour trouver \Gamma_{lam nu mu}
    for (int lam = 0; lam < NDIM; lam++)
    {
        for (int nu = 0; nu < NDIM; nu++)
        {
            for (int mu = 0; mu < NDIM; mu++)
            {
                tmp[lam][nu][mu] = 0.5 * (G->conn[nu][lam][mu][j][i] + G->conn[mu][lam][nu][j][i] - G->conn[mu][nu][lam][j][i]);
            }
        }
    }

    // Élever l'indice pour obtenir \Gamma^lam_{nu mu}
    for (int lam = 0; lam < NDIM; lam++)
    {
        for (int nu = 0; nu < NDIM; nu++)
        {
            for (int mu = 0; mu < NDIM; mu++)
            {
                double sum = 0.0;
                for (int kap = 0; kap < NDIM; kap++)
                {
                    sum += G->gcon[CENT][lam][kap][j][i] * tmp[kap][nu][mu];
                }
                G->conn[lam][nu][mu][j][i] = sum;
            }
        }
    }
	if (G->gcon[CENT][0][0][j][i] < 0.)
	{
		fprintf(stderr, "gcon[0][0] < 0\n");
	}
}

inline void lower_grid(GridVector vcon, GridVector vcov, struct GridGeom *G, GridIndices idx, int loc)
{
    int i = idx.i;
    int j = idx.j;
    int k = idx.k;
    
    for (int mu = 0; mu < NDIM; mu++)
    {
        vcov[mu][k][j][i] = 0.;
        for (int nu = 0; nu < NDIM; nu++)
        {
            vcov[mu][k][j][i] += G->gcov[loc][mu][nu][j][i] * 
								vcon[nu][k][j][i];
        }
    }
}


// Lower the grid of contravariant rank-1 tensors to covariant ones
void lower_grid_vec(GridVector vcon, GridVector vcov, struct GridGeom *G, int kstart, int kstop, int jstart, int jstop, int istart, int istop, int loc)
{
#pragma omp parallel for simd collapse(3)
	DLOOP1
	{
		ZSLOOP(kstart, kstop, jstart, jstop, istart, istop)
		vcov[mu][k][j][i] = 0.;
	}
#pragma omp parallel for simd collapse(4)
	DLOOP2
	{
		ZSLOOP(kstart, kstop, jstart, jstop, istart, istop)
		vcov[mu][k][j][i] += G->gcov[loc][mu][nu][j][i] * vcon[nu][k][j][i];
	}
}

inline void raise_grid(GridVector vcov, GridVector vcon, struct GridGeom *G, int i, int j, int k, int loc)
{
	for (int mu = 0; mu < NDIM; mu++)
	{
		vcon[mu][k][j][i] = 0.;
		for (int nu = 0; nu < NDIM; nu++)
		{
			vcon[mu][k][j][i] += G->gcon[loc][mu][nu][j][i] * vcov[nu][k][j][i];
		}
	}
}

// TODO revise this out of the following: Fcon_calc, fixup1zone, get_state,
// Utoprim, Wp_func And while you're at it revise out get_state
inline void lower(double ucon[NDIM], double gcov[NDIM][NDIM], double ucov[NDIM])
{
	for (int mu = 0; mu < NDIM; mu++)
	{
		ucov[mu] = 0.;
		for (int nu = 0; nu < NDIM; nu++)
		{
			ucov[mu] += gcov[mu][nu] * ucon[nu];
		}
	}
}

// Raise a covariant rank-1 tensor to a contravariant one
inline void raise(double ucov[NDIM], double gcon[NDIM][NDIM], double ucon[NDIM])
{
	for (int mu = 0; mu < NDIM; mu++)
	{
		ucon[mu] = 0.;
		for (int nu = 0; nu < NDIM; nu++)
		{
			ucon[mu] += gcon[mu][nu] * ucov[nu];
		}
	}
}

inline double dot_grid(GridVector vcon, GridVector vcov, int i, int j, int k)
{
	double dot = 0.;
	for (int mu = 0; mu < NDIM; mu++)
	{
		dot += vcon[mu][k][j][i] * vcov[mu][k][j][i];
	}
	return dot;
}

inline double dot(double vcon[NDIM], double vcov[NDIM])
{
	double dot = 0.;
	for (int mu = 0; mu < NDIM; mu++)
	{
		dot += vcon[mu] * vcov[mu];
	}
	return dot;
}

// TODO debug this GSL version to save lines?
// double invert(double *m, double *inv) {
//  gsl_matrix_view mat = gsl_matrix_view_array(m, 4, 4);
//  gsl_matrix_view inv_mat = gsl_matrix_view_array(inv, 4, 4);
//  gsl_permutation * p = gsl_permutation_alloc(4);
//  int s;
//
//  gsl_linalg_LU_decomp(&mat.matrix, p, &s);
//  gsl_linalg_LU_invert(&mat.matrix, p, &inv_mat.matrix);
//
//  gsl_permutation_free(p);
//  return gsl_linalg_LU_det(&mat.matrix, s);
//}


/*
 * The `MINOR` function calculates the minor of a 3x3 submatrix within a 4x4 matrix.
 * 
 * In linear algebra, the minor of a matrix element is the determinant of the 
 * smaller matrix formed by removing the row and column containing that element.
 * The `MINOR` function is used here to compute the determinant of a 3x3 submatrix 
 * that excludes a specific row and column from the original 4x4 matrix.
 * 
 * Parameters:
 * - `m`: A 4x4 matrix represented as a flat array of 16 elements.
 * - `r0`, `r1`, `r2`: The row indices of the 3x3 submatrix within the 4x4 matrix.
 * - `c0`, `c1`, `c2`: The column indices of the 3x3 submatrix within the 4x4 matrix.
 * 
 * The function computes the determinant of the 3x3 submatrix using the standard 
 * cofactor expansion method. The result is used in the calculation of the adjoint 
 * matrix, which is necessary for finding the inverse of the original 4x4 matrix.
 * 
 * Return value:
 * - The function returns the determinant of the specified 3x3 submatrix.
 */

inline double MINOR(double m[16], int r0, int r1, int r2, int c0, int c1, int c2)
{
	return m[4 * r0 + c0] * (m[4 * r1 + c1] * m[4 * r2 + c2] -
		m[4 * r2 + c1] * m[4 * r1 + c2]) - m[4 * r0 + c1] * 
		(m[4 * r1 + c0] * m[4 * r2 + c2] - m[4 * r2 + c0] * 
		m[4 * r1 + c2]) + m[4 * r0 + c2] * (m[4 * r1 + c0] * 
		m[4 * r2 + c1] - m[4 * r2 + c0] * m[4 * r1 + c1]);
}

inline void adjoint(double m[16], double adjOut[16])
{
	adjOut[0] = MINOR(m, 1, 2, 3, 1, 2, 3);
	adjOut[1] = -MINOR(m, 0, 2, 3, 1, 2, 3);
	adjOut[2] = MINOR(m, 0, 1, 3, 1, 2, 3);
	adjOut[3] = -MINOR(m, 0, 1, 2, 1, 2, 3);

	adjOut[4] = -MINOR(m, 1, 2, 3, 0, 2, 3);
	adjOut[5] = MINOR(m, 0, 2, 3, 0, 2, 3);
	adjOut[6] = -MINOR(m, 0, 1, 3, 0, 2, 3);
	adjOut[7] = MINOR(m, 0, 1, 2, 0, 2, 3);

	adjOut[8] = MINOR(m, 1, 2, 3, 0, 1, 3);
	adjOut[9] = -MINOR(m, 0, 2, 3, 0, 1, 3);
	adjOut[10] = MINOR(m, 0, 1, 3, 0, 1, 3);
	adjOut[11] = -MINOR(m, 0, 1, 2, 0, 1, 3);

	adjOut[12] = -MINOR(m, 1, 2, 3, 0, 1, 2);
	adjOut[13] = MINOR(m, 0, 2, 3, 0, 1, 2);
	adjOut[14] = -MINOR(m, 0, 1, 3, 0, 1, 2);
	adjOut[15] = MINOR(m, 0, 1, 2, 0, 1, 2);
}

inline double determinant(double m[16])
{
	return m[0] * MINOR(m, 1, 2, 3, 1, 2, 3) - m[1] * 
				MINOR(m, 1, 2, 3, 0, 2, 3) + m[2] * 
				MINOR(m, 1, 2, 3, 0, 1, 3) - m[3] * 
				MINOR(m, 1, 2, 3, 0, 1, 2);
}

inline double invert(double *m, double *invOut)
{
	adjoint(m, invOut);

	double det = determinant(m);
	double inv_det = 1. / det;
	for (int i = 0; i < 16; ++i)
	{
		invOut[i] = invOut[i] * inv_det;
	}

	return det;
}
