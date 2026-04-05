#include "blocksp_include.hpp"
#include "blocksp_tests.hpp"
//#include "blocksp_eigen_compare.hpp"

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
//Welcome to BlockSp!
//A templated C++ library for block sparse linear algebra. Version 1.0
//
//See more at https://BlockSp.github.io 
//
//This software package is designed to address linear algebra problems
//  where the majority of elements within the linear system are zero (sparse)
//	and the non-zero entries are dense matrices (blocks).
//  Block sparse linear systems can be found in, e.g., high-order 
//  finite element methods (FEM) and discontinuous Galerkin methods (DG), 
//  as well as a wide range of other problems.
// 
//This header-only C++ library provides support for both multiplying and solving 
//	block sparse linear systems (Ax = b). The library includes 
//  containers for block sparse matrices and multi-dimension block vectors,
//  block sparse matrix-matrix and matrix-vector multiplication,
//  preconditioned iterative block sparse system solvers
//	including conjugate gradient (pCG) and pGMRES, 
//  and several preconditioners including block SOR and ILU0.
// 
//The data type, block size, and dimension are specified at compile-time
//  via templating, allowing for fast optimized stack-based operations.
//  Through templating, BlockSp handles both real and complex problems.
//	BlockSp requires C++20 or later.
// 
//Additional dependencies
//  To run BlockSp, you must first include and link the following libraries:
//  MKL -- Intel's Math Kernel Library, used for the dense/block linear algebra.
//	  https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html
//  blitz++ -- A high-performance matrix, vector, and multi-dimensional array library.
//	 https://github.com/blitzpp/blitz
// 
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// 
//Here is a brief overview of the key files, containers, and functions
//  of the BlockSp library:
// 
//blocksp_containers.hpp
//  BlockSp	  --  A block sparse matrix in condensed row storage (CSR) format. 
//	solArray  --  A multi-dimensioned block vector.
//	the file also includes the dot product, infinity norm, and two norm of solArrays.
// 
//	The containers use the following template parameters, specified at compile time:
//    Data type T  --  float, double, std::complex<float>, and std::complex<double>.
//	  int       B  --  block size of the solArray.
//	  int       D  --  dimension of the solArray. D = 1 is the default.
//	  Note: the block size of a BlockSp matrix it typically B or B*D.
//		Setting B = D = 1 reduces the problem to regular sparse linear algebra.
//		Testing indicates that the code often runs fastest when B = 2 and D = 1, use when applicable.		
//
//blocksp_multiply.hpp
//	bsmm	     --  Block sparse matrix-matrix multiplication.
//	bsmv	     --  Block sparse matrix-vector multiplication.
//  transpose  --  Block sparse matrix transpose.
//
//blocksp_solvers.hpp
//	BlockSp provides the following iterative linear system solvers.
//	pCG			   --  Preconditioned block conjugate gradient.
//	pGMRES	   --	 Preconditioned block GMRES.
//	_dr			   --  pGMRES with deflated restarting (in blocksp_deflated_restart.hpp).
//	arnoldi_eigenvalues  --  Arnoldi eigenvalue solver.
// 
//blocksp_preconditioners
//	The following functions precondition the linear system for the iterative solvers:
//  Diag      --  Block diagonal preconditioner.
//	SOR			  --  Successive over-relaxation.
//	ILU0		  --  Incomplete LU factorization with zero fill.
//	IChol0    --  Incomplete Cholesky factorization with zero fill.
// 
//Additional files in the library include:
//	blocksp_dense_multiply.hpp	--  Dense matrix-matrix and matrix-vector multiplication.
//	blocksp_dense_solvers.hpp   --  Dense linear system solvers.
//	blocksp_include.hpp					--  All of the include files for running BlockSp.
//	blocksp_lapack.hpp					--  Templated access to the Lapack routines used by BlockSp.
//	blocksp_utility.hpp				  --  Useful utility functions.
//
// BlockSp Copyright © 2026, Luke P. Corcos Ph.D. (lpcorcos@gmail.com)
//
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
	/////////////////////////////////
	//BlockSp tests
	/////////////////////////////////




	///block_pseudo_heat - multiply vector by matrix, then calculate inverse.
	///Uses a pseudo-heat operator symmetric BlockSp matrix and a random solArray.
	///Data type: float, double, std::complex<float>, or std::complex<double>.

	//BlockSp::tests::block_pseudo_heat<float>();
	BlockSp::tests::block_pseudo_heat<double>();
	//BlockSp::tests::block_pseudo_heat<std::complex<double>>();
	//std::cout << std::endl;

	///////////////////////////////////////

	///LDG_2D_vecLap_test - test BlockSp with a linear system originating
	/// from a nodal Local Discontinuous Galerkin method (LDG) for a 2D vector
	/// Poisson problem on a 16x16 cartesian grid, using bi-cubic piecewise polynomials.
	///The system is symmetric positive definite (SPD) and includes 
	/// an exact solution and right-hand-side vector. 
	///If solved properly, the system should generate a solution vector with 
	/// infinity norm error near 0.00029 when compared to the exact solution.

	//BlockSp::tests::LDG_2D_vecLap_test<2>();
	//BlockSp::tests::LDG_2D_vecLap_test<2,2>();
	//BlockSp::tests::LDG_2D_vecLap_test<4>();

	///////////////////////////////////////

	///rdb_test - test BlockSp with a matrix from the SparseSuit Matrix Collection.
	///This 5000x5000 real matrix comes from a reaction-diffusion-Brusselator model (rdb).
	///This test shows the utility of blocking, which provides significant speedup.
	///https://sparse.tamu.edu/Bai/rdb5000
	///Block size B must divide 5000.
	///Data type: double. 

	//BlockSp::tests::rdb_test<1>();
	//BlockSp::tests::rdb_test<2>();
	//BlockSp::tests::rdb_test<4>();
	//BlockSp::tests::rdb_test<8>();
	//std::cout << std::endl;

	///////////////////////////////////////

	///qc_test - test BlockSp with a matrix from the SparseSuit Matrix Collection.
	///The matrix comes from a quantum chemistry model of H2+ in an electromagnetic field,
	///is Hermitian, with a condition number of ~10^5.
	///The matrix is 2534x2534 and relatively dense, with 463,360 nonzero entries.
	///https://sparse.tamu.edu/Bai/qc2534.
	///Possible block sizes B of 1, 2, 7, and 14.
	///Data type: std::complex<double>.
	///
	///NOTE: ez::sor_pGMRES fails the qc_test due to its high SOR damping parameter,
	///ez::sor_pGMRES_robust, which is SOR pGMRES with a lower parameter (0.6 vs 1.6), 
	/// converges for each B except B = 1.
	///Incomplete LU and Cholesky preconditioners perform much better for this problem. 

	//BlockSp::tests::qc_test<1>();
	//BlockSp::tests::qc_test<2>();
	//BlockSp::tests::qc_test<14>();


	///dense_test tests all of the dense linear system solvers.
	///Errors should all be zero/machine precision.
	///Note: a fast explict solve is used for small matrices (N <= 3),
	/// lapack is used for larger (N > 3)
	//BlockSp::tests::dense_test<double, 2>();
	//BlockSp::tests::dense_test<double, 3>();
	//BlockSp::tests::dense_test<double, 6>();
	//BlockSp::tests::dense_test<std::complex<double>, 2>();
	//BlockSp::tests::dense_test<std::complex<double>, 3>();
	//BlockSp::tests::dense_test<std::complex<double>, 6>();


	//////////////////////////////////
	//Eigen comparison tests
	//////////////////////////////////

	//test set up of Eigen
	//eigen_comparison::eigenTestSetUp();
	//eigen_comparison::eigenSparseExample();


	////////////////////////
	//Real tests

	//SPD tests

	//Eigen's sparse example
	//eigen_comparison::eigenSparseExample_compare<2>();

	//BlockSp's example
	//eigen_comparison::block_pseudo_heat<float>();
	//eigen_comparison::block_pseudo_heat<double>();
	//eigen_comparison::block_pseudo_heat<std::complex<double>>();
	//eigen_comparison::block_pseudo_heat<std::complex<float>>();

	//eigen_comparison::LDG_2D_vecLap_test<1>();
	//eigen_comparison::LDG_2D_vecLap_test<2>();
	//eigen_comparison::LDG_2D_vecLap_test<4>();
	//eigen_comparison::LDG_2D_vecLap_test<2, 2>();


	//Indefinite or nonsymmetric tests

	//eigen_comparison::rdb_test<1>();
	//eigen_comparison::rdb_test<2>();
	//eigen_comparison::rdb_test<8>();

	//eigen_comparison::saylr_test<1>();
	//eigen_comparison::saylr_test<2>();

	//////////////////////////////
	//Complex tests

	//eigen_comparison::qc_test<2>();
	//eigen_comparison::qc_test<14>();


	return 0;

} //end main