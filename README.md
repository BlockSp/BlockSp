# BlockSp
# A templated C++ library for block sparse linear algebra

Welcome to BlockSp!
A templated C++ library for block sparse linear algebra. Version 1.0

You can download BlockSp via

git clone https://github.com/BlockSp/BlockSp.git

or visit https://BlockSp.github.io

This software package is designed to address linear algebra problems
  where the majority of elements within the linear system are zero (sparse)
	and the non-zero entries are dense matrices (blocks).
	Block sparse linear systems can be found in, e.g., high-order 
  finite element methods (FEM) and discontinuous Galerkin methods (DG), 
  as well as a wide range of other problems.
 
This header-only C++ library provides support for both multiplying and solving 
	block sparse linear systems (Ax = b). The library includes 
  containers for block sparse matrices and multi-dimension block vectors,
  block sparse matrix-matrix (bsmm) and matrix-vector (bsmv) multiplication,
  preconditioned iterative block sparse system solvers
	including conjugate gradient (pCG) and pGMRES, 
  and several preconditioners including block SOR and ILU0.
 
The data type, block size, and dimension are specified at compile-time
  via templating, allowing for fast optimized stack-based operations.
  Through templating, BlockSp handles both real and complex problems.
	BlockSp requires C++20 or later.
 
Additional dependencies, to run BlockSp,
you must first include and link the following libraries: \n
  MKL -- Intel's Math Kernel Library, used for the dense/block linear algebra.
	  https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html \n
  blitz++ -- A high-performance matrix, vector, and multi-dimensional array library.
	 https://github.com/blitzpp/blitz
 
/////////////////////////////////////////////////////////////
 
 A typical BlockSp code structure to solve Ax = b is

	int n;					 //vector length/matrix size \n
	int constexpr B; //solArray blocksize, defined at compile time	 \n	
	int constexpr D; //solArray dimension, defined at compile time	 \n
	BlockSp::BlockSp<double, B*D> A(n); \n
	//Assemble A \n
	BlockSp::solArray<double, B, D> b(n); \n
	//Assemble b \n
	auto x { BlockSp::ez::sor_pGMRES(A, B) }; //solve Ax = b

/////////////////////////////////////////////////////////////

Here is a brief overview of the key files, containers, and functions
  of the BlockSp library:
 
blocksp_containers.hpp \n
  BlockSp	  --  A block sparse matrix in condensed row storage (CSR) format. \n
	solArray  --  A multi-dimensioned block vector. \n
	the file also includes the dot product, infinity norm, and two norm of solArrays. \n
  
	 \n
	The containers use the following template parameters, specified at compile time: \n
    Data type T  --  float, double, std::complex<float>, and std::complex<double>. \n
	  int       B  --  block size of the solArray.																		\n
	  int       D  --  dimension of the solArray. D = 1 is the default. \n
	  Note: the block size of a BlockSp matrix it typically B or B*D.		\n
		Setting B = D = 1 reduces the problem to regular sparse linear algebra. \n
		Testing indicates that the code often runs fastest when B = 2 and D = 1, use when applicable.		

blocksp_multiply.hpp	\n
	bsmm	    --  Block sparse matrix-matrix multiplication. \n
	bsmv	    --  Block sparse matrix-vector multiplication. \n
  transpose --  Block sparse matrix transpose.

blocksp_solvers.hpp \n
	BlockSp provides the following iterative linear system solvers.	\n
	pCG			  --  Preconditioned block conjugate gradient.					\n
	pGMRES	  --	Preconditioned block GMRES.		\n
	_dr			  --  pGMRES with deflated restarting (in blocksp_deflated_restart.hpp).\n
	arnoldi_eigenvalues  --  Arnoldi eigenvalue solver.
 
blocksp_preconditioners \n
	The following functions precondition the linear system for the iterative solvers: \n 
	Diag      --  Block diagonal preconditioner. \n
	SOR			  --  Successive over-relaxation.	\n
	ILU0		  --  Incomplete LU factorization with zero fill.				\n
	IChol0    --  Incomplete Cholesky factorization with zero fill.	
 
Additional files in the library include: \n
	blocksp_dense_multiply.hpp	--  Dense matrix-matrix and matrix-vector multiplication. \n
	blocksp_dense_solvers.hpp   --  Dense linear system solvers.													\n
	blocksp_include.hpp					--  All of the include files for running BlockSp.					\n
	blocksp_lapack.hpp					--  Templated access to the Lapack routines used by BlockSp. \n
	blocksp_utility.hpp				  --  Useful utility functions.


 BlockSp Copyright © 2026, Luke P. Corcos Ph.D. (lpcorcos@gmail.com)

