# BlockSp
# A templated C++ library for block sparse linear algebra

Welcome to BlockSp!
A templated C++ library for block sparse linear algebra. Version 2.0

***** Version 2.0 introduces the compressed form of the BlockSp matrix.
This provides a significant speedup to bsmv, the preconditioners, and the solvers.
This version of BlockSp is competitive with and even 
outperforms Eigen in several block-sparse tests. *****
<br/>

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
you must first include and link the following libraries: <br/>
  MKL -- Intel's Math Kernel Library, used for the dense/block linear algebra. <br/>
	  https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html <br/>
  blitz++ -- A high-performance matrix, vector, and multi-dimensional array library. <br/>
	 https://github.com/blitzpp/blitz
 
/////////////////////////////////////////////////////////////
 
 An "ez" BlockSp code structure to solve Ax = b is

 ```
	int n;           //vector length/matrix size 
	int constexpr B; //solArray blocksize, defined at compile time	 
	int constexpr D; //solArray dimension, defined at compile time	 
	BlockSp::BlockSp<double, B*D> A(n); 
	//Assemble A 
	BlockSp::solArray<double, B, D> b(n); 
	//Assemble b 
	auto x { BlockSp::ez::sor_pGMRES(A, B) }; //solve Ax = b
```

A faster version involves compressing the BlockSp matrix:

```
	int n;           //vector length/matrix size 
	int constexpr B; //solArray blocksize, defined at compile time	 
	int constexpr D; //solArray dimension, defined at compile time	 
	BlockSp::BlockSp<double, B*D> A(n); 
	//Assemble A 
	BlockSp::solArray<double, B, D> b(n); 
	//Assemble b 
	BlockSp::SOR_pGMRES sor_pgmres(A);				//construct preconditioner/solver
	auto A_compr{ A.compress_and_move() }			//compress matrix A
	auto x { sor_pgmres.solve(A_compr, b) };	//solve Ax = b using compressed matrix
```

/////////////////////////////////////////////////////////////

Here is a brief overview of the key files, containers, and functions
  of the BlockSp library:
 
blocksp_containers.hpp <br/>
  BlockSp	  --  A block sparse matrix in condensed row storage (CSR) format. <br/>
	BlockSp_Compr				-- a compressed BlockSp matrix. Read only. <br/>
	BlockSp_ComprSorted -- a compressed and sorted BlockSp matrix. Read only. <br/> 
	solArray  --  A multi-dimensioned block vector. <br/>
	the file also includes the dot product, infinity norm, and two norm of solArrays. <br/>
  
```	 
	The containers use the following template parameters, specified at compile time:
    Data type T  --  float, double, std::complex<float>, and std::complex<double>.
	  int       B  --  block size of the solArray.
	  int       D  --  dimension of the solArray. D = 1 is the default. 
	  Note: the block size of a BlockSp matrix it typically B or B*D.		
		Setting B = D = 1 reduces the problem to regular sparse linear algebra.
		BlockSp is best suited to problems with natural block structure.
```

blocksp_multiply.hpp	<br/>
	bsmm	    --  Block sparse matrix-matrix multiplication. <br/>
	bsmv	    --  Block sparse matrix-vector multiplication. <br/>
  transpose --  Block sparse matrix transpose.

blocksp_solvers.hpp <br/>
	BlockSp provides the following iterative linear system solvers.	<br/>
	pCG			  --  Preconditioned block conjugate gradient.					<br/>
	pGMRES	  --	Preconditioned block GMRES.		<br/>
	_dr			  --  pGMRES with deflated restarting (in blocksp_deflated_restart.hpp).<br/>
	arnoldi_eigenvalues  --  Arnoldi eigenvalue solver.
 
blocksp_preconditioners <br/>
	The following functions precondition the linear system for the iterative solvers: <br/>
	Diag      --  Block diagonal preconditioner. <br/>
	SOR			  --  Successive over-relaxation.	<br/>
	ILU0		  --  Incomplete LU factorization with zero fill.	<br/>
	IChol0    --  Incomplete Cholesky factorization with zero fill. SPD systems only.	
 
Additional files in the library include: <br/>
	blocksp_dense_multiply.hpp	--  Dense matrix-matrix and matrix-vector multiplication. <br/>
	blocksp_dense_solvers.hpp   --  Dense linear system solvers. <br/>
	blocksp_include.hpp					--  All of the include files for running BlockSp.	<br/>
	blocksp_lapack.hpp					--  Templated access to the Lapack routines used by BlockSp. <br/>
	blocksp_utility.hpp				  --  Useful utility functions.


 BlockSp Copyright © 2026, Luke P. Corcos Ph.D. (lpcorcos@gmail.com)

