#ifndef BLOCKSP_TESTS_HPP
#define BLOCKSP_TESTS_HPP

#include <chrono>
#include <complex>
#include <fstream>

#include "blocksp_include.hpp"

///blocksp_tests.hpp provides a test engine and two tests for the BlockSp library.
///The engine tests BlockSp's matrix-vector multiplication and iterative solvers.
///
///The frist provided test solves a random symmetric BlockSp matrix with 
/// a random solArray(a multi-dimensional vector).
///	The matrix is outlined similarly to a heat operator: ASym = (I - d A^T A).
/// 
///The second test reads in the provided matrix LDG_vecLap_2D.mtx,
/// along with exact solution and right-hand-side
///
///Additionally, there are two commented out tests for real world sparse matrices
/// from the SparseSuite library. These must be downloaded before use.
///		(1) rdb_test - the matrix is real and comes from a reaction-diffusion-Brusselator model.
///		(1) qcd_test - the matrix is complex and comes from quantum chemistry.
///		Both matrices are converted from .mtx format to BlockSp using the mtxMatrix_to_BlockSp function.
///
///Note: these tests use namespace BlockSp.
namespace BlockSp::tests
{
	///////////////////////////
	//Test engine
	///////////////////////////

	///Engine to test BlockSp's iterative solvers.
	///Similar to eigen_compare in blocksp_eigen_compare.hpp
	/// 
	///Inputs a BlockSp matrix A and a pointer to a solArray
	/// for the exact solution x and/or right-hand-side b. 
	///Then solves Ax = b. 
	///If x and b provided, Ax = b is solved and compared to exact solution. 
	///If only x is provided, b = Ax is calculated via bsmv, then used as input for the solvers.
	///If only b is provided, the engine solves Ax = b, but does not provide the error.
	///If neither x nor b is provided, a random normal solution vector x is calculated
	/// and rhs b generated.
	/// 
	///Tests 
	///BlockSp's SOR and ILU0 pGMRES, and,
	///if symmetric positive definite (SPD), SOR and IChol0 pCG.
	///If noPre, also test solvers with no preconditioner.
	///If diag, test solvers with diagonal preconditioner. 
	///If robust, tests BlockSp's ez::sor_pGMRES_robust in addition to ez::sor_pGMRES.
	template<typename T, int B, int D = 1>
	void test_engine(const BlockSp<T, B* D>& A, const solArray<T, B, D>* x_ptr,
		const solArray<T, B, D>* b_ptr, bool SPD = false,
		bool noPre = false, bool diag = false, bool robust = false)
	{
		std::cout << "Start test_engine" << std::endl;
		int m = A.m(); int n = A.n();
		std::cout << "Dimension " << D << ", Blocksize " << B << std::endl;
		std::cout << "Matrix size: " << m << "x" << n << ", non-zeros: "
			<< A.nz() * B * D * B * D << std::endl;
		std::cout << std::endl;

		//timing lambda
		std::chrono::time_point<std::chrono::steady_clock> start;
		auto time_since = [](auto start) 
			{ return  std::chrono::duration<double>(std::chrono::steady_clock::now() - start); };

		//Assemble solution x_sol and right-hand-side b
		bool have_sol = false;
		solArray<T, B, D> x_sol(n), b(m);
		if (x_ptr && b_ptr) //both solution x and rhs b provided
		{
			assert(x_ptr->len() == n);
			assert(b_ptr->len() == m);
			x_sol = *x_ptr;
			b = *b_ptr;
			have_sol = true;
		}
		else if (x_ptr)		 //solution x provided
		{
			assert(x_ptr->len() == n);
			x_sol = *x_ptr;						 //provided exact solution
			bsmv(b, A, x_sol);				 //b = Ax, BlockSp matrix-vector multiplication (bsmv)!
			have_sol = true;
		}
		else if (b_ptr)		 //rhs b provided
		{
			assert(b_ptr->len() == m);
			b = *b_ptr;
		}
		else //neither x nor b provided, use random normal solution
		{
			x_sol = util::rand_solArray<T, B, D>(n);
			bsmv(b, A, x_sol); //b = Ax, BlockSp matrix-vector multiplication (bsmv)!
			have_sol = true;
		}

		//Tolerance for convergence. Different for double or float.
		ScalarType_t<T> tol;
		tol = std::is_same<double, ScalarType_t<T>>::value ? 1.0e-8 : 1.0e-5;
		int maxIters = 5000;	//maximum iterations
		int restart = 30;			//size of Krylov subspace before restarting pGMRES
		int re_k = 10;				//size of restarted Krylov subspace for pGMRES

		/////////////////
		//Run solvers

		if (noPre)
		{
			//pGMRES, no preconditioner, no restart
			{
				start = std::chrono::steady_clock::now();
				auto x{ pGMRES<T, B, D>(A, b, tol, [](const auto& x) {}, maxIters, -1, -1, true) };
				std::cout << "BlockSp no pre no re GMRES time  " << time_since(start) << std::endl;
				if (have_sol)	std::cout << "error                            " << infNorm(x - x_sol) << std::endl;
				std::cout << std::endl;
			}

			//pGMRES_dr, no preconditioner.
			{
				start = std::chrono::steady_clock::now();
				auto x{ pGMRES<T, B, D>(A, b, tol, [](const auto& x) {}, maxIters, restart, re_k, true) };
				std::cout << "BlockSp no pre GMRES_dr time     " << time_since(start).count() << std::endl;
				if (have_sol) std::cout << "error                            " << infNorm(x - x_sol) << std::endl;
				std::cout << std::endl;
			}
		}

		//block diagonal precondtioner
		if (diag)
		{
			//no restart
			{
				start = std::chrono::steady_clock::now();
				preConditioners::Diag<dense::LUFactor<T, B* D>, T, B, D> diag(A);
				auto x{ pGMRES<T, B, D>(A, b, tol, diag, maxIters, -1, -1, true) };
				std::cout << "BlockSp diag pGMRES time         " << time_since(start) << std::endl;
				if (have_sol) std::cout << "error                            " << infNorm(x - x_sol) << std::endl;
				std::cout << std::endl;
			}

			//with restart
			{
				start = std::chrono::steady_clock::now();
				preConditioners::Diag<dense::LUFactor<T, B* D>, T, B, D> diag(A);
				auto x{ pGMRES<T, B, D>(A, b, tol, diag, maxIters, restart, re_k, true) };
				std::cout << "BlockSp diag pGMRES_dr time      " << time_since(start) << std::endl;
				if (have_sol) std::cout << "error                            " << infNorm(x - x_sol) << std::endl;
				std::cout << std::endl;
			}
		}

		//ILU0 pGMRES, not restarted
		{
			start = std::chrono::steady_clock::now();
			ILU0_pGMRES<T, B, D> ilu0_pgmres(A, tol, maxIters, -1, -1, true);
			auto x{ ilu0_pgmres.solve(A, b) };
			std::cout << "BlockSp ILU0 pGMRES time         " << time_since(start) << std::endl;
			if (have_sol) std::cout << "error                            " << infNorm(x - x_sol) << std::endl;
			std::cout << std::endl;
		}

		//ILU0 pGMRES_dr,
		{
			start = std::chrono::steady_clock::now();
			ILU0_pGMRES<T, B, D> ilu0_pgmres(A, tol, maxIters, restart, re_k, true);
			auto x{ ilu0_pgmres.solve(A, b) };
			std::cout << "BlockSp ILU0 pGMRES_dr time      " << time_since(start) << std::endl;
			if (have_sol) std::cout << "error                            " << infNorm(x - x_sol) << std::endl;
			std::cout << std::endl;
		}

		//BlockSp's ez::sor_pGMRES
		//pGMRES, SOR preconditioner with LU inversion, no restart
		{
			start = std::chrono::steady_clock::now();
			auto x{ ez::sor_pGMRES(A, b, true) };
			std::cout << "BlockSp ez::sor_pGMRES time      " << time_since(start) << std::endl;
			if (have_sol) std::cout << "error                            " << infNorm(x - x_sol) << std::endl;
			std::cout << std::endl;
		}

		//BlockSp's ez::sor_pGMRES_dr
		//pGMRES, SOR preconditioner with LU inversion, with deflated_restart
		{
			start = std::chrono::steady_clock::now();
			auto x{ ez::sor_pGMRES_dr(A, b, true) };
			std::cout << "BlockSp ez::sor_pGMRES_dr time   " << time_since(start) << std::endl;
			if (have_sol) std::cout << "error                            " << infNorm(x - x_sol) << std::endl;
			std::cout << std::endl;
		}

		if (robust)
		{
			//BlockSp's ez::sor_pGMRES_robust
			//pGMRES, light damping SOR preconditioning with LU inversion, no restart
			{
				start = std::chrono::steady_clock::now();
				auto x{ ez::sor_pGMRES_robust(A, b, true) };
				std::cout << "BlockSp ez::sor_pGMRES_robust time " << time_since(start) << std::endl;
				if (have_sol) std::cout << "error                            " << infNorm(x - x_sol) << std::endl;
				std::cout << std::endl;
			}
		}

		/////If SPD --Symmetric Positive Definite-- also test Conjugate gradient (CG).
		if constexpr (!is_complex_v<T>)
			if (SPD)
			{
				//cg, no preconditioner.
				if (noPre)
				{
					{
						start = std::chrono::steady_clock::now();
						auto x{ pCG<T, B, D>(A, b, tol, [](const auto&) {}, maxIters, true) };
						std::cout << "BlockSp no Pre CG time           " << time_since(start) << std::endl;
						if (have_sol) std::cout << "error                            " << infNorm(x - x_sol) << std::endl;
						std::cout << std::endl;
					}
				}

				//pcg, diagonal preconitioner.
				if (diag)
				{
					start = std::chrono::steady_clock::now();
					preConditioners::Diag<dense::CholeskyFactor<T, B* D>, T, B, D> diag(A);
					auto x{ pCG<T, B, D>(A, b, tol, diag, maxIters, true) };
					std::cout << "BlockSp diag pCG time            " << time_since(start) << std::endl;
					if (have_sol) std::cout << "error                            " << infNorm(x - x_sol) << std::endl;
					std::cout << std::endl;
				}

				//pCG, SOR preconditioner with Cholesky inversion.
				{
					start = std::chrono::steady_clock::now();
					SOR_pCG<T, B, D> sor_pcg_solve(A);
					sor_pcg_solve.change_printIters(true);
					auto x{ sor_pcg_solve.solve(A, b) };
					std::cout << "BlockSp SOR_pCG_Solver time      " << time_since(start) << std::endl;
					if (have_sol) std::cout << "error                            " << infNorm(x - x_sol) << std::endl;
					std::cout << std::endl;
				}

				//pCG, IChol0 preconditioner.
				{
					start = std::chrono::steady_clock::now();
					IChol0_pCG<T, B, D> iChol0_pcg(A);
					iChol0_pcg.change_printIters(true);
					auto x{ iChol0_pcg.solve(A, b) };
					std::cout << "BlockSp IChol0_pCG_Solver time   " << time_since(start) << std::endl;
					if (have_sol) std::cout << "error                            " << infNorm(x - x_sol) << std::endl;
					std::cout << std::endl;
				}

			}

		std::cout << "end test_engine" << std::endl;
	}


	////////////////////////
	//Tests
	////////////////////////

	///Multiply a random BlockSp matrix with a solArray(multi-dimensional vector), then calculate inverse.
	///Tests BlockSp matrix-matrix (bsmm) and matrix-vector (bsmv) products.
	///Tests preconditioned CG and GMRES solvers.
	///Can use data type T = float, double, std::complex<float>, or std::complex<double>.
	template<typename T> requires lapack::is_lapack_type<T>
	void block_pseudo_heat()
	{
		//Try changing D, B, l, nCol, and delta. 
		int constexpr D = 2;				//Vector Dimension //D and B are compile time constants!
		int constexpr B = 16;				//Vector Blocksize
		int constexpr BD = B * D;
		using S = ScalarType_t<T>;  //Float or double, derived from T.
		int l = 256;								//Vector/BlockSp matrix length

		//Make random BlockSp matrix A and symmetric positive definite matrix ASym = I - delta*A^T*A
		int nCol = 4;			 //Number of nonzero columns's per row of A. 
		//ASym will have ~2*nCol nonzero columns per row.
		BlockSp<T, BD> A{ util::rand_BlockSp<T, BD>(l, l, nCol) };
		S delta = 2.0e-4; //The value of delta may be tuned to stress test the solvers.
		//If delta is too large, the solvers will diverge. 
		auto ASym{ bs_id<T, BD>(l) - delta * bsmm(A, A, true) }; //ASym = I - delta*A^T*A
		std::cout << "block_pseudo_heat: data type " << typeid(T).name() << std::endl;
		ASym.shrink_to_fit(); //Once a BlockSp matrix is assembled, 
		//use this function to shrink its vector's storage.


		//Find the linear system A's eigenvalues with Arnoldi solver
		//The eigenvalues of a preconditioned linear system can be also found. 
		//auto ev = arnoldi_eigenvalues(ASym, 30, nullptr, true);

		//run test engine for solver timing
		test_engine<T, B, D>(ASym, nullptr, nullptr, true, true, true, true);

		std::cout << "end block_pseudo_heat" << std::endl;
		std::cout << std::endl;
	}

	//Forward declaration of .mtx to BlockSp/solArray converters.
	template<typename T, int B = 1>
	BlockSp<T, B> mtxMatrix_to_BlockSp(const std::string& mat_loc,
		unsigned int start_line = 0, bool symmetric = false,
		bool base_zero = false, bool printMatSize = false);
	template<typename T, int B = 1, int D = 1>
	solArray<T, B, D> mtxArray_to_solArray(const std::string& vec_loc,
		unsigned int start_line = 0, bool printVecSize = false);

	
	///LDG_2D_vecLap_test - test BlockSp with a linear system originating
	/// from a nodal Local Discontinuous Galerkin method (LDG) for a 2D vector
	/// Poisson problem on a 16x16 cartesian grid, using bi-cubic piecewise polynomials.
	///The system is symmetric positive definite (SPD) and includes 
	/// an exact solution and right-hand-side vector. 
	///If solved properly, the system should generate a solution vector with 
	/// infinity norm error near 0.00029 when compared to the exact solution.
	template<int B, int D = 1>
	void LDG_2D_vecLap_test()
	{
		std::cout << "/////////////////////////////////////" << std::endl;
		std::cout << "LDG_2D_vecLap_test B = " << B << ", D = " << D << std::endl;

		//Get matrix A from .mtx file
		std::string mtx_A = "test_mats/LDG_2D_vecLap.mtx";
		auto A{ mtxMatrix_to_BlockSp<double, B * D>(mtx_A, 1, true) };
		A.shrink_to_fit(); //Once a BlockSp matrix is assembled, 
		//use this function to shrink its vector's storage.

		//Get solution x and rhs b from .mtx files
		std::string mtx_x = "test_mats/LDG_2D_vecLap_sol.mtx";
		auto x{ mtxArray_to_solArray<double, B, D>(mtx_x, 1) };

		std::string mtx_b = "test_mats/LDG_2D_vecLap_b.mtx";
		auto b{ mtxArray_to_solArray<double, B, D>(mtx_b, 1) };

		//Compare Eigen and BlockSp, random normal solution
		test_engine(A, &x, &b, true, true, true);
		std::cout << "end LDG_2D_vecLap_test" << std::endl;
		std::cout << std::endl;
	}
	//*/

	///////////////////////////
	//SuiteSparse Tests
	///////////////////////////

	/*
	///Test solvers using the SuiteSparse rdb5000 matrix by Bai
	/// https://sparse.tamu.edu/Bai/rdb5000.
	///The matrix comes from a reaction-diffusion-Brusselator model,
	///is non-symmetric indefinite, with a condition number of ~10^3.
	///The matrix is 5000x5000 with 29,600 nonzero entries.
	///We test the solver speed with several different block sizes.
	///solArray dimension D is set to one.
	template<int B>
	void rdb_test()
	{
		std::string rdb_mtxMat = "matrix_location/rbd5000.mtx";
		auto A{ mtxMatrix_to_BlockSp<double, B>(rdb_mtxMat, 14) }; //matrix size starts on line 14 of .mtx file
		A.shrink_to_fit();
		std::cout << "rdb_test with Blocksize B = " << B << std::endl;
		std::cout << "Size of BlockSp matrix: " << A.m() << ' ' << A.n()
			<< ' ' << A.nz() << std::endl;


		//run test engine
		test_engine<double, B, 1>(A, nullptr, nullptr);

		std::cout << "end rdb_test" << std::endl;
		std::cout << std::endl;
	}

	///Test solvers using the SuiteSparse qc2534 matrix by Bai
	/// https://sparse.tamu.edu/Bai/qc2534.
	///The matrix comes from a quantum chemistry model of H2+ in an electromagnetic field,
	///is Hermitian, with a condition number of ~10^5.
	///The matrix is 2534x2534 and relatively dense, with 463,360 nonzero entries.
	///We test the solver speed with different Block sizes, B = 1, 2, 7, and 14.
	///solArray dimension D is set to one.
	///
	///NOTE: SOR preconditioned GMRES fails with a high SOR damping parameter w
	/// (ez::sor_pGMRES has w = 1.6 by default).
	///We also test ez::sor_pGMRES_robust, which has a lower SOR damping paramter w = 0.6.
	///This solves the problem for block sizes B = 2,7, and 14,
	/// also demonstrating the utility of blocking.
	template<int B>
	void qc_test(bool testNoPre = false)
	{
		std::string qc_mtxMat = "matrix_location/qc2534.mtx";

		//get .mtx matrix and convert to BlockSp
		auto A{ mtxMatrix_to_BlockSp<std::complex<double>, B>(qc_mtxMat, 14, true) }; //matrix size starts on line 14 of .mtx file
		A.shrink_to_fit();
		std::cout << "qc_test with Blocksize B = " << B << std::endl;
		std::cout << "Size of BlockSp matrix: " << A.m() << ' ' << A.n() << ' ' << A.nz() << std::endl;

		//run test engine
		test_engine<std::complex<double>, B, 1>(A, nullptr, nullptr, false, false, false, true);

		std::cout << "end qc_test" << std::endl;
		std::cout << std::endl;
	}
	//*/

	///Test all dense matrix solvers.
	///Errors should all be zero/machine precision.
	///Note: a fast explict solve is used for small matrices (N <= 3),
	/// lapack is used for larger (N > 3)
	template<typename T, int N>
	void dense_test()
	{
		std::cout << "Begin dense_test" << std::endl;
		std::cout << "T = " << typeid(T).name() << ", N = " << N << std::endl;
		std::cout << std::endl;

		std::uniform_real_distribution<double> rnd(0.0, 1.0);
		std::default_random_engine gen;

		//random matrix A and random normal rhs b
		blitz::TinyMatrix<T, N, N> A;
		blitz::TinyVector <T, N> b;
		for (int i = 0; i < N; ++i)
			for (int j = 0; j < N; ++j)
			{
				if constexpr (is_complex_v<T>)
				{
					A(i, j).real(rnd(gen));
					A(i, j).imag(rnd(gen));
				}
				else A(i, j) = rnd(gen);
			}
		{
			auto v{ util::rand_vec<T>(N) };
			for (int i = 0; i < N; ++i)
				b(i) = v[i];
		}

		//if real, make spd matrix A_spd from A, A_spd = I - d A^*A
		//d = 1e-2
		double d = 1.0e-2;
		blitz::TinyMatrix<T, N, N> A_spd;
		A_spd = dense::identity<T, N>() - d * dense::matMat_transpose(A, A);

		//Use Lapack LU and Cholesky factorization as exact solution.
		blitz::TinyMatrix<T, N, N> ALU{ A }, AChol_spd{ A_spd };
		blitz::TinyVector<lapack_int, N> ipiv;
		lapack::getrf<T>(LAPACK_ROW_MAJOR, N, N, ALU.data(), N, ipiv.data());
		if constexpr (!is_complex_v<T>)
			lapack::potrf<T>(LAPACK_ROW_MAJOR, 'L', N, AChol_spd.data(), N);

		blitz::TinyVector <T, N> x{ b }, x_spd{ b };

		lapack::getrs(LAPACK_ROW_MAJOR, 'N', N, 1, ALU.data(), N, ipiv.data(),
			x.data(), 1);
		if constexpr (!is_complex_v<T>)
			lapack::potrs<T>(LAPACK_ROW_MAJOR, 'L', N, 1, AChol_spd.data(), N,
				x_spd.data(), 1);

		blitz::TinyMatrix <T, N, N> Id{ dense::identity<T, N>() };

		//test solvers against exact solution

		dense::LUFactor<T, N> LU(A);
		dense::NonSymEigenFactor<T, N> NSEF(A);

		blitz::TinyVector<T, N> x_lu, x_nsef;
		LU.invmul(b, x_lu);
		NSEF.invmul(b, x_nsef);

		blitz::TinyMatrix<T, N, N> I_lu, I_nsef;
		LU.invmul(A, I_lu);
		NSEF.invmul(A, I_nsef);

		std::cout << "LUFactor error" << std::endl;
		std::cout << blitz::max(blitz::abs(x - x_lu)) << ' ' <<
			blitz::max(blitz::abs(Id - I_lu)) << std::endl;
		std::cout << std::endl;

		std::cout << "NonSymEigenFactor error" << std::endl;
		std::cout << blitz::max(blitz::abs(x - x_nsef)) << ' ' <<
			blitz::max(blitz::abs(Id - I_nsef)) << std::endl;
		std::cout << std::endl;

		//if T = real, also test SPD solvers
		if constexpr (!is_complex_v<T>)
		{
			dense::CholeskyFactor<T, N> Chol(A_spd);
			dense::SymEigenFactor<T, N> SEF(A_spd);

			blitz::TinyVector<T, N> x_chol, x_sef;
			Chol.invmul(b, x_chol);
			SEF.invmul(b, x_sef);

			blitz::TinyMatrix<T, N, N> I_chol, I_sef;
			Chol.invmul(A_spd, I_chol);
			SEF.invmul(A_spd, I_sef);

			std::cout << "CholeskyFactor error" << std::endl;
			std::cout << blitz::max(blitz::abs(x_spd - x_chol)) << ' ' <<
				blitz::max(blitz::abs(Id - I_chol)) << std::endl;
			std::cout << std::endl;

			std::cout << "SymEigenFactor error" << std::endl;
			std::cout << blitz::max(blitz::abs(x_spd - x_sef)) << ' ' <<
				blitz::max(blitz::abs(Id - I_sef)) << std::endl;
			std::cout << std::endl;
		}

		//2D blitz::Array version
		{
			blitz::Array<T, 2> A_arr(N), A_spd_arr(N);
			A_arr = A;
			A_spd_arr = A_spd;

			blitz::Array<T, 1> x_arr(N), x_spd_arr(N), b_arr(N);
			x_arr = x; x_spd_arr = x_spd; b_arr = b;
			auto Id_arr{ dense::identity_Arr<T>(N) };

			dense::LUFactor_Arr<T> LU_arr(A_arr);
			dense::NonSymEigenFactor_Arr<T> NSEF_arr(A_arr);

			blitz::Array<T, 1> x_lu_arr(N), x_nsef_arr(N);
			x_lu_arr = 0.0;	x_nsef_arr = 0.0;
			LU_arr.invmul(b_arr, x_lu_arr);
			NSEF_arr.invmul(b_arr, x_nsef_arr);

			blitz::Array<T, 2> I_lu_arr(N), I_nsef_arr(N);
			I_lu_arr = 0.0;	I_nsef_arr = 0.0;
			LU_arr.invmul(A_arr, I_lu_arr);
			NSEF_arr.invmul(A_arr, I_nsef_arr);

			std::cout << "LUFactor_Arr error" << std::endl;
			std::cout << blitz::max(blitz::abs(x_arr - x_lu_arr)) << ' ' <<
				blitz::max(blitz::abs(Id_arr - I_lu_arr)) << std::endl;
			std::cout << std::endl;

			std::cout << "NonSymEigenFactor_Arr error" << std::endl;
			std::cout << blitz::max(blitz::abs(x_arr - x_nsef_arr)) << ' ' <<
				blitz::max(blitz::abs(Id_arr - I_nsef_arr)) << std::endl;
			std::cout << std::endl;

			//if T = real, also test SPD solvers
			if constexpr (!is_complex_v<T>)
			{
				dense::CholeskyFactor_Arr<T> Chol_arr(A_spd_arr);
				dense::SymEigenFactor_Arr<T> SEF_arr(A_spd_arr);

				blitz::Array<T, 1> x_chol_arr(N), x_sef_arr(N);
				x_chol_arr = 0.0; x_sef_arr = 0.0;
				Chol_arr.invmul(b_arr, x_chol_arr);
				SEF_arr.invmul(b_arr, x_sef_arr);

				blitz::Array<T, 2> I_chol_arr(N), I_sef_arr(N);
				I_chol_arr = 0.0;  I_sef_arr = 0.0;

				Chol_arr.invmul(A_spd_arr, I_chol_arr);
				SEF_arr.invmul(A_spd_arr, I_sef_arr);

				std::cout << "CholeskyFactor_Arr error" << std::endl;
				std::cout << blitz::max(blitz::abs(x_spd_arr - x_chol_arr)) << ' ' <<
					blitz::max(blitz::abs(Id_arr - I_chol_arr)) << std::endl;
				std::cout << std::endl;

				std::cout << "SymEigenFactor_Arr error" << std::endl;
				std::cout << blitz::max(blitz::abs(x_spd_arr - x_sef_arr)) << ' ' <<
					blitz::max(blitz::abs(Id_arr - I_sef_arr)) << std::endl;
				std::cout << std::endl;
			}
		}

		std::cout << "end dense_test" << std::endl;
		std::cout << std::endl;
	}


	//////////////////////////
	//Mtx file converters
	//////////////////////////

	///Go to a specific line in a file.
	std::ifstream& GotoLine(std::ifstream& file, int num)
	{
		file.seekg(std::ios::beg);
		for (int i = 0; i < num - 1; ++i) {
			file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		}
		return file;
	}

	///Read in a matrix-market matrix file (.mtx) and convert it to a BlockSp matrix.
	///If symmetric, only half is in file, must make entire matrix.
	///If base_zero, the matrix indexing in the .mtx file starts from zero
	template<typename T, int B>
	BlockSp<T, B> mtxMatrix_to_BlockSp(const std::string& mat_loc,
		unsigned int start_line, bool symmetric, bool base_zero, bool printMatSize)
	{
		std::ifstream f(mat_loc, std::ios::in);

		//Matrix market files are formatted in following manner
		//1) Name of matrix / matrix details
		//2) Number of rows, number of columns, number of entries/nonzeros
		//3+) Values of matrix in matrix market format
		//Matrix market format: row index, col index, value
		GotoLine(f, start_line); //user must ensure starting line is matrix size line (2)
		std::string size; std::getline(f, size);
		std::stringstream ss(size);
		int M, N, nnz; ss >> M; ss >> N; ss >> nnz;
		if (printMatSize)
		{
			std::cout << M << ' ' << N << ' ' << nnz;
			if (symmetric) std::cout << " Symmetric";
			std::cout << std::endl;
		}

		//Note the choice of block size B for the BlockSp matrix must divide
		//mtx's number of rows M and number of columns N
		assert(M % B == 0 && N % B == 0);
		int lm = M / B;		//number of rows in BlockSp matrix
		int ln = N / B;		//number of cols in BlockSp matrix
		BlockSp<T, B> A(lm, ln);

		//Read values and fill BlockSp matrix.
		for (int innz = 0; innz < nnz; ++innz)
		{
			std::string entry; std::getline(f, entry);
			int i, j; T val;
			std::stringstream ss2(entry);
			ss2 >> i; ss2 >> j; //write matrix index and values. 
			i -= 1; j -= 1;			//change to base 0 indexing.
			if constexpr (is_complex_v<T>) //also read imaginary value.
			{
				ScalarType_t<T> valReal, valImg; ss2 >> valReal; ss2 >> valImg;
				val.real(valReal); val.imag(valImg);
				util::setFromGlobal(A, i, j, val);
				if (symmetric && i != j) //if symmetric, fill (j,i) with complex conjugate.
					util::setFromGlobal(A, j, i, std::conj(val));
			}
			else
			{
				ss2 >> val;
				util::setFromGlobal(A, i, j, val);
				if (symmetric && i != j) //if symmetric, fill (j,i)
					util::setFromGlobal(A, j, i, val);
			}
		}
		return A;
	}

	///Read in a matrix-market vector file (.mtx) and convert it to a solArray.
	template<typename T, int B, int D>
	solArray<T, B, D> mtxArray_to_solArray(const std::string& vec_loc,
		unsigned int start_line, bool printVecSize)
	{
		std::ifstream f(vec_loc, std::ios::in);

		//matrix market files are formatted in following manner
		//1) name of vector
		//2) number of rows, number of cols (1)
		//3+) values of vector
		GotoLine(f, start_line); //user must ensure starting line is vector size line 
		std::string size; std::getline(f, size);
		std::stringstream ss(size);
		int M, N; ss >> M; ss >> N;
		if (printVecSize) std::cout << M << ' ' << N << std::endl;

		//Note the choice of block size B and dimension D for the solArray must divide
		//mtx's number of rows M, N must = 1
		int constexpr BD = B * D;
		assert(M % BD == 0 && N == 1);
		int l = M / BD; //number of rows in BlockSp matrix
		solArray<T, B, D> x(l);

		//convert mtx array index to sol index and fill it's value
		auto fill_solArray = [&](int i, T val)
			{
				int im = i / BD;			//find the index of the solArray
				int i_BD = i % B;			//find the index within the local blocks
				int dim = i_BD / B;
				int ib = i_BD % B;
				x(im, dim)(ib) = val; //fill value
			};

		for (int im = 0; im < M; ++im)
		{
			std::string entry; std::getline(f, entry);
			std::stringstream ss2(entry);
			T val;
			if constexpr (is_complex_v<T>) //also read imaginary value.
			{
				long double valReal, valImg; ss2 >> valReal; ss2 >> valImg;
				val.real(valReal); val.imag(valImg);
			}
			else ss2 >> val;
			fill_solArray(im, val);
		}
		return x;
	}

}//end namespace tests

#endif