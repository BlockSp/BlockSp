#ifndef BLOCKSP_EIGEN_COMPARE_HPP
#define BLOCKSP_EIGEN_COMPARE_HPP

#include <chrono>
#include <complex>
#include <fstream>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "blocksp_include.hpp"
#include "blocksp_tests.hpp"

///blocksp_eigen_compare.hpp compares the speed of
/// Eigen's and BlockSp's iterative sparse linear system solvers.
/// 
///Included in this file are
///i)  A test engine for various sparse linear systems, comparing 
///		 Eigen's BICGSTAB and CG to BlockSp's pGMRES and pCG.
/// 
///ii) A collection of tests, these include:
///		1) A random symmetric block matrix outlined similarly to a heat operator (BlockSp's test problem).
///		2) A straightforward 2D 2nd order finite difference Poisson problem (Eigen's test problem).
///		3) Matrices from the SparseSuite library
///			3a) rdb5000 - a real non-symmetric matrix from a reaction-diffusion-Brusselator model.
///			3b) qc2534  - a complex Hermitian matrix from quantum chemistry.
///			3c) saylr4  - a real non-symmetric matrix from a 3D simulation of oil reservoirs.
/// 
///iii) Matrix file converters, convert from .mtx files to Eigen triplets and vectors.
///			Convert from Eigen triplets and vectors to BlockSp matrices and solArrays.
namespace eigen_comparison
{

	////////////////////////////////////////
	//Forward declare conversion functions

	///Eigen Triplet with matrix size information M and N.
	///For reading in from mtx files.
	template<typename T>
	struct Eigen_Trip
	{
		std::vector<Eigen::Triplet<T>> m_trip;
		int M; int N;
		Eigen_Trip() : M{ 0 }, N{ 0 } {};
		Eigen_Trip(const std::vector<Eigen::Triplet<T>>& trip, int M, int N) :
			m_trip{ trip }, M{ M }, N{ N } {
		}
		Eigen_Trip(int M, int N, const std::vector<Eigen::Triplet<T>>& trip) :
			m_trip{ trip }, M{ M }, N{ N } {
		}
		void push_back(const Eigen::Triplet<T>& trip) { m_trip.push_back(trip); }
	};

	///Build a BlockSp matrix from an Eigen_Trip of type T.
	template<typename T, int B>
	BlockSp::BlockSp<T, B> BlockSp_from_Eigen_Trip(const Eigen_Trip<T>& trip);

	///Build a solArray from an Eigen::Vector of type T.
	template<typename T, int B, int D = 1>
	BlockSp::solArray<T, B, D> solArray_from_EigenVector(const Eigen::Vector<T, Eigen::Dynamic>& vec);

	///Read in a matrix-market matrix file (.mtx) and convert it to an Eigen_Trip.
	template<typename T>
	Eigen_Trip<T> mtxMatrix_to_Eigen_Trip(const std::string& mat_loc, unsigned int start_line,
		bool symmetric = false, bool base_zero = false, bool printMatSize = false);

	///Read in an mtx matrix and make a row major Eigen::SparseMatrix
	template<typename T>
	Eigen::SparseMatrix<T, Eigen::RowMajor> mtxMatrix_to_EigenSpMatrix(
		const std::string& mat_loc, unsigned int start_line, bool symmetric = false,
		bool base_zero = false, bool printMatSize = false);

	///Read in a matrix-market vector file (.mtx) and convert it to an Eigen::Vector.
	template<typename T>
	Eigen::Vector<T, Eigen::Dynamic> mtxArray_to_EigenVector(const std::string& vec_loc,
		unsigned int start_line, bool printVecSize = false);


	////////////////////////////////////////
	//Comparison Engine
	////////////////////////////////////////

	///Engine to compare the timing of Eigen's sparse matrix iterative solvers
	/// with BlockSp's iterative solvers.
	/// 
	///Reads in an Eigen_Trip<T> A_trip that constructs sparse matrix A, and
	/// a pointer to an Eigen::Vector for the exact solution x or right-hand-side b. 
	///Then solves Ax = b. 
	/// 
	///The triplet A_Trip and vectors x/b may be contructed from matrix market files .mtx
	/// via the mtx file converters in this file. From these, the BlockSp structures are created.
	///
	///If x and b are provided. Ax = b is solved and compared to the exact solution. 
	///If only x is provided, b = Ax is calculated, then used as input for the solvers.
	///If only b is provided, Ax = b is first solved by Eigen's SparseLU direct solver,
	/// this is used as a reference solution. Note: direct solvers may not be suitable for large matrices.
	///If neither x nor b is provided, a random normal solution vector x is calculated
	///	and rhs b generated.
	/// 
	///Tests 
	///Compares Eigen's BICGSTAB with diagonal and IncompleteLUT preconditioners
	///to BlockSp's pGMRES with block SOR and ILU0 preconditioners.
	/// 
	///If symmetric positive definite (SPD), we also test both libraries' 
	/// preconditioneed conjugate gradient (pCG).
	///For Eigen, we test diagonal and IncompleteCholesky preconditioners.
	///For BlockSp, we test symmetric SOR and IChol0 preconditioners.
	/// 
	///If noPre, test solvers with no preconditioner.
	///If diag, test BlockSp solvers with a block diagonal preconditioner.
	///If robust, test BlockSp's ez::sor_pGMRES_robust.
	///If dr, run deflated restart versions for BlockSp's pGMRES. 
	template<typename T, int B, int D = 1>
	void eigen_compare(const Eigen_Trip<T>& A_trip, const Eigen::Vector<T, Eigen::Dynamic>* x_ptr,
		const Eigen::Vector<T, Eigen::Dynamic>* b_ptr, bool SPD = false, bool noPre = false,
		bool diag = false, bool robust = false, bool dr = true)
	{
		std::cout << "Start eigen_compare" << std::endl;
		int m = A_trip.M; int n = A_trip.N; 
		std::cout << "Matrix size: " << m << "x" << n << ", non-zeros: "
			<< A_trip.m_trip.size() << std::endl;

		//timing lambda
		std::chrono::time_point<std::chrono::steady_clock> start;
		auto time_since = [](auto start)
			{ return  std::chrono::duration<double>(std::chrono::steady_clock::now() - start); };

		//Assemble Eigen::SparseMatrix A_eig
		Eigen::SparseMatrix<T, Eigen::RowMajor> A_eig(m, n);
		A_eig.setFromTriplets(A_trip.m_trip.begin(), A_trip.m_trip.end());

		//Assemble Eigen::vectors x_eig and b_eig
		Eigen::Vector<T, Eigen::Dynamic> x_eig(n);
		Eigen::Vector<T, Eigen::Dynamic> b_eig(m);
		if (x_ptr && b_ptr) //both solution x and rhs b provided
		{
			assert(x_ptr->rows() == n);
			assert(b_ptr->rows() == m);
			x_eig = *x_ptr;
			b_eig = *b_ptr;
		}
		else if (x_ptr)					//solution x provided
		{
			assert(x_ptr->rows() == n);
			x_eig = *x_ptr;					//provided exact solution
			b_eig = A_eig * b_eig;	//b = Ax
		}
		else if (b_ptr)			//rhs b provided
		{
			assert(b_ptr->rows() == m);
			b_eig = *b_ptr;

			//Use Eigen SparseLU solver as reference solution
			std::cout << "Direct LU solve as reference solution -- direct solves may take some time." << std::endl;
			start = std::chrono::steady_clock::now();
			Eigen::SparseLU<Eigen::SparseMatrix<T>, Eigen::COLAMDOrdering<int>> eig_LU;
			eig_LU.compute(A_eig);
			x_eig = eig_LU.solve(b_eig);
			std::cout << "Eigen LU time              " << time_since(start) << std::endl;
			std::cout << std::endl;
		}
		else //neither x nor b provided, use random normal solution
		{
			auto x_rand{ BlockSp::util::rand_vec<T>(n) };
			for (int i = 0; i < n; ++i)
				x_eig(i) = x_rand[i];
			b_eig = A_eig * x_eig;
		}

		//Assemble BlockSp matrix A_bsp with block size B*D
		//and assemble solution and rhs solArrays of dimension D with block size B.
		auto A_bsp{ BlockSp_from_Eigen_Trip<T, B * D>(A_trip) };
		auto x_sol_bsp{ solArray_from_EigenVector<T, B, D>(x_eig) };
		auto b_bsp{ solArray_from_EigenVector<T, B, D>(b_eig) };
		A_bsp.shrink_to_fit(); //Once a BlockSp matrix is assembled, 
													 //use this function to shrink its vectors' storage.

		std::cout << "BlockSp, B = " << B << ", D = " << D << ", matrix size: " << A_bsp.m() << "x" <<
			A_bsp.n() << ", blocksize: " << B * D << "x" << B * D << "," << std::endl;
			std::cout << "nonzero blocks " << A_bsp.nz() << ", total nnz "
			<< A_bsp.nz() * B * B * D * D << ", overfill " <<
			double(A_bsp.nz() * B * B * D * D) / A_trip.m_trip.size() << std::endl;

		//Tolerance for solver convergence. Different for double or float.
		//These are the default tolerances used in BlockSp.
		BlockSp::ScalarType_t<T> tol;
		tol = std::is_same<double, BlockSp::ScalarType_t<T>>::value ? 1.0e-8 : 1.0e-5;

		/////////////////////////
		//Eigen's solvers

		std::cout << std::endl;
		std::cout << "//////////////////////////////////////" << std::endl;
		std::cout << "EIGEN" << std::endl;
		std::cout << "//////////////////////////////////////" << std::endl;
		std::cout << std::endl;

		//Eigen's BICGSTAB, no preconditioner
		if (noPre)
		{
			start = std::chrono::steady_clock::now();
			Eigen::BiCGSTAB<Eigen::SparseMatrix<T, Eigen::RowMajor>,
				Eigen::IdentityPreconditioner> BICG;
			BICG.setTolerance(tol);
			BICG.compute(A_eig);
			Eigen::Vector<T, Eigen::Dynamic> x_BICG(m);
			x_BICG = BICG.solve(b_eig);
			std::cout << "BICGSTAB #iterations:      " << BICG.iterations() << std::endl;
			std::cout << "Eigen no pre BICGSTAB time " << time_since(start) << std::endl;
			std::cout << "error:                     " << (x_BICG - x_eig).template lpNorm<Eigen::Infinity>() << std::endl;
			std::cout << std::endl;
		}

		//Eigen's BICGSTAB, Diag preconditioner
		{
			start = std::chrono::steady_clock::now();
			Eigen::BiCGSTAB<Eigen::SparseMatrix<T, Eigen::RowMajor>,
				Eigen::DiagonalPreconditioner<T>> BICG;
			BICG.setTolerance(tol);
			BICG.compute(A_eig);
			Eigen::Vector<T, Eigen::Dynamic> x_BICG(m);
			x_BICG = BICG.solve(b_eig);
			std::cout << "BICGSTAB #iterations:      " << BICG.iterations() << std::endl;
			std::cout << "Eigen Diag BICGSTAB time   " << time_since(start) << std::endl;
			std::cout << "error:                     " << (x_BICG - x_eig).template lpNorm<Eigen::Infinity>() << std::endl;
			std::cout << std::endl;
		}
		//*/

		//Eigen's BICGSTAB, ILUT preconditioner
		{
			start = std::chrono::steady_clock::now();
			Eigen::BiCGSTAB<Eigen::SparseMatrix<T, Eigen::RowMajor>,
				Eigen::IncompleteLUT<T>> BICG;
			BICG.setTolerance(tol);
			BICG.compute(A_eig);
			Eigen::Vector<T, Eigen::Dynamic> x_BICG(m);
			x_BICG = BICG.solve(b_eig);
			std::cout << "BICGSTAB #iterations:      " << BICG.iterations() << std::endl;
			std::cout << "Eigen ILUT BICGSTAB time   " << time_since(start) << std::endl;
			std::cout << "error:                     " << (x_BICG - x_eig).template lpNorm<Eigen::Infinity>() << std::endl;
			std::cout << std::endl;
		}

		///If SPD --Symmetric Positive Definite-- also test Conjugate gradient (CG).
		if constexpr (!BlockSp::is_complex_v<T>)
			if (SPD)
			{
				//Eigen's CG, no preconditioner
				if (noPre)
				{
					start = std::chrono::steady_clock::now();
					Eigen::ConjugateGradient<Eigen::SparseMatrix<T>, Eigen::Lower,
						Eigen::IdentityPreconditioner> cg;
					cg.setTolerance(tol);
					cg.compute(A_eig);
					Eigen::Vector<T, Eigen::Dynamic> x_CG(m);
					x_CG = cg.solve(b_eig);
					std::cout << "CG #iterations:            " << cg.iterations() << std::endl;
					std::cout << "Eigen no pre CG time       " << time_since(start) << std::endl;
					std::cout << "error:                     " << (x_CG - x_eig).template lpNorm<Eigen::Infinity>() << std::endl;
					std::cout << std::endl;
				}

				//Eigen's CG, Diagonal preconditioner
				{
					start = std::chrono::steady_clock::now();
					Eigen::ConjugateGradient<Eigen::SparseMatrix<T>, Eigen::Lower,
						Eigen::DiagonalPreconditioner<T>> cg;
					cg.setTolerance(tol);
					cg.compute(A_eig);
					Eigen::Vector<T, Eigen::Dynamic> x_CG(m);
					x_CG = cg.solve(b_eig);
					std::cout << "CG #iterations:            " << cg.iterations() << std::endl;
					std::cout << "Eigen Diag pCG time        " << time_since(start) << std::endl;
					std::cout << "error:                     " << (x_CG - x_eig).template lpNorm<Eigen::Infinity>() << std::endl;
					std::cout << std::endl;
				}

				//Eigen's CG, IChol preconditioner
				{
					start = std::chrono::steady_clock::now();
					Eigen::ConjugateGradient<Eigen::SparseMatrix<T>, Eigen::Lower,
						Eigen::IncompleteCholesky<T>> cg;
					cg.setTolerance(tol);
					cg.compute(A_eig);
					Eigen::Vector<T, Eigen::Dynamic> x_CG(m);
					x_CG = cg.solve(b_eig);
					std::cout << "CG #iterations:            " << cg.iterations() << std::endl;
					std::cout << "Eigen IChol pCG time       " << time_since(start) << std::endl;
					std::cout << "error:                     " << (x_CG - x_eig).template lpNorm<Eigen::Infinity>() << std::endl;
					std::cout << std::endl;
				}
			}
		//*/


		/////////////////////////////////
		//Now test BlockSp

		std::cout << std::endl;
		std::cout << "//////////////////////////////////////" << std::endl;
		std::cout << "BlockSp, B = " << B << ", D = " << D << std::endl;
		std::cout << "//////////////////////////////////////" << std::endl;
		std::cout << std::endl;

		if (noPre)
		{
			//pGMRES, no preconditioner, no restart.
			{
				start = std::chrono::steady_clock::now();
				auto x_bsp{ BlockSp::pGMRES<T, B, D>(A_bsp, b_bsp, tol, [](const auto& x) {}, 2000, -1, -1, true) };
				std::cout << "BlockSp no pre no re GMRES time  " << time_since(start) << std::endl;
				std::cout << "error                            " << BlockSp::infNorm(x_bsp - x_sol_bsp) << std::endl;
				std::cout << std::endl;
			}

			//pGMRES_dr, no preconditioner, with deflated_restart.
			if (dr)
			{
				start = std::chrono::steady_clock::now();
				auto x_bsp{ BlockSp::pGMRES<T, B, D>(A_bsp, b_bsp, tol, [](const auto& x) {}, 2000, 30, 10, true) };
				std::cout << "BlockSp no pre GMRES_dr time     " << time_since(start) << std::endl;
				std::cout << "error                            " << BlockSp::infNorm(x_bsp - x_sol_bsp) << std::endl;
				std::cout << std::endl;
			}
		}
		//*/

		//block diagonal precondtioner
		if (diag)
		{
			//pGMRES with digaonal preconditioner, no restart.
			{
				start = std::chrono::steady_clock::now();
				BlockSp::preConditioners::Diag<BlockSp::dense::LUFactor<T, B* D>, T, B, D> diag(A_bsp);
				auto x_bsp{ BlockSp::pGMRES<T, B, D>(A_bsp, b_bsp, tol, diag, 5000, -1, -1, true) };
				std::cout << "BlockSp diag pGMRES time         " << time_since(start) << std::endl;
				std::cout << "error                            " << BlockSp::infNorm(x_bsp - x_sol_bsp) << std::endl;
				std::cout << std::endl;
			}

			//pGMRES with digaonal preconditioner, with deflated_restart.
			if (dr)
			{
				start = std::chrono::steady_clock::now();
				BlockSp::preConditioners::Diag<BlockSp::dense::LUFactor<T, B* D>, T, B, D> diag(A_bsp);
				auto x_bsp{ BlockSp::pGMRES<T, B, D>(A_bsp, b_bsp, tol, diag, 5000, 30, 10, true) };
				std::cout << "BlockSp diag pGMRES_dr time      " << time_since(start) << std::endl;
				std::cout << "error                            " << BlockSp::infNorm(x_bsp - x_sol_bsp) << std::endl;
				std::cout << std::endl;
			}
		}

		//BlockSp's ez::sor_pGMRES
		//pGMRES, SOR preconditioner with LU inversion, no restart
		{
			start = std::chrono::steady_clock::now();
			auto x_bsp{ BlockSp::ez::sor_pGMRES(A_bsp, b_bsp, true) };
			std::cout << "BlockSp SOR pGMRES time          " << time_since(start) << std::endl;
			std::cout << "error                            " << BlockSp::infNorm(x_bsp - x_sol_bsp) << std::endl;
			std::cout << std::endl;
		}

		//BlockSp's ez::sor_pGMRES_dr
		//pGMRES, SOR preconditioner with LU inversion, with deflated_restart
		if (dr)
		{
			start = std::chrono::steady_clock::now();
			auto x_bsp{ BlockSp::ez::sor_pGMRES_dr(A_bsp, b_bsp, true) };
			std::cout << "BlockSp SOR pGMRES_dr time       " << time_since(start) << std::endl;
			std::cout << "error                            " << BlockSp::infNorm(x_bsp - x_sol_bsp) << std::endl;
			std::cout << std::endl;
		}

		if (robust)
		{
			//BlockSp's ez::sor_pGMRES_robust
			//pGMRES, light damping SOR preconditioning with LU inversion, no restart
			start = std::chrono::steady_clock::now();
			auto x_bsp{ BlockSp::ez::sor_pGMRES_robust(A_bsp, b_bsp, true) };
			std::cout << "BlockSp SOR pGMRES_robust time   " << time_since(start) << std::endl;
			std::cout << "error                            " << BlockSp::infNorm(x_bsp - x_sol_bsp) << std::endl;
			std::cout << std::endl;
		}

		//BlockSp's ez::ilu0_pGMRES
		//pGMRES, ilu0 preconditioner. Not restarted.
		{
			start = std::chrono::steady_clock::now();
			auto x_bsp{ BlockSp::ez::ilu0_pGMRES<T, B, D>(A_bsp, b_bsp, true) };
			std::cout << "BlockSp ILU0 pGMRES time         " << time_since(start) << std::endl;
			std::cout << "error                            " << BlockSp::infNorm(x_bsp - x_sol_bsp) << std::endl;
			std::cout << std::endl;
		}

		//BlockSp's ez::ilu0_pGMRES_dr
		//pGMRES, ilu0 preconditioner, with deflated_restart.
		if (dr)
		{
			start = std::chrono::steady_clock::now();
			auto x_bsp{ BlockSp::ez::ilu0_pGMRES_dr(A_bsp, b_bsp, true) };
			std::cout << "BlockSp ILU0 pGMRES_dr time      " << time_since(start) << std::endl;
			std::cout << "error                            " << BlockSp::infNorm(x_bsp - x_sol_bsp) << std::endl;
			std::cout << std::endl;
		}

		//If SPD --Symmetric Positive Definite-- also test Conjugate Gradient (CG).
		if constexpr (!BlockSp::is_complex_v<T>)
			if (SPD)
			{
				//BlockSp CG, no preconditioner
				if (noPre)
				{
					start = std::chrono::steady_clock::now();
					auto x_bsp{ BlockSp::pCG<T, B, D>(A_bsp, b_bsp, tol, [](const auto&) {}, 2000, true) };
					std::cout << "BlockSp no Pre CG time           " << time_since(start) << std::endl;
					std::cout << "error                            " << BlockSp::infNorm(x_bsp - x_sol_bsp) << std::endl;
					std::cout << std::endl;
				}

				//pCG with digaonal preconditioner.
				if (diag)
				{
					start = std::chrono::steady_clock::now();
					BlockSp::preConditioners::Diag<BlockSp::dense::CholeskyFactor<T, B* D>, T, B, D> diag(A_bsp);
					auto x_bsp{ BlockSp::pCG<T, B, D>(A_bsp, b_bsp, tol, diag, 5000, true) };
					std::cout << "BlockSp diag pCG time            " << time_since(start) << std::endl;
					std::cout << "error                            " << BlockSp::infNorm(x_bsp - x_sol_bsp) << std::endl;
					std::cout << std::endl;
				}

				//pCG, SOR preconditioner with Cholesky inversion.
				{
					start = std::chrono::steady_clock::now();
					auto x_bsp{ BlockSp::ez::sor_pCG<T, B, D>(A_bsp, b_bsp, true) };
					std::cout << "BlockSp SOR pCG time             " << time_since(start) << std::endl;
					std::cout << "error                            " << BlockSp::infNorm(x_bsp - x_sol_bsp) << std::endl;
					std::cout << std::endl;
				}

				//pCG, IChol0 preconditioner.
				{
					start = std::chrono::steady_clock::now();
					auto x_bsp{ BlockSp::ez::ichol0_pCG<T, B, D>(A_bsp, b_bsp, true) };
					std::cout << "BlockSp IChol0 pCG time          " << time_since(start) << std::endl;
					std::cout << "error                            " << BlockSp::infNorm(x_bsp - x_sol_bsp) << std::endl;
					std::cout << std::endl;
				}

			}

		std::cout << "end eigen_compare" << std::endl;
	}


	////////////////////////////////////
	//Comparison tests
	////////////////////////////////////

	///Compare the performance of BlockSp to Eigen
	///Solve a random symmetric BlockSp matrix with a random solArray(a multi-dimensional vector).
	///The matrix is outlined similarly to a heat operator: ASym = (I - d A^T A).
	///Can use data type T = float, double, std::complex<float>, or std::complex<double>. 
	template<typename T>
	void block_pseudo_heat()
	{
		//Try changing D, B, l, nCol, and delta. 
		int constexpr D = 2;				//vector Dimension //D and B are compile time constants!
		int constexpr B = 16;				//vector Blocksize
		int constexpr BD = B * D;
		using S = BlockSp::ScalarType_t<T>;  //float or double, derived from T.
		int l = 256;								//vector/BlockSp matrix length
		int eig_l = l * B * D;

		//make random BlockSp matrix A and symmetric positive definite matrix ASym = I - delta*A^T*A
		int nCol = 4;			 //num of nonzero columns's per row of A. ASym will have ~2*nCol nonzero columns per row.
		BlockSp::BlockSp<T, BD> A{ BlockSp::util::rand_BlockSp<T, BD>(l, l, nCol) };
		S delta = 2.0e-4; //the value of delta may be tuned to stress test the solvers.
		auto ASym{ BlockSp::bs_id<T, BD>(l) - delta * BlockSp::bsmm(A, A, true) }; //ASym = I - delta*A^T*A
		std::cout << "block_pseudo_heat: data type " << typeid(T).name() << std::endl;

		//convert BlockSp to Eigen_Trip
		Eigen_Trip<T> trip;
		trip.M = eig_l; trip.N = eig_l;

		//Convert BlockSp index to triplet index and fill it's value
		auto fill_trip = [&](int i, int j, const blitz::TinyMatrix<T, BD, BD>& blk)
			{
				int i_start = i * BD; //starting index for triplets
				int j_start = j * BD;
				for (int i_blk = 0; i_blk < BD; ++i_blk)
					for (int j_blk = 0; j_blk < BD; ++j_blk)
					{
						int i_trip = i_start + i_blk;
						int j_trip = j_start + j_blk;
						trip.push_back(Eigen::Triplet<T>(i_trip, j_trip, blk(i_blk, j_blk)));
					}
			};
		for (int im = 0; im < l; ++im)
			for (const auto& icol : ASym.colInd(im))
				fill_trip(im, icol, ASym(im, icol));

		//compare Eigen and BlockSp, random normal solution
		eigen_compare<T, B, D>(trip, nullptr, nullptr, true, true, true, false, false);
		std::cout << "end block_pseudo_heat" << std::endl;
		std::cout << std::endl;
	}

	///Basic Eigen test, i.e., does it compile?
	void eigenTestSetUp()
	{
		Eigen::MatrixXd m(2, 2);
		m(0, 0) = 3;
		m(1, 0) = 2.5;
		m(0, 1) = -1;
		m(1, 1) = m(1, 0) + m(0, 1);
		std::cout << m << std::endl;
	}

	///Eigen sparse matrix test.
	///Includes the functions insertCoefficient and buildProblem. 
	///Builds a straightforward sparse 2D Poisson finite difference problem.
	///Resulting matrix is symmetric positive definite (SPD).
	void insertCoefficient(int id, int i, int j, double w, std::vector<Eigen::Triplet<double>>& coeffs,
		Eigen::VectorXd& b, const Eigen::VectorXd& boundary)
	{
		int n = int(boundary.size());
		int id1 = i + j * n;

		if (i == -1 || i == n)
			b(id) -= w * boundary(j);  // constrained coefficient
		else if (j == -1 || j == n)
			b(id) -= w * boundary(i);  // constrained coefficient
		else
			coeffs.push_back(Eigen::Triplet<double>(id, id1, w));  // unknown coefficient
	}
	void buildProblem(std::vector<Eigen::Triplet<double>>& coefficients, Eigen::VectorXd& b, int n)
	{
		b.setZero();
		Eigen::ArrayXd boundary = Eigen::ArrayXd::LinSpaced(n, 0, EIGEN_PI).sin().pow(2);
		for (int j = 0; j < n; ++j) {
			for (int i = 0; i < n; ++i) {
				int id = i + j * n;
				insertCoefficient(id, i - 1, j, -1, coefficients, b, boundary);
				insertCoefficient(id, i + 1, j, -1, coefficients, b, boundary);
				insertCoefficient(id, i, j - 1, -1, coefficients, b, boundary);
				insertCoefficient(id, i, j + 1, -1, coefficients, b, boundary);
				insertCoefficient(id, i, j, 4, coefficients, b, boundary);
			}
		}
	}

	///Eigen's sparse matrix example:
	///https://libeigen.gitlab.io/eigen/docs-nightly/group__TutorialSparse.html
	void eigenSparseExample()
	{
		int n = 5;			// size of the image
		int m = n * n;  // number of unknowns (=number of pixels)

		// Assembly:
		std::vector<Eigen::Triplet<double>> coefficients;  // list of non-zeros coefficients
		Eigen::VectorXd b(m);         // the right-hand-side vector resulting from the constraints
		buildProblem(coefficients, b, n);

		for (const auto& c : coefficients)
			std::cout << c.row() << ' ' << c.col() << ' ' << c.value() << std::endl;
		std::cout << std::endl;
		std::cout << b << std::endl;
		std::cout << std::endl;

		Eigen::SparseMatrix<double, Eigen::RowMajor> A(m, m);
		A.setFromTriplets(coefficients.begin(), coefficients.end());

		// Solving:
		Eigen::SimplicialCholesky<Eigen::SparseMatrix<double, Eigen::RowMajor>> chol(A);  // performs a Cholesky factorization of A
		Eigen::VectorXd x = chol.solve(b);            // use the factorization to solve for the given right-hand-side

		std::cout << A << std::endl;
		std::cout << std::endl;
		std::cout << x << std::endl;
	}

	///Compare the performance of BlockSp to Eigen,
	///using Eigen's intro sparse matrix test problem.
	///Matrix is from a 2nd order 2D Poisson problem and is symmetric-positive definite.
	///https://libeigen.gitlab.io/eigen/docs-nightly/group__TutorialSparse.html
	template<int B = 2, int D = 1>
	void eigenSparseExample_compare()
	{
		int n = 300;    // size of the image
		int m = n * n;  // number of unknowns (=number of pixels)

		std::cout << "eigenSparseExample_compare" << std::endl;

		// Eigen Assembly:
		std::vector<Eigen::Triplet<double>> coefficients;  // list of non-zeros coefficients
		Eigen::VectorXd b_eig(m);         // the right-hand-side vector resulting from the constraints
		buildProblem(coefficients, b_eig, n);

		///Compare Eigen and BlockSp
		eigen_compare<double, B, D>(Eigen_Trip(coefficients, m, m), nullptr, &b_eig, true);
		std::cout << "end eigenSparseExample_compare" << std::endl;
		std::cout << std::endl;
	}

	/////////////////////////////
	//SparseSuite tests

	//Note: you must first download the test matrix and set the
	//matrix location to the correct path.

	
	///LDG_2D_vecLap_test - compare Eigen and BlockSp with a linear system originating
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

		//make Eigen_Triplet from mtx file
		std::string mtx_A = "test_mats/LDG_2D_vecLap.mtx";
		auto A_trip{ mtxMatrix_to_Eigen_Trip<double>(mtx_A, 1, true) };

		std::string mtx_x = "test_mats/LDG_2D_vecLap_sol.mtx";
		auto x{ mtxArray_to_EigenVector<double>(mtx_x, 1) };

		std::string mtx_b = "test_mats/LDG_2D_vecLap_b.mtx";
		auto b{ mtxArray_to_EigenVector<double>(mtx_b, 1) };

		//Compare Eigen and BlockSp, random normal solution
		eigen_compare<double, B, D>(A_trip, &x, &b, true);
		std::cout << "end LDG_2D_vecLap_test" << std::endl;
		std::cout << std::endl;
	}

	/*
	///Compare Eigen and BlockSp for SparseSuite rdb5000 matrix.
	///The matrix comes from a reaction-diffusion-Brusselator model,
	///is non-symmetric indefinite, with a condition number of ~10^3.
	///The matrix is 5000x5000 with 29,600 nonzero entries.
	///https://sparse.tamu.edu/Bai/rdb5000
	template<int B, int D = 1>
	void rdb_test()
	{
		std::cout << "/////////////////////////////////////" << std::endl;
		std::cout << "rdb_test B = " << B << ", D = " << D << std::endl;

		//make Eigen_Triplet from mtx file
		std::string mtxMat = "/matrix_location/rdb5000.mtx";
		auto trip{ mtxMatrix_to_Eigen_Trip<double>(mtxMat, 14, false) };

		//Compare Eigen and BlockSp, random normal solution
		eigen_compare<double, B, D>(trip, nullptr, nullptr, false, true);
		std::cout << "end rdb_test" << std::endl;
		std::cout << std::endl;
	}

	///Compare Eigen and BlockSp for SparseSuite qc2534 matrix.
	///The matrix comes from a quantum chemistry model of H2+ in an electromagnetic field,
	///is Hermitian, with a condition number of ~10^5.
	///The matrix is 2534x2534 and relatively dense, with 463,360 nonzero entries.
	///Available block sizes are B = 1, 2, 7, and 14.
	///NOTE: For BlockSp, ez::sor_pGMRES (SOR pGMRES) fails due to its high SOR damping parameter,
	///We also test ez::sor_pGMRES_robust, which is SOR pGMRES with a lower parameter (0.6 vs 1.6).
	///Non preconditioned GMRES succeeds for each B, but requires many more iterations.
	///https://sparse.tamu.edu/Bai/qc2534.
	template<int B, int D = 1>
	void qc_test(bool testNoPre = false)
	{
		std::cout << "/////////////////////////////////////" << std::endl;
		std::cout << "qc_test B = " << B << ", D = " << D << std::endl;

		//make Eigen_Triplet from mtx file
		std::string mtxMat = "/matrix_location/qc2534.mtx";
		auto trip{ mtxMatrix_to_Eigen_Trip<std::complex<double>>(mtxMat, 14, true) };

		//Compare Eigen and BlockSp, random normal solution
		eigen_compare<std::complex<double>, B, D>(trip, nullptr, nullptr, false, true, false, true);
		std::cout << "end qc_test" << std::endl;
		std::cout << std::endl;
	}

	///Compare Eigen and BlockSp for Harwell-Boeing saylr4 matrix.
	///The matrix comes from a 3D simulation of oil reservoirs,
	///is real nonsymmetric, with a condition number of ~10^2.
	///The matrix is 3564x3564 with 22316 non-zero entries.
	///https://sparse.tamu.edu/HB/saylr4
	///Note: this test problem was used in Morgan's deflated restart paper,
	/// and the restarted pGMRES should run faster than the non-restarted version.
	template<int B, int D = 1>
	void saylr_test()
	{
		std::cout << "/////////////////////////////////////" << std::endl;
		std::cout << "saylr_test B = " << B << ", D = " << D << std::endl;

		//make Eigen_Triplet from mtx file
		std::string mtxMat = "/matrix_location/saylr4.mtx";
		auto trip{ mtxMatrix_to_Eigen_Trip<double>(mtxMat, 2, false) };

		//Compare Eigen and BlockSp, random normal solution
		eigen_compare<double, B, D>(trip, nullptr, nullptr);
		std::cout << "end saylr_test" << std::endl;
		std::cout << std::endl;
	}
	//*/


	///////////////////////////////
	//Matrix converters
	///////////////////////////////

	//This section includes several conversion functions for the test matrices.
	//1) Eigen_Trip - a struct for Eigen Triplets.
	//2) Convert from Eigen structures to BlockSp/solArray.
	//3) Convert .mtx to EigenTrip, Eigen::SparseMatrix and Eigen::Vector.

	////////////////////////////////////////
	//Build BlockSp and solArray from Eigen.

	///Build a BlockSp matrix from an Eigen_Trip of type T.
	template<typename T, int B>
	BlockSp::BlockSp<T, B> BlockSp_from_Eigen_Trip(const Eigen_Trip<T>& trip)
	{
		int m = trip.M; int n = trip.N; //the size of Eigen SparseMatrix
		//Note the choice of block size B for the BlockSp matrix must divide
		//the number of rows M and number of columns N
		assert((m % B) == 0 && (n % B) == 0);
		int lm = m / B;		//number of rows in BlockSp matrix
		int ln = n / B;		//number of cols in BlockSp matrix
		BlockSp::BlockSp<T, B> A(lm, ln);

		//Fill BlockSp
		for (const auto& tr : trip.m_trip)
			BlockSp::util::setFromGlobal(A, tr.row(), tr.col(), tr.value());
		return A;
	}

	///Build a solArray from an Eigen::Vector of type T.
	template<typename T, int B, int D>
	BlockSp::solArray<T, B, D> solArray_from_EigenVector(const Eigen::Vector<T, Eigen::Dynamic>& vec)
	{
		//Note the choice of block size B and dimension D for the solArray must divide
		//the length of the Eigen::Vector
		int v_len = vec.rows();
		int constexpr BD = B * D;
		assert(v_len % BD == 0);
		int l = v_len / BD; //number of rows in BlockSp matrix
		BlockSp::solArray<T, B, D> x(l);

		//Fill solArray
		for (int i = 0; i < v_len; ++i)
			BlockSp::util::setFromGlobal(x, i, vec(i));
		return x;
	}

	/////////////////////////////
	//Mtx matrix converters

	///Read in a matrix-market matrix file (.mtx) and convert it to an Eigen_Trip.
	///If symmetric, only half is in file, must make entire matrix.
	///If base_zero, the matrix indexing in the .mtx file starts from zero
	template<typename T>
	Eigen_Trip<T> mtxMatrix_to_Eigen_Trip(const std::string& mat_loc,
		unsigned int start_line, bool symmetric, bool base_zero, bool printMatSize)
	{
		std::ifstream f(mat_loc, std::ios::in);

		//Matrix market files are typically formatted in following manner:
		//1) Name of matrix / matrix details.
		//2) Number of rows, number of columns, number of entries/nonzeros.
		//3) Values of matrix in matrix market format.
		//Matrix market format: row index, col index, value.
		//User must ensure start_line is the matrix size line (2).
		BlockSp::tests::GotoLine(f, start_line);
		std::string size; std::getline(f, size);
		std::stringstream ss(size);
		int M, N, nnz; ss >> M; ss >> N; ss >> nnz;
		if (printMatSize)
		{
			std::cout << M << ' ' << N << ' ' << nnz;
			if (symmetric) std::cout << " Symmetric";
			std::cout << std::endl;
		}

		Eigen_Trip<T> trip; trip.M = M; trip.N = N;
		for (int innz = 0; innz < nnz; ++innz)
		{
			std::string entry; std::getline(f, entry);
			int i, j; T val;
			std::stringstream ss2(entry);
			ss2 >> i; ss2 >> j; //write matrix index and values. 
			if (!base_zero)
			{
				i -= 1; j -= 1;		//change to base 0 indexing.
			}
			if constexpr (BlockSp::is_complex_v<T>) //also read imaginary value.
			{
				double valReal, valImg; ss2 >> valReal; ss2 >> valImg;
				val.real(valReal); val.imag(valImg);
				trip.push_back(Eigen::Triplet<T>(i, j, val));
				if (symmetric && i != j) //if symmetric, fill (j,i) with complex conjugate.
					trip.push_back(Eigen::Triplet<T>(j, i, std::conj(val)));
			}
			else
			{
				ss2 >> val;
				trip.push_back(Eigen::Triplet<T>(i, j, val));
				if (symmetric && i != j) //if symmetric, fill (j,i)
					trip.push_back(Eigen::Triplet<T>(j, i, val));
			}
		}
		return trip;
	}

	///Read in an mtx matrix and make a row major Eigen::SparseMatrix
	template<typename T>
	Eigen::SparseMatrix<T, Eigen::RowMajor> mtxMatrix_to_EigenSpMatrix(
		const std::string& mat_loc, unsigned int start_line, bool symmetric,
		bool base_zero, bool printMatSize)
	{
		auto trip{ mtxMatrix_to_Eigen_Trip<T>(mat_loc, start_line, symmetric, base_zero, printMatSize) };
		Eigen::SparseMatrix<T, Eigen::RowMajor> A(trip.M, trip.N);
		A.setFromTriplets(trip.m_trip.begin(), trip.m_trip.end());
		return A;
	}

	///Read in a matrix-market vector file (.mtx) and convert it to an Eigen::Vector.
	template<typename T>
	Eigen::Vector<T, Eigen::Dynamic> mtxArray_to_EigenVector(const std::string& vec_loc,
		unsigned int start_line, bool printVecSize)
	{
		std::ifstream f(vec_loc, std::ios::in);

		//matrix market files are formatted in following manner
		//1) name of vector
		//2) number of rows, number of cols (1)
		//3+) values of vector
		BlockSp::tests::GotoLine(f, start_line); //user must ensure starting line is vector size line 
		std::string size; std::getline(f, size);
		std::stringstream ss(size);
		int M, N; ss >> M; ss >> N;
		assert(N == 1);

		Eigen::Vector<T, Eigen::Dynamic> vec(M);
		for (int im = 0; im < M; ++im)
		{
			std::string entry; std::getline(f, entry);
			std::stringstream ss2(entry);
			T val;
			if constexpr (BlockSp::is_complex_v<T>) //also read imaginary value.
			{
				long double valReal, valImg; ss2 >> valReal; ss2 >> valImg;
				val.real(valReal); val.imag(valImg);
			}
			else ss2 >> val;
			vec(im) = val;
		}
		return vec;
	}

}//end namespace eigen_compare

#endif //end BLOCKSP_EIGEN_COMPARE_HPP
