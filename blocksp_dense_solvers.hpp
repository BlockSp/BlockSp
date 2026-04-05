#ifndef BLOCKSP_DENSE_SOLVERS_HPP
#define BLOCKSP_DENSE_SOLVERS_HPP

#include <blitz/array.h>
#include "blocksp_dense_multiply.hpp"
#include "blocksp_lapack.hpp"
#include "blocksp_utility.hpp"


///blocksp_dense_solvers.hpp contains the dense linear algebra solver functions used in BlockSp. \n
///These include: an explicit small matrix solver, LU and Cholesky factorizations/solvers, 
///symmetric and nonsymmetric eigenvalue factorizations/solvers. \n
///There are two versions of each: \n
/// - one is compile-time optimized, for TinyMatrices. \n
/// - the other is for 2D blitz::Arrays, the size of which is specified at run-time. 
///		these are suffixed by _Arr. \n
///There is also an eigenvalue calculator for 2D blitz::Arrays (not a solver).
namespace BlockSp::dense
{

	/////////////////////////////////////////
	///Dense matrix factorizations/solvers
	/////////////////////////////////////////
	
	///Four factorizations/solvers for dense square matrices.
	///1) LUFactor					 - LU decomposition/solver, for general matrices. \n
	///2) CholeskyFactor		 - Cholesky decomposition/solver, for real symmetric positive definite (SPD) matrices.\n 
	///3) SymEigenFactor		 - Eigenvalue decomposition/solver, for real symmetric matrices.\n
	///4) NonSymEigenFactor  - Eigenvalue decomposition/solver, for general matrices.\n
	///with _Arr versions for 2D blitz::Arrays. \n
	///Also, \n
	///small_solve					 - Fast expicit solver for small 1x1, 2x2, and 3x3 linear systems.\n
	///Eigenvalues_Arr			 - Eigenvalue calculator for general 2D blitz::Arrays.  \n
	///Note: NonSymEigenFactor(_Arr) can be used as a robust solver for general dense matrices.  

	//Forward declare small_solve.
	//Solve Ax = b for A small, 1x1, 2x2, and 3x3 matrices.
	template<typename T, int N> requires (N <= 3)
	blitz::TinyVector<T, N> small_solve(const blitz::TinyMatrix<T, N, N>& A,
		const blitz::TinyVector<T, N>& b);
	template<typename T, int N> requires (N <= 3)
	blitz::TinyMatrix<T, N, N> small_solve(const blitz::TinyMatrix<T, N, N>& A,
		const blitz::TinyMatrix<T, N, N>& B, bool right_solve = false);

	///LU factorization of NxN TinyMatrix.
	///May be used for inverting block diagonals in SOR.
	template<typename T, int N>
	struct LUFactor
	{
		blitz::TinyMatrix <T, N, N> LU;
		blitz::TinyVector<lapack_int, N> ipiv;

		LUFactor() {}
		LUFactor(const blitz::TinyMatrix <T, N, N>& A) : LU{ A }
		{
			//Use explicit small_solve for N <= 3. LU factorization for N > 3. 
			if constexpr (N > 3) lapack::getrf<T>(LAPACK_ROW_MAJOR, N, N, LU.data(), N, ipiv.data());
		}
		~LUFactor() {}

		///inverse calculation, y = A^-1 x
		void invmul(const blitz::TinyMatrix<T, N, N>& X, blitz::TinyMatrix<T, N, N>& Y) const
		{
			if constexpr (N <= 3)	Y = small_solve<T, N>(LU, X);
			else
			{
				Y = X;
				lapack::getrs<T>(LAPACK_ROW_MAJOR, 'N', N, N, LU.data(), N, ipiv.data(), Y.data(), N);
			}
		}

		void invmul(blitz::TinyMatrix<T, N, N>& X) const
		{
			if constexpr (N <= 3)	X = small_solve<T, N>(LU, X);
			else lapack::getrs<T>(LAPACK_ROW_MAJOR, 'N', N, N, LU.data(), N, ipiv.data(), X.data(), N);
		}

		void invmul(const blitz::TinyVector<T, N>& x, blitz::TinyVector<T, N>& y) const
		{
			if constexpr (N <= 3)	y = small_solve<T, N>(LU, x);
			else
			{
				y = x;
				lapack::getrs<T>(LAPACK_ROW_MAJOR, 'N', N, 1, LU.data(), N, ipiv.data(), y.data(), 1);
			}
		}

		void invmul(blitz::TinyVector<T, N>& x) const
		{
			if constexpr (N <= 3)	x = small_solve<T, N>(LU, x);
			else lapack::getrs<T>(LAPACK_ROW_MAJOR, 'N', N, 1, LU.data(), N, ipiv.data(), x.data(), 1);
		}

		///Right inverse calculation, Y = X A^-1.
		///Explicit small_solve used for small matrices,
		///calculated via A^T X^T = Y^T for larger matrices.
		void r_invmul(blitz::TinyMatrix<T, N, N>& X) const
		{
			if constexpr (N <= 3) X = small_solve<T, N>(LU, X, true);
			else
			{
				transpose_in_place(X); //make Y^T
				if constexpr (is_complex_v<T>) //solve A^T X^T Y^T
					lapack::getrs<T>(LAPACK_ROW_MAJOR, 'C', N, N, LU.data(), N, ipiv.data(), X.data(), N);
				else lapack::getrs<T>(LAPACK_ROW_MAJOR, 'T', N, N, LU.data(), N, ipiv.data(), X.data(), N);
				transpose_in_place(X); //get X
			}
		}

	};//end LUFactor

	//Forward declare small_solve for symmetric matrices stored in compact vector form.
	//The vector spans the LOWER triangular portion of matrix A.
	template<typename T, int N> requires (N <= 3)
	blitz::TinyVector<T, N> small_solve(const blitz::TinyVector<T, ((N + 1)* N) / 2>& A,
		const blitz::TinyVector<T, N>& b);
	template<typename T, int N> requires (N <= 3)
	blitz::TinyMatrix<T, N, N> small_solve(const blitz::TinyVector <T, ((N + 1)* N) / 2>& A,
		const blitz::TinyMatrix<T, N, N>& B, bool right_solve = false);

	///Cholesky factorization: A = LL^T of an NxN symmetric positive definite TinyMatrix.
	///May be used for inverting block diagonals in symmetric SOR.
	template<typename T, int N> requires (!is_complex_v<T>)
	struct CholeskyFactor
	{
		blitz::TinyVector <T, ((N + 1)* N) / 2> L;

		CholeskyFactor() {}
		CholeskyFactor(const blitz::TinyMatrix <T, N, N>& A)
		{
			int il = 0;
			for (int im = 0; im < N; ++im)
				for (int in = 0; in <= im; ++in)
				{
					L(il) = A(im, in);
					++il;
				}
			if constexpr (N > 3) //Use explicit small_solve for N <= 3. Cholesky factorization for N > 3. 
				lapack::pptrf<T>(LAPACK_ROW_MAJOR, 'L', N, L.data());
		}
		~CholeskyFactor() {}

		///Inverse calculation, y = A^-1 x
		void invmul(const blitz::TinyMatrix<T, N, N>& X, blitz::TinyMatrix<T, N, N>& Y) const
		{
			if constexpr (N <= 3)	Y = small_solve<T, N>(L, X);
			else
			{
				Y = X;
				lapack::pptrs<T>(LAPACK_ROW_MAJOR, 'L', N, N, L.data(), Y.data(), N);
			}
		}

		void invmul(blitz::TinyMatrix<T, N, N>& X) const
		{
			if constexpr (N <= 3)	X = small_solve<T, N>(L, X);
			else lapack::pptrs<T>(LAPACK_ROW_MAJOR, 'L', N, N, L.data(), X.data(), N);
		}

		void invmul(const blitz::TinyVector<T, N>& x, blitz::TinyVector<T, N>& y) const
		{
			if constexpr (N <= 3)	y = small_solve<T, N>(L, x);
			else
			{
				y = x;
				lapack::pptrs<T>(LAPACK_ROW_MAJOR, 'L', N, 1, L.data(), y.data(), 1);
			}
		}

		void invmul(blitz::TinyVector<T, N>& x) const
		{
			if constexpr (N <= 3)	x = small_solve<T, N>(L, x);
			else lapack::pptrs<T>(LAPACK_ROW_MAJOR, 'L', N, 1, L.data(), x.data(), 1);
		}

		//Right inverse calculation, Y = X A^-1.
		//Explicit small_solve used for small matrices,
		//calculated via A^T X^T = Y^T for larger matrices.
		void r_invmul(blitz::TinyMatrix<T, N, N>& X) const
		{
			if constexpr (N <= 3) X = small_solve<T, N>(L, X, true);
			else
			{
				transpose_in_place(X); //make Y^T
				lapack::pptrs<T>(LAPACK_ROW_MAJOR, 'L', N, N, L.data(), X.data(), N);
				transpose_in_place(X); //get X
			}
		}

	};//end CholeskyFactor

	///Symmetric eigenvalue factorization - for real, symmetric NxN matrix: \n
	///A = QDQ^T, A^-1 = QD^-1Q^T. Stores A^-1 and D. \n
	///Regularizing: removes near zero eigen values.
	///May be used for inverting block diagonals in symmetric SOR.
	template<typename T, int N> requires (!is_complex_v<T>)
	struct SymEigenFactor
	{
		blitz::TinyVector<T, N> D;
		blitz::TinyMatrix<T, N, N> AInv;
		T tolFactor;

		SymEigenFactor() {}

		SymEigenFactor(blitz::TinyMatrix <T, N, N> A, T tolFactor = 1.0) :
			tolFactor{ tolFactor }
		{
			//eigen decomp
			lapack_int m;
			blitz::TinyVector <lapack_int, N * 2> supp;
			blitz::TinyMatrix<T, N, N> Q;
			lapack::syevr<T>(LAPACK_ROW_MAJOR, 'V', 'A', 'U', N, A.data(), N, NULL,
				NULL, NULL, NULL, 0.0, &m, D.data(), Q.data(), N, supp.data());

			//Determine threshold for regularisation, equal to a prefactor times
			//N*N*eps times the first (and largest) eigenvalue
			T tol = 0;
			for (int i = 0; i < N; ++i)
				if (std::abs(D(i)) > tol)
					tol = std::max(tol, std::abs(D(i)));
			tol *= tolFactor * N * N * std::numeric_limits<T>::epsilon();
			assert(tol > 0.0);

			//remove eigenvalues below threshhold
			for (int i = 0; i < N; ++i)
				if (std::abs(D(i)) < tol)
					D(i) = 0.0;

			//make A^-1 = QD^-1Q^T
			blitz::TinyMatrix<T, N, N> QDI(Q);
			for (int i = 0; i < N; ++i)
				for (int j = 0; j < N; ++j)
					if (std::abs(D(j)) > tol)
						QDI(i, j) /= D(j);
					else QDI(i, j) = 0.0;
			lapack::gemm<T>(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, QDI.data(), N,
				Q.data(), N, 0, AInv.data(), N);
		}

		///Inverse calculation, y = A^-1 x. A^-1 = QD^-1Q^T
		void invmul(const blitz::TinyMatrix<T, N, N>& X, blitz::TinyMatrix<T, N, N>& Y) const
		{
			Y = matMat<T, N>(AInv, X);
		}

		void invmul(blitz::TinyMatrix<T, N, N>& X) const
		{
			X = matMat<T, N>(AInv, X);
		}

		void invmul(const blitz::TinyVector<T, N>& x, blitz::TinyVector<T, N>& y) const
		{
			y = matVec<T, N, N>(AInv, x);
		}

		void invmul(blitz::TinyVector<T, N>& x) const
		{
			x = matVec<T, N, N>(AInv, x);
		}

		///////////////////////
		//Eigenvalue info

		void printEigVals() const
		{
			std::cout << "Eigenvalues" << std::endl;
			std::cout << D << std::endl;
		}

		void printEigInfo() const
		{
			std::cout << "min eig " << min_ev() << std::endl;
			std::cout << "max eig " << max_ev() << std::endl;
			std::cout << "smallest magn eig " << min_mag() << std::endl;
			std::cout << "largest magn eig  " << max_mag() << std::endl;
			std::cout << "ratio/cond " << cond_number() << std::endl;
			std::cout << std::endl;
		}

		//eigenvalue statistics
		T min_ev() const { return blitz::min(D); }	//smallest eigenvalue
		T max_ev() const { return blitz::max(D); }	//largest eigenvalue
		T min_mag() const { return blitz::min(blitz::abs(D)); }	//smallest eigenvalue
		T max_mag() const { return blitz::max(blitz::abs(D)); }	//largest eigenvalue
		int arg_min_mag() const { return util::tvArgMin(blitz::abs(D)); } //index with smallest eigenvalue
		int arg_max_mag() const { return util::tvArgMax(blitz::abs(D)); } //index with largest eigenvalue
		T cond_number() const { return max_mag() / min_mag(); } //matrix A condition number.

	};//end SymEigenFactor

	///Nonsymmetric Eigenvalue factorization - for real or complex nonsymmetric NxN matrix: \n
	///Schur decomp A = QUQ^T, A^-1 = QU^-1Q^T. Stores A^-1 and eigenvalues.
	///U upper triangular with eigenvalues on diagonal. \n
	///Regularizing: removes near zero eigen values.
	///May be used for inverting block diagonals in SOR. \n
	///Note: there may be imaginary eigenvalues, which gives a block triangular U (real Schur form).
	///Block sizes are 1 if eig real, 2 if imaginary. For real matrix, blocks are complex conjugate pairs.
	template<typename T, int N>
	struct NonSymEigenFactor
	{
		using S = lapack::ScalarType_t<T>;
		blitz::TinyMatrix<T, N, N> U, AInv;
		blitz::TinyVector<S, N> WR, WI, WM;	//real and imaginary parts of evals + magnitude
		S tol, tolFactor;
		bool imagE; //imaginary eigenvalues 

		NonSymEigenFactor() {}

		NonSymEigenFactor(const blitz::TinyMatrix <T, N, N>& A, S tolFactor = 1.0) :
			U{ A }, tolFactor{ tolFactor }
		{
			//eigen decomp
			blitz::TinyVector<T, N - 1> tau;
			lapack::gehrd<T>(LAPACK_ROW_MAJOR, N, 1, N, U.data(), N, tau.data()); //reduce to upper Hessenburg
			auto Q = U;
			if constexpr (lapack::is_complex_v<T>)
			{
				lapack::unghr<T>(LAPACK_ROW_MAJOR, N, 1, N, Q.data(), N, tau.data()); //generate Q
				blitz::TinyVector<T, N> W;
				lapack::hseqr<T>(LAPACK_ROW_MAJOR, 'S', 'V', N, 1, N, U.data(), N,		//find eigenvalues
					W.data(), Q.data(), N);
				for (int i = 0; i < N; ++i)
				{
					WR(i) = W(i).real(); WI(i) = W(i).imag();
				}
			}
			else
			{
				lapack::orghr<T>(LAPACK_ROW_MAJOR, N, 1, N, Q.data(), N, tau.data()); //generate Q
				lapack::hseqr<T>(LAPACK_ROW_MAJOR, 'S', 'V', N, 1, N, U.data(), N,		//find eigenvalues
					WR.data(), WI.data(), Q.data(), N);
			}

			//magnitude of eigenvalues
			for (int i = 0; i < N; ++i)
				WM(i) = std::sqrt(WR(i) * WR(i) + WI(i) * WI(i));

			//Determine threshold for regularisation, equal to a prefactor times
			//N*N*eps times the first (and largest) eigenvalue
			tol = 0;
			for (int i = 0; i < N; ++i)
				if (WM(i) > tol)
					tol = std::max(tol, WM(i));
			tol *= tolFactor * N * N * std::numeric_limits<S>::epsilon();
			assert(tol > 0.0);

			//check WI all zeros
			imagE = false;
			for (const auto& eigI : WI)
				if (std::abs(eigI) > tol)
					imagE = true;

			//remove eigenvalues below threshhold
			for (int i = 0; i < N; ++i)
				if (std::abs(U(i, i)) < tol)
					U(i, i) = 0.0;

			//make A^-1 = Q U^-1 Q^T
			auto ph = transpose(Q);
			for (int j = 0; j < N; ++j)
			{
				blitz::TinyVector<T, N> col;
				for (int i = 0; i < N; ++i)
					col(i) = ph(i, j);
				UInv(col);
				for (int i = 0; i < N; ++i)
					ph(i, j) = col(i);
			}
			lapack::gemm<T>(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1, Q.data(), N, ph.data(), N, 0, AInv.data(), N);
		}

		///Apply U^-1 via backwards substitution.
		///If there are imaginary eigenvalues, use a block backwards substution.
		void UInv(blitz::TinyVector<T, N>& y) const
		{
			for (int i = N - 1; i >= 0; --i)
			{
				//determine block size(ie is there a nonzero entry on left side of diagonal?)
				int blkSize;
				if (i != 0 && imagE) blkSize = std::abs(U(i, i - 1)) > tol ? 2 : 1;
				else blkSize = 1;

				if (blkSize == 1)
				{
					for (int j = i + 1; j < N; ++j)
						y(i) -= U(i, j) * y(j);
					if (U(i, i) == 0.0)
						y(i) = 0.0;
					else
						y(i) /= U(i, i);
				}
				else
				{
					//block regularize

					//block back subst
					blitz::TinyVector<T, 2> yii; yii(0) = y(i - 1); yii(1) = y(i);
					for (int k = 0; k < 2; ++k)
						for (int j = i + 1; j < N; ++j)
							yii(k) -= U(i - 1 + k, j) * y(j);

					//apply block Uii^-1
					T a = U(i - 1, i - 1); T b = U(i - 1, i); T c = U(i, i - 1); T d = U(i, i);
					blitz::TinyMatrix<T, 2, 2> UiiInv; UiiInv(0, 0) = d; UiiInv(0, 1) = -b;
					UiiInv(1, 0) = -c; UiiInv(1, 1) = a;
					UiiInv /= (a * d - b * c);
					y(i - 1) = UiiInv(0, 0) * yii(0) + UiiInv(0, 1) * yii(1);
					y(i) = UiiInv(1, 0) * yii(0) + UiiInv(1, 1) * yii(1);
					--i; //skip past top row of block
				}
			}
		}

		///Inverse calculation: y = A^-1 x = Q U^-1 Q^T x
		void invmul(const blitz::TinyMatrix<T, N, N>& X, blitz::TinyMatrix<T, N, N>& Y) const
		{
			Y = matMat<T, N>(AInv, X);
		}

		void invmul(blitz::TinyMatrix<T, N, N>& X) const
		{
			X = matMat<T, N, N>(AInv, X);
		}

		void invmul(const blitz::TinyVector<T, N>& x, blitz::TinyVector<T, N>& y) const
		{
			y = matVec<T, N, N>(AInv, x);
		}

		void invmul(blitz::TinyVector<T, N>& x) const
		{
			x = matVec<T, N, N>(AInv, x);
		}

		//////////////////////////
		//Eigenvalue info

		void printEigVals() const
		{
			std::cout << "Eigenvalues - real component" << std::endl;
			std::cout << WR << std::endl;
			std::cout << "Eigenvalues - imaginary component" << std::endl;
			std::cout << WI << std::endl;
			std::cout << "Eigenvalues - magnitude" << std::endl;
			std::cout << WM << std::endl;
		}

		void printEigInfo() const
		{
			std::cout << "min/max real eig " << min_real() << ' ' << max_real() << std::endl;
			std::cout << "min/max img  eig " << min_imag() << ' ' << max_imag() << std::endl;
			std::cout << "smallest magn eig " << min_mag() << std::endl;
			std::cout << "largest magn eig  " << max_mag() << std::endl;
			std::cout << "ratio/cond " << cond_number() << std::endl;
		}

		//real eigenvalue statistics
		S min_real() const { return blitz::min(WR); }
		S max_real() const { return blitz::max(WR); }
		S min_abs_real() const { return blitz::min(blitz::abs(WR)); }
		S max_abs_real() const { return blitz::max(blitz::abs(WR)); }
		
		//imaginary eigenvalue statistics
		S min_imag() const { return blitz::min(WI); }
		S max_imag() const { return blitz::max(WI); }
		S min_abs_imag() const { return blitz::min(blitz::abs(WI)); }
		S max_abs_imag() const { return blitz::max(blitz::abs(WI)); }

		//magnitude eigenvalue statistics
		S min_mag() const { return blitz::min(WM); }	//smallest eigenvalue
		S max_mag() const { return blitz::max(WM); }	//largest eigenvalue
		int arg_min_mag() const{ return util::tvArgMin(WM); } //index with smallest eigenvalue
		int arg_max_mag() const{ return util::tvArgMax(WM); } //index with largest eigenvalue
		S cond_number() const { return max_mag() / min_mag(); } //matrix A condition number.

	};//end NonSymEigenFactor


	////////////////////////////////
	//2D blitz::Array versions
	////////////////////////////////

	///LU factorization of general square 2D blitz::Array.
	template<typename T>
	struct LUFactor_Arr
	{
		int N;
		blitz::Array <T, 2> LU;
		blitz::Array<lapack_int, 1> ipiv;

		LUFactor_Arr() {}
		LUFactor_Arr(const blitz::Array<T, 2>& A)
		{
			assert(A.extent(0) == A.extent(1));
			N = A.extent(0);
			ipiv.resize(N);
			LU.resize(N); LU = A;
			lapack::getrf<T>(LAPACK_ROW_MAJOR, N, N, LU.data(), N, ipiv.data());
		}
		~LUFactor_Arr() {}

		bool arr_size_check(const blitz::Array<T, 1>& x) const { return x.extent(0) == N; }
		bool arr_size_check(const blitz::Array<T, 2>& A) const { return A.extent(0) == N && A.extent(1) == N; }

		void invmul(const blitz::Array<T, 2>& X, blitz::Array<T, 2>& Y) const
		{
			assert(arr_size_check(X) && arr_size_check(Y));
			Y = X;
			lapack::getrs<T>(LAPACK_ROW_MAJOR, 'N', N, N, LU.data(), N, ipiv.data(), Y.data(), N);
		}

		void invmul(blitz::Array<T, 2>& X) const
		{
			assert(arr_size_check(X));
			lapack::getrs<T>(LAPACK_ROW_MAJOR, 'N', N, N, LU.data(), N, ipiv.data(), X.data(), N);
		}

		void invmul(const blitz::Array<T, 1>& x, blitz::Array<T, 1>& y) const
		{
			assert(arr_size_check(x) && arr_size_check(y));
			y = x;
			lapack::getrs<T>(LAPACK_ROW_MAJOR, 'N', N, 1, LU.data(), N, ipiv.data(), y.data(), 1);
		}

		void invmul(blitz::Array<T, 1>& x) const
		{
			assert(arr_size_check(x));
			lapack::getrs<T>(LAPACK_ROW_MAJOR, 'N', N, 1, LU.data(), N, ipiv.data(), x.data(), 1);
		}

	};//end LUFactor_Arr

	///Cholesky factorization: A = LL^T of a symmetric positive definite 2D blitz::Array.
	template<typename T> requires (!is_complex_v<T>)
	struct CholeskyFactor_Arr
	{
		int N;
		blitz::Array<T, 1> L;

		CholeskyFactor_Arr() {}
		CholeskyFactor_Arr(const blitz::Array<T, 2>& A)
		{
			assert(A.extent(0) == A.extent(1));
			N = A.extent(0);
			L.resize(((N + 1) * N) / 2);
			int il = 0;
			for (int im = 0; im < N; ++im)
				for (int in = 0; in <= im; ++in)
				{
					L(il) = A(im, in);
					++il;
				}
			lapack::pptrf<T>(LAPACK_ROW_MAJOR, 'L', N, L.data());
		}
		~CholeskyFactor_Arr() {}

		bool arr_size_check(const blitz::Array<T, 1>& x) const { return x.extent(0) == N; }
		bool arr_size_check(const blitz::Array<T, 2>& A) const { return A.extent(0) == N && A.extent(1) == N; }

		void invmul(const blitz::Array<T, 2>& X, blitz::Array<T, 2>& Y) const
		{
			assert(arr_size_check(X) && arr_size_check(Y));
			Y = X;
			lapack::pptrs<T>(LAPACK_ROW_MAJOR, 'L', N, N, L.data(), Y.data(), N);
		}

		void invmul(blitz::Array<T, 2>& X) const
		{
			assert(arr_size_check(X));
			lapack::pptrs<T>(LAPACK_ROW_MAJOR, 'L', N, N, L.data(), X.data(), N);
		}

		void invmul(const blitz::Array<T, 1>& x, blitz::Array<T, 1>& y) const
		{
			assert(arr_size_check(x) && arr_size_check(y));
			y = x;
			lapack::pptrs<T>(LAPACK_ROW_MAJOR, 'L', N, 1, L.data(), y.data(), 1);
		}

		void invmul(blitz::Array<T, 1>& x) const
		{
			assert(arr_size_check(x));
			lapack::pptrs<T>(LAPACK_ROW_MAJOR, 'L', N, 1, L.data(), x.data(), 1);
		}

	};//end CholeskyFactor_Arr

	///Symmetric eigenvalue factorization - for real, symmetric 2D blitz::Arrays \n
	///A = QDQ^T, A^-1 = QD^-1Q^T. \n
	///Regularizing: removes near zero eigen values.
	///Used for inverting diagonals in SOR, stores A^-1 and D.
	template<typename T>  requires (!is_complex_v<T>)
	struct SymEigenFactor_Arr
	{
		blitz::Array<T, 1> D;
		blitz::Array<T, 2> AInv;
		int N;
		T tolFactor;

		SymEigenFactor_Arr() {}

		SymEigenFactor_Arr(const blitz::Array<T, 2>& A, T tolFactor = 1.0) :
			tolFactor{ tolFactor }
		{
			blitz::Array<T, 2> ACopy{ A.copy() };
			assert(A.extent(0) == A.extent(1));
			N = A.extent(0);
			D.resize(N);
			AInv.resize(N);

			//eigen decomp
			lapack_int m;
			blitz::Array<lapack_int, 1> supp(2 * N);
			blitz::Array<T, 2> Q(N);
			lapack::syevr<T>(LAPACK_ROW_MAJOR, 'V', 'A', 'U', N, ACopy.data(), N, NULL,
				NULL, NULL, NULL, 0.0, &m, D.data(), Q.data(), N, supp.data());

			//Determine threshold for regularisation, equal to a prefactor times
			//N*N*eps times the first (and largest) eigenvalue
			T tol = 0;
			for (int i = 0; i < N; ++i)
				if (std::abs(D(i)) > tol)
					tol = std::max(tol, std::abs(D(i)));
			tol *= tolFactor * N * N * std::numeric_limits<T>::epsilon();
			assert(tol > 0.0);

			//remove eigenvalues below threshhold
			for (int i = 0; i < N; ++i)
				if (std::abs(D(i)) < tol)
					D(i) = 0.0;

			//make A^-1 = QD^-1Q^T
			blitz::Array<T, 2> QDI{ Q.copy() };
			for (int i = 0; i < N; ++i)
				for (int j = 0; j < N; ++j)
					if (std::abs(D(j)) > tol)
						QDI(i, j) /= D(j);
					else QDI(i, j) = 0.0;
			lapack::gemm<T>(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, QDI.data(), N,
				Q.data(), N, 0, AInv.data(), N);
		}

		bool arr_size_check(const blitz::Array<T, 1>& x) const { return x.extent(0) == N; }
		bool arr_size_check(const blitz::Array<T, 2>& A) const { return A.extent(0) == N && A.extent(1) == N; }

		///y = A^-1 x
		void invmul(const blitz::Array<T, 2>& X, blitz::Array<T, 2>& Y) const
		{
			assert(arr_size_check(X) && arr_size_check(Y));
			lapack::gemm<T>(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1, AInv.data(), N, X.data(), N, 0, Y.data(), N);
		}

		void invmul(blitz::Array<T, 2>& X) const
		{
			assert(arr_size_check(X));
			blitz::Array<T, 2> ph(N);
			lapack::gemm<T>(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1, AInv.data(), N, X.data(), N, 0, ph.data(), N);
			X = ph;
		}

		void invmul(const blitz::Array<T, 1>& x, blitz::Array<T, 1>& y) const
		{
			assert(arr_size_check(x) && arr_size_check(y));
			lapack::gemv<T>(CblasRowMajor, CblasNoTrans, N, N, 1.0, AInv.data(), N, x.data(), 1, 0.0, y.data(), 1);
		}

		void invmul(blitz::Array<T, 1>& x) const
		{
			assert(arr_size_check(x));
			blitz::Array<T, 1> ph(N);
			lapack::gemv<T>(CblasRowMajor, CblasNoTrans, N, N, 1.0, AInv.data(), N, x.data(), 1, 0.0, ph.data(), 1);
			x = ph;
		}

		void printEigVals() const
		{
			std::cout << "Eigenvalues" << std::endl;
			std::cout << D << std::endl;
		}

		void printEigInfo() const
		{
			std::cout << "min eig " << min_ev() << std::endl;
			std::cout << "max eig " << max_ev() << std::endl;
			std::cout << "smallest magn eig " << min_mag() << std::endl;
			std::cout << "largest magn eig  " << max_mag() << std::endl;
			std::cout << "ratio/cond " << cond_number() << std::endl;
			std::cout << std::endl;
		}

		//eigenvalue statistics
		T min_ev() const { return blitz::min(D); }	//smallest eigenvalue
		T max_ev() const { return blitz::max(D); }	//largest eigenvalue
		T min_mag() const { return blitz::min(blitz::abs(D)); }	//smallest eigenvalue
		T max_mag() const { return blitz::max(blitz::abs(D)); }	//largest eigenvalue
		int arg_min_mag() const { return util::tvArgMin(blitz::abs(D)); } //index with smallest eigenvalue
		int arg_max_mag() const { return util::tvArgMax(blitz::abs(D)); } //index with largest eigenvalue
		T cond_number() const { return max_mag() / min_mag(); } //matrix A condition number.

	};//end SymEigenFactor_Arr

	///Same as NonSymEigenFactor, but for 2D blitz::Arrays. \n
	///Eigen decomposition of nonsymmetric NxN matrix, real or complex: \n 
	///Schur decomp A = QUQ^T, A^-1 = QU^-1Q^T, U upper triangular with eigs on diag. \n
	///Regularizing: removes near zero eigen values
	///Used for robust dense solves, stores A^-1 and eigenvalues
	///Note: there may be imaginary eigenvalues, which gives a block triangular U (real Schur form).
	///Block sizes are 1 if eig real, 2 if imaginary. For real matrix, blocks are complex conjugate pairs
	template<typename T>
	struct NonSymEigenFactor_Arr
	{
		using S = lapack::ScalarType_t<T>;
		blitz::Array<T, 2> U, AInv;
		blitz::Array<S, 1> WR, WI, WM;	///< Real and imaginary parts of evals + magnitude
		S tol, tolFactor;
		int N;			///Size of matrix, must be square.
		bool imagE; ///Are there imaginary eigenvalues? 

		NonSymEigenFactor_Arr() {}

		NonSymEigenFactor_Arr(const blitz::Array<T, 2>& A, S tolFactor = 1.0) :
			U{ A.copy() }, tolFactor{ tolFactor }
		{
			assert(A.extent(0) == A.extent(1));
			N = A.extent(0);
			WR.resize(N); WI.resize(N); WM.resize(N);

			//eigen decomp
			blitz::Array<T, 1> tau(N - 1);
			lapack::gehrd<T>(LAPACK_ROW_MAJOR, N, 1, N, U.data(), N, tau.data());
			auto Q{ U.copy() };
			if constexpr (lapack::is_complex_v<T>)
			{
				lapack::unghr<T>(LAPACK_ROW_MAJOR, N, 1, N, Q.data(), N, tau.data()); //generate Q
				blitz::Array<T, 1> W(N);
				lapack::hseqr<T>(LAPACK_ROW_MAJOR, 'S', 'V', N, 1, N, U.data(), N,		//find eigenvalues
					W.data(), Q.data(), N);
				for (int i = 0; i < N; ++i)
				{
					WR(i) = W(i).real(); WI(i) = W(i).imag();
				}
			}
			else
			{
				lapack::orghr<T>(LAPACK_ROW_MAJOR, N, 1, N, Q.data(), N, tau.data()); //generate Q
				lapack::hseqr<T>(LAPACK_ROW_MAJOR, 'S', 'V', N, 1, N, U.data(), N,		//find eigenvalues
					WR.data(), WI.data(), Q.data(), N);
			}

			//magnitude of eigenvalues
			for (int i = 0; i < N; ++i)
				WM(i) = std::sqrt(WR(i) * WR(i) + WI(i) * WI(i));

			//Determine threshold for regularisation, equal to a prefactor times
			//N*N*eps times the first (and largest) eigenvalue
			tol = 0;
			for (int i = 0; i < N; ++i)
				if (WM(i) > tol)
					tol = std::max(tol, WM(i));
			tol *= tolFactor * N * N * std::numeric_limits<S>::epsilon();
			assert(tol > 0.0);

			//check WI all zeros
			imagE = false;
			for (const auto& eigI : WI)
				if (std::abs(eigI) > tol)
					imagE = true;

			//remove eigenvalues below threshhold
			for (int i = 0; i < N; ++i)
				if (std::abs(U(i, i)) < tol)
					U(i, i) = 0.0;

			//make A^-1 = Q U^-1 Q^T
			auto ph = transpose(Q);
			for (int j = 0; j < N; ++j)
			{
				blitz::Array<T, 1> col(N);
				for (int i = 0; i < N; ++i)
					col(i) = ph(i, j);
				UInv(col);
				for (int i = 0; i < N; ++i)
					ph(i, j) = col(i);
			}
			AInv.resize(N);
			lapack::gemm<T>(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1, Q.data(), N, ph.data(), N, 0, AInv.data(), N);
		}

		bool arr_size_check(const blitz::Array<T, 1>& x) const { return x.extent(0) == N; }
		bool arr_size_check(const blitz::Array<T, 2>& A) const { return A.extent(0) == N && A.extent(1) == N; }

		//apply U^-1 via backwards substitution
		//if there are imaginary eigenvalues, use a block backwards substution
		void UInv(blitz::Array<T, 1>& y) const
		{
			assert(y.extent(0) == N);
			for (int i = N - 1; i >= 0; --i)
			{
				//determine block size(ie is there a nonzero entry on left side of diagonal?)
				int blkSize;
				if (i != 0 && imagE) blkSize = std::abs(U(i, i - 1)) > tol ? 2 : 1;
				else blkSize = 1;

				if (blkSize == 1)
				{
					for (int j = i + 1; j < N; ++j)
						y(i) -= U(i, j) * y(j);
					if (U(i, i) == 0.0)
						y(i) = 0.0;
					else
						y(i) /= U(i, i);
				}
				else
				{
					//block regularize

					//block back subst
					blitz::TinyVector<T, 2> yii; yii(0) = y(i - 1); yii(1) = y(i);
					for (int k = 0; k < 2; ++k)
						for (int j = i + 1; j < N; ++j)
							yii(k) -= U(i - 1 + k, j) * y(j);

					//apply block Uii^-1
					T a = U(i - 1, i - 1); T b = U(i - 1, i); T c = U(i, i - 1); T d = U(i, i);
					blitz::TinyMatrix<T, 2, 2> UiiInv; UiiInv(0, 0) = d; UiiInv(0, 1) = -b;
					UiiInv(1, 0) = -c; UiiInv(1, 1) = a;
					UiiInv /= (a * d - b * c);
					y(i - 1) = UiiInv(0, 0) * yii(0) + UiiInv(0, 1) * yii(1);
					y(i) = UiiInv(1, 0) * yii(0) + UiiInv(1, 1) * yii(1);
					--i; //skip past top row of block
				}
			}
		}

		//y = A^-1 x = Q U^-1 Q^T x
		void invmul(const blitz::Array<T, 2>& X, blitz::Array<T, 2>& Y) const
		{
			assert(arr_size_check(X) && arr_size_check(Y));
			lapack::gemm<T>(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1, AInv.data(), N, X.data(), N, 0, Y.data(), N);
		}

		void invmul(blitz::Array<T, 2>& X) const
		{
			assert(arr_size_check(X));
			blitz::Array<T, 2> ph(N);
			lapack::gemm<T>(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1, AInv.data(), N, X.data(), N, 0, ph.data(), N);
			X = ph;
		}

		void invmul(const blitz::Array<T, 1>& x, blitz::Array<T, 1>& y) const
		{
			assert(arr_size_check(x) && arr_size_check(y));
			lapack::gemv<T>(CblasRowMajor, CblasNoTrans, N, N, 1.0, AInv.data(), N, x.data(), 1, 0.0, y.data(), 1);
		}

		void invmul(blitz::Array<T, 1>& x) const
		{
			assert(arr_size_check(x));
			blitz::Array<T, 1> ph(N);
			lapack::gemv<T>(CblasRowMajor, CblasNoTrans, N, N, 1.0, AInv.data(), N, x.data(), 1, 0.0, ph.data(), 1);
			x = ph;
		}

		void printEigVals() const
		{
			std::cout << "Eigenvalues - real component" << std::endl;
			std::cout << WR << std::endl;
			std::cout << "Eigenvalues - imaginary component" << std::endl;
			std::cout << WI << std::endl;
			std::cout << "Eigenvalues - magnitude" << std::endl;
			std::cout << WM << std::endl;
		}

		void printEigInfo() const
		{
			std::cout << "min/max real eig " << min_real() << ' ' << max_real() << std::endl;
			std::cout << "min/max img  eig " << min_imag() << ' ' << max_imag() << std::endl;
			std::cout << "smallest magn eig " << min_mag() << std::endl;
			std::cout << "largest magn eig  " << max_mag() << std::endl;
			std::cout << "ratio/cond " << cond_number() << std::endl;
		}

		//real eigenvalue statistics
		S min_real() const { return blitz::min(WR); }
		S max_real() const { return blitz::max(WR); }
		S min_abs_real() const { return blitz::min(blitz::abs(WR)); }
		S max_abs_real() const { return blitz::max(blitz::abs(WR)); }

		//imaginary eigenvalue statistics
		S min_imag() const { return blitz::min(WI); }
		S max_imag() const { return blitz::max(WI); }
		S min_abs_imag() const { return blitz::min(blitz::abs(WI)); }
		S max_abs_imag() const { return blitz::max(blitz::abs(WI)); }

		//magnitude eigenvalue statistics
		S min_mag() const { return blitz::min(WM); }	//smallest eigenvalue
		S max_mag() const { return blitz::max(WM); }	//largest eigenvalue
		int arg_min_mag() const { return util::tvArgMin(WM); } //index with smallest eigenvalue
		int arg_max_mag() const { return util::tvArgMax(WM); } //index with largest eigenvalue
		S cond_number() const { return max_mag() / min_mag(); } //matrix A condition number.

	};//end NonSymEigenFactor_Arr

	///Calculate the eigenvalues of a 2D square blitz::Array.
	///Not used for solves/inversion, eigenvalue calculation only.
	///Used in Arnodi Ritz value calculations.
	///There is an option to calculate the harmonic Ritz values of an upper hessenburg
	/// matrix H, must include the lower subdiagonal h_mn, see deflated_restart.
	template<typename T>
	struct Eigenvalues_Arr
	{
		using S = lapack::ScalarType_t<T>;
		blitz::Array<S, 1> WR, WI, WM;	///< Real and imaginary parts of eigvals + magnitude
		S tol, tolFactor;
		int N;
		bool harmonic_Ritz;

		Eigenvalues_Arr() {}
		Eigenvalues_Arr(const blitz::Array<T, 2>& A, bool harmonic_Ritz = false, T h_mn = 0.0,
			S tolFactor = 1.0) : tolFactor{ tolFactor }, harmonic_Ritz{ harmonic_Ritz }
		{
			assert(A.extent(0) == A.extent(1)); //square matrix
			N = A.extent(0);
			WR.resize(N); WI.resize(N); WM.resize(N);
			lapack_int sdim;
			blitz::Array<T, 2> vs(N);
			auto ACopy{ A.copy() };

			///If calculating harmonic Ritz values, 
			///Compute H_tilde = H_n + h_{m,n}^2 * H_n^{-T} * e_n * e_n^T.
			///The eigenvalues of H_tilde are the harmonic Ritz values. See deflated_restart.
			///H_n is the nxn square portion of H (i.e., A in this construction).
			///h_mn the bottom right sub-diagonal element of H. 
			///H_n^{-T} is the inverse of the transpose of H_n.
			///e_n is the n-th standard basis vector.
			if (harmonic_Ritz)
			{
				//Solve H_n^T * w = e_n for w (compute H_n^{-T} * e_n), via LU factorization.
				blitz::Array<T, 1> w(N); w = T(0);
				w(N - 1) = T(1);  // e_n
				{
					LUFactor_Arr<T> H_nT_LU(transpose(A));
					H_nT_LU.invmul(w);
				}

				//Form H_tilde = H_n + |h_sub|^2 * w * e_n^T
				//The second rhs term only modifies the last column of H_n
				//H_tilde is ACopy in this struct.
				T scale;
				if constexpr (is_complex_v<T>) scale = std::conj(h_mn) * h_mn;
				else scale = h_mn * h_mn;
				for (int i = 0; i < N; ++i)
					ACopy(i, N - 1) += scale * w(i);
			}

			//calculate eigenvalues of ACopy
			if constexpr (lapack::is_complex_v<T>)
			{
				blitz::Array<T, 1> W(N);
				lapack::gees<T>(LAPACK_ROW_MAJOR, 'N', 'N', nullptr, N, ACopy.data(), N, &sdim,
					W.data(), vs.data(), N);
				for (int i = 0; i < N; ++i)
				{
					WR(i) = W(i).real(); WI(i) = W(i).imag();
				}
			}
			else
				lapack::gees<T>(LAPACK_ROW_MAJOR, 'N', 'N', nullptr, N, ACopy.data(), N, &sdim,
					WR.data(), WI.data(), vs.data(), N);

			//magnitude of eigenvalues
			for (int i = 0; i < N; ++i)
				WM(i) = std::sqrt(WR(i) * WR(i) + WI(i) * WI(i));
		}

		blitz::Array<S, 1> real() const { return WR; }
		blitz::Array<S, 1> imag() const { return WI; }
		blitz::Array<S, 1> mag()  const { return WM; }

		void printEigVals() const
		{
			if (harmonic_Ritz) std::cout << "Harmonic Ritz values" << std::endl;
			std::cout << "Eigenvalues - real component" << std::endl;
			std::cout << WR << std::endl;
			std::cout << "Eigenvalues - imaginary component" << std::endl;
			std::cout << WI << std::endl;
			std::cout << "Eigenvalues - magnitude" << std::endl;
			std::cout << WM << std::endl;
		}

		void printEigInfo() const
		{
			if (harmonic_Ritz) std::cout << "Harmonic Ritz info" << std::endl;
			else std::cout << "Eigenvalue info" << std::endl;
			std::cout << "min/max real eig " << min_real() << ' ' << max_real() << std::endl;
			std::cout << "min/max img  eig " << min_imag() << ' ' << max_imag() << std::endl;
			std::cout << "smallest magn eig " << min_mag() << std::endl;
			std::cout << "largest magn eig  " << max_mag() << std::endl;
			std::cout << "ratio/cond " << cond_number() << std::endl;
		}

		//real eigenvalue statistics
		S min_real() const { return blitz::min(WR); }
		S max_real() const { return blitz::max(WR); }
		S min_abs_real() const { return blitz::min(blitz::abs(WR)); }
		S max_abs_real() const { return blitz::max(blitz::abs(WR)); }

		//imaginary eigenvalue statistics
		S min_imag() const { return blitz::min(WI); }
		S max_imag() const { return blitz::max(WI); }
		S min_abs_imag() const { return blitz::min(blitz::abs(WI)); }
		S max_abs_imag() const { return blitz::max(blitz::abs(WI)); }

		//magnitude eigenvalue statistics
		S min_mag() const { return blitz::min(WM); }	//smallest eigenvalue
		S max_mag() const { return blitz::max(WM); }	//largest eigenvalue
		int arg_min_mag() const { return util::tvArgMin(WM); } //index with smallest eigenvalue
		int arg_max_mag() const { return util::tvArgMax(WM); } //index with largest eigenvalue
		S cond_number() const { return max_mag() / min_mag(); } //matrix A condition number.

	}; //end Eigenvalues_Arr


	////////////////////////
	//Small solve
	////////////////////////

	///Solve Ax = b for x. A small, 1x1, 2x2, and 3x3 matrices.
	template<typename T, int N> requires (N <= 3)
	blitz::TinyVector<T, N> small_solve(const blitz::TinyMatrix<T, N, N>& A,
		const blitz::TinyVector<T, N>& b)
	{
		if constexpr (N == 1) return blitz::TinyVector<T, N>(b(0) / A(0, 0));
		else if constexpr (N == 2)
		{
			//A = [a b] , A^-1 = 1/(ad - bc)[d  -b] <-- different b from vector b
			//		[c d]											[-c  a]
			T det = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0); //Determinant of A
			return blitz::TinyVector<T, N>((A(1, 1) * b(0) - A(0, 1) * b(1)) / det,
				(-A(1, 0) * b(0) + A(0, 0) * b(1)) / det);
		}
		else if constexpr (N == 3)
		{
			//return A^-1 b
			return matVec(util::explicit_small_Inv<T, 3>(A), b);
		}
		else
		{
			std::cout << "N = " << N << " too large for small_solve, b returned" << std::endl;
			return b;
		}
	}

	///Solve AX = B for X. A,X,B small, 1x1, 2x2, and 3x3 matrices.
	///if right_solve, solve XA = B.
	template<typename T, int N> requires (N <= 3)
	blitz::TinyMatrix<T, N, N> small_solve(const blitz::TinyMatrix<T, N, N>& A,
		const blitz::TinyMatrix<T, N, N>& B, bool right_solve)
	{
		if constexpr (N == 1) return blitz::TinyMatrix<T, N, N>(B(0, 0) / A(0, 0));
		else if constexpr (N == 2)
		{
			//A = [a b] , A^-1 = 1/(ad - bc)[d  -b]
			//		[c d]											[-c  a]
			T det = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0); //Determinant of A
			blitz::TinyMatrix<T, N, N> X;
			if (!right_solve) //left inverse
			{
				X(0, 0) = A(1, 1) * B(0, 0) - A(0, 1) * B(1, 0);
				X(0, 1) = A(1, 1) * B(0, 1) - A(0, 1) * B(1, 1);
				X(1, 0) = -A(1, 0) * B(0, 0) + A(0, 0) * B(1, 0);
				X(1, 1) = -A(1, 0) * B(0, 1) + A(0, 0) * B(1, 1);
			}
			else //right inverse
			{
				X(0, 0) = B(0,0) * A(1, 1) - B(0, 1) * A(1, 0);
				X(0, 1) = -B(0, 0) * A(0, 1) + B(0, 1) * A(0, 0);
				X(1, 0) = B(1, 0) * A(1, 1) - B(1, 1) * A(1, 0);
				X(1, 1) = -B(1, 0) * A(0, 1) + B(1, 1) * A(0, 0);
			}
			X /= det;
			return X;
		}
		else if constexpr (N == 3)
		{
			//return A^-1 B
			if(!right_solve) return matMat(util::explicit_small_Inv<T, 3>(A), B);
			else return matMat(B, util::explicit_small_Inv<T, 3>(A));
		}
		else
		{
			std::cout << "N = " << N << " too large for small_solve, B returned" << std::endl;
			return B;
		}
	}

	///Solve Ax = b for x. A small, 1x1, 2x2, and 3x3 matrices.
	///For symmetric matrix A stored as ((N + 1)* N) / 2 length TinyVector
	///spanning the LOWER triangular portion of a matrix.
	template<typename T, int N> requires (N <= 3)
	blitz::TinyVector<T, N> small_solve(const blitz::TinyVector<T, ((N + 1)* N) / 2>& A,
		const blitz::TinyVector<T, N>& b)
	{
		if constexpr (N == 1) return blitz::TinyVector<T, N>(b(0) / A(0));
		else if constexpr (N == 2)
		{
			//A = [a b] , A^-1 = 1/(ad - bc)[d  -b] <-- different b from vector b
			//		[c d]											[-c  a]
			T det = A(0) * A(2) - A(1) * A(1); //Determinant of A
			return blitz::TinyVector<T, N>((A(2) * b(0) - A(1) * b(1)) / det,
				(-A(1) * b(0) + A(0) * b(1)) / det);
		}
		else if constexpr (N == 3)
		{
			//return A^-1 b
			return matVec(util::explicit_small_Inv<T, 3>(A), b);
		}
		else
		{
			std::cout << "N = " << N << " too large for small_solve, b returned" << std::endl;
			return b;
		}
	}

	///Solve AX = B for X. A,X,B small, 1x1, 2x2, and 3x3 matrices.
	///For symmetric matrix A stored as ((N + 1)* N) / 2 length TinyVector
	///spanning the LOWER triangular portion of a matrix. 
	template<typename T, int N> requires (N <= 3)
	blitz::TinyMatrix<T, N, N> small_solve(const blitz::TinyVector <T, ((N + 1)* N) / 2>& A,
		const blitz::TinyMatrix<T, N, N>& B, bool right_solve)
	{
		if constexpr (N == 1) return blitz::TinyMatrix<T, N, N>(B(0, 0) / A(0));
		else if constexpr (N == 2)
		{
			//A = [a b] , A^-1 = 1/(ad - bc)[d  -b]
			//		[c d]											[-c  a]
			T det = A(0) * A(2) - A(1) * A(1); //Determinant of A
			blitz::TinyMatrix<T, N, N> X;
			if (!right_solve) //left inverse
			{
				X(0, 0) = A(2) * B(0, 0) - A(1) * B(1, 0);
				X(0, 1) = A(2) * B(0, 1) - A(1) * B(1, 1);
				X(1, 0) = -A(1) * B(0, 0) + A(0) * B(1, 0);
				X(1, 1) = -A(1) * B(0, 1) + A(0) * B(1, 1);
			}
			else //right inverse
			{
				X(0, 0) = B(0, 0) * A(2) - B(0, 1) * A(1);
				X(0, 1) = -B(0, 0) * A(1) + B(0, 1) * A(0);
				X(1, 0) = B(1, 0) * A(2) - B(1, 1) * A(1);
				X(1, 1) = -B(1, 0) * A(1) + B(1, 1) * A(0);
			}
			X /= det;
			return X;
		}
		else if constexpr (N == 3)
		{
			//return A^-1 B
			if(!right_solve) return matMat(util::explicit_small_Inv<T, 3>(A), B);
			else  return matMat(B, util::explicit_small_Inv<T, 3>(A));
		}
		else
		{
			std::cout << "N = " << N << " too large for small_solve, B returned" << std::endl;
			return B;
		}
	}

}//end namespace BlockSp::dense

#endif //end BLOCKSP_DENSE_SOLVERS_HPP