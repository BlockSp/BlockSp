#ifndef BLOCKSP_LAPACK_HPP
#define BLOCKSP_LAPACK_HPP

#include <mkl.h>
#include <mkl_lapack.h>
#include <complex>

///blocksp_lapack.hpp provides templated access to the blas/lapack/mkl routines used in BlockSp.
///See intel's mkl documention for more details. 
///https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/overview.html \n
///For floating point types: singles, doubles, single complex, double complex.
///Includes helper concepts for complex/scalar type constraints.

//Matrix-matrix and matrix-vector multiply
//gemm - General matrix-matrix multiplication.
//gemv - General matrix-vector multiplication.

//Matrix scale/transpose
//imatcopy - In place matrix scale/transpose
//omatcopy - Other matrix copy/scale/transpose.

//QR solver for least squares  
//gels  - solves over/under determined linear systems, QR factorization.
//geqrf - Computes the QR factorization of a general m by n matrix. 
//orgqr - Generates the real orthogonal matrix Q of the QR factorization formed by geqrf.
//ungqr - Generates the complex unitary matrix Q of the QR factorization formed by geqrf.  

//LU and Cholesky factorization and solver 
//getrf - Computes the LU factorization of a general m by n matrix.
//getrs - Solves a system of linear equations with an LU matrix.
//pptrf - Computes the Cholesky factorization of a sym pos def matrix (ondensed vector).
//pptrs - Solves a system of linear equations with an Cholesky matrix (condensed vector).
//potrf - Computes the Cholesky factorization of a sym pos def matrix (full matrix).
//potrs - Solves a system of linear equations with an Cholesky matrix (full matrix).
//trtrs - Solves a system of linear equations with a triangular coefficient matrix

//Symmetric eigenvalue solvers
//sytrd - Reduces a real symmetric matrix to tridiagonal form. 
//sterf - Computes all eigenvalues of a real symmetric tridiagonal matrix.
//syevr - Computes eigenvalues and eigenvectors of a real symmetric matrix.

//Nonsymmetric eigenvalue solvers 
//gehrd - Reduces a general matrix to upper Hessenberg form.
//orghr - Generates the real orthogonal matrix Q determined by gehrd.
//unghr - Generates the complex unitary matrix Q determined by gehrd. 
//hseqr - Computes all eigenvalues and the Schur factorization of matrix reduced to Hessenberg form.
//gees	- Computes the eigenvalues and Schur factorization of a general matrix
//trsen - Reorders the Schur factorization of a matrix
//ggev  - Computes the generalized eigenvalues, and the generalized eigenvectors for a pair of nonsymmetric matrices.

//Matrix norms
//lange - Returns the 1-norm, Frobenius norm, or inf-norm of a general rectangular matrix.

/*
if constexpr (std::is_same<T, float>::value)
else if constexpr (std::is_same<T, double>::value)
else if constexpr (std::is_same<T, std::complex<float>>::value)
else if constexpr (std::is_same<T, std::complex<double>>::value)
else {
	std::cout << "is not defined for type " << typeid(T).name() << std::endl;
	std::terminate();
}
//*/

///blocksp_lapack.hpp provides templated access to the blas/lapack/mkl routines used in BlockSp.
namespace BlockSp::lapack
{
	//////////////////////
	//Helper concepts

	///A concept that requires type T be lapack compatible.
	///Accepted types: float, double, std::complex<float>, std::complex<double>.
	///The use of concepts requires C++20 or later.
	template <typename T>
	concept is_lapack_type = std::is_same<T, float>::value || std::is_same<T, double>::value ||
		std::is_same<T, std::complex<float>>::value || std::is_same<T, std::complex<double>>::value;

	///Check if template type is complex.
	template <typename T>
	struct is_complex : std::false_type {};
	template <std::floating_point T>
	struct is_complex<std::complex<T>> : std::true_type {};
	template<typename T>
	inline constexpr bool is_complex_v = is_complex<T>::value;

	///Extract value type from std::complex.
	//Primary template - for non-complex types, S = T.
	template<typename T>
	struct ScalarType { using type = T; };
	// Specialization for std::complex<U> - extract U.
	template<typename U>
	struct ScalarType<std::complex<U>> { using type = U; };
	template<typename T>
	using ScalarType_t = typename ScalarType<T>::type;

	////////////////////////
	//LAPACKE routines

	///cblas_?gemm
	///General matrix-matrix multiplication.
	template<typename T> requires is_lapack_type<T>
	void gemm(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
		const MKL_INT m, const MKL_INT n, const MKL_INT k, const T alpha, const T* a,
		const MKL_INT lda, const T* b, const MKL_INT ldb, const T beta, T* c, const MKL_INT ldc)
	{
		if constexpr (std::is_same<T, float>::value)
			cblas_sgemm(Layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
		else if constexpr (std::is_same<T, double>::value)
			cblas_dgemm(Layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
		else if constexpr (std::is_same<T, std::complex<float>>::value)
			cblas_cgemm(Layout, transa, transb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
		else if constexpr (std::is_same<T, std::complex<double>>::value)
			cblas_zgemm(Layout, transa, transb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
		else {
			std::cout << "gemm is not defined for type " << typeid(T).name() << std::endl;
			std::terminate();
		}
	};

	///cblas_?gemv
	///General matrix-vector multiplication.
	template<typename T> requires is_lapack_type<T>
	void gemv(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE trans, const MKL_INT m, const MKL_INT n,
		const T alpha, const T* a, const MKL_INT lda, const T* x, const MKL_INT incx,
		const T beta, T* y, const MKL_INT incy)
	{
		if constexpr (std::is_same<T, float>::value)
			cblas_sgemv(Layout, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
		else if constexpr (std::is_same<T, double>::value)
			cblas_dgemv(Layout, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
		else if constexpr (std::is_same<T, std::complex<float>>::value)
			cblas_cgemv(Layout, trans, m, n, &alpha, a, lda, x, incx, &beta, y, incy);
		else if constexpr (std::is_same<T, std::complex<double>>::value)
			cblas_zgemv(Layout, trans, m, n, &alpha, a, lda, x, incx, &beta, y, incy);
		else {
			std::cout << "gemv is not defined for type " << typeid(T).name() << std::endl;
			std::terminate();
		}
	};

	///mkl_?imatcopy
	///In place matrix scale/transpose.
	template<typename T> requires is_lapack_type<T>
	void imatcopy(char ordering, char trans, size_t rows, size_t cols, const T alpha,
		T* AB, size_t lda, size_t ldb)
	{
		if constexpr (std::is_same<T, float>::value)
			mkl_simatcopy(ordering, trans, rows, cols, alpha, AB, lda, ldb);
		else if constexpr (std::is_same<T, double>::value)
			mkl_dimatcopy(ordering, trans, rows, cols, alpha, AB, lda, ldb);
		else if constexpr (std::is_same<T, std::complex<float>>::value)
		{
			const MKL_Complex8* alpha_8 = reinterpret_cast<const MKL_Complex8*>(&alpha);
			mkl_cimatcopy(ordering, trans, rows, cols, *alpha_8, reinterpret_cast<MKL_Complex8*>(AB),
				lda, ldb);
		}
		else if constexpr (std::is_same<T, std::complex<double>>::value)
		{
			const MKL_Complex16* alpha_16 = reinterpret_cast<const MKL_Complex16*>(&alpha);
			mkl_zimatcopy(ordering, trans, rows, cols, *alpha_16, reinterpret_cast<MKL_Complex16*>(AB),
				lda, ldb);
		}
		else {
			std::cout << "imatcopy is not defined for type " << typeid(T).name() << std::endl;
			std::terminate();
		}
	};

	///mkl_?omatcopy
	///Other matrix copy/scale/transpose.
	template<typename T> requires is_lapack_type<T>
	void omatcopy(char ordering, char trans, size_t rows, size_t cols, const T alpha,
		const T* A, size_t lda, T* B, size_t ldb)
	{
		if constexpr (std::is_same<T, float>::value)
			mkl_somatcopy(ordering, trans, rows, cols, alpha, A, lda, B, ldb);
		else if constexpr (std::is_same<T, double>::value)
			mkl_domatcopy(ordering, trans, rows, cols, alpha, A, lda, B, ldb);
		else if constexpr (std::is_same<T, std::complex<float>>::value)
		{
			const MKL_Complex8* alpha_8 = reinterpret_cast<const MKL_Complex8*>(&alpha);
			mkl_comatcopy(ordering, trans, rows, cols, *alpha_8, reinterpret_cast<const MKL_Complex8*>(A),
				lda, reinterpret_cast<MKL_Complex8*>(B), ldb);
		}
		else if constexpr (std::is_same<T, std::complex<double>>::value)
		{
			const MKL_Complex16* alpha_16 = reinterpret_cast<const MKL_Complex16*>(&alpha);
			mkl_zomatcopy(ordering, trans, rows, cols, *alpha_16, reinterpret_cast<const MKL_Complex16*>(A),
				lda, reinterpret_cast<MKL_Complex16*>(B), ldb);
		}
		else {
			std::cout << "omatcopy is not defined for type " << typeid(T).name() << std::endl;
			std::terminate();
		}
	};

	///LAPACKE_?gels
	///Solve overdetermined or underdetermined linear system with full rank matrix.
	///Uses QR or LQ factorization.
	///a overwritten by factorization, b overwritten by solution.
	template<typename T> requires is_lapack_type<T>
	void gels(int matrix_layout, char trans, lapack_int m, lapack_int n, lapack_int nrhs, T* a,
		lapack_int lda, T* b, lapack_int ldb)
	{
		if constexpr (std::is_same<T, float>::value)
			LAPACKE_sgels(matrix_layout, trans, m, n, nrhs, a, lda, b, ldb);
		else if constexpr (std::is_same<T, double>::value)
			LAPACKE_dgels(matrix_layout, trans, m, n, nrhs, a, lda, b, ldb);
		else if constexpr (std::is_same<T, std::complex<float>>::value)
			LAPACKE_cgels(matrix_layout, trans, m, n, nrhs, reinterpret_cast<MKL_Complex8*>(a),
				lda, reinterpret_cast<MKL_Complex8*>(b), ldb);
		else if constexpr (std::is_same<T, std::complex<double>>::value)
			LAPACKE_zgels(matrix_layout, trans, m, n, nrhs, reinterpret_cast<MKL_Complex16*>(a),
				lda, reinterpret_cast<MKL_Complex16*>(b), ldb);
		else {
			std::cout << "gels is not defined for type " << typeid(T).name() << std::endl;
			std::terminate();
		}
	};

	///LAPACKE_?geqrf
	///Computes the QR factorization of a general m by n matrix.
	template<typename T> requires is_lapack_type<T>
	void geqrf(int matrix_layout, lapack_int m, lapack_int n, T* a, lapack_int lda, T* tau)
	{
		if constexpr (std::is_same<T, float>::value)
			LAPACKE_sgeqrf(matrix_layout, m, n, a, lda, tau);
		else if constexpr (std::is_same<T, double>::value)
			LAPACKE_dgeqrf(matrix_layout, m, n, a, lda, tau);
		else if constexpr (std::is_same<T, std::complex<float>>::value)
			LAPACKE_cgeqrf(matrix_layout, m, n, reinterpret_cast<MKL_Complex8*>(a),
				lda, reinterpret_cast<MKL_Complex8*>(tau));
		else if constexpr (std::is_same<T, std::complex<double>>::value)
			LAPACKE_zgeqrf(matrix_layout, m, n, reinterpret_cast<MKL_Complex16*>(a),
				lda, reinterpret_cast<MKL_Complex16*>(tau));
		else {
			std::cout << "geqrf is not defined for type " << typeid(T).name() << std::endl;
			std::terminate();
		}
	}

	///LAPACKE_?orgqr
	///Generates the real orthogonal matrix Q of the QR factorization formed by geqrf.
	template<typename T> requires is_lapack_type<T>
	void orgqr(int matrix_layout, lapack_int m, lapack_int n, lapack_int k, T* a, lapack_int lda, const T* tau)
	{
		if constexpr (std::is_same<T, float>::value)
			LAPACKE_sorgqr(matrix_layout, m, n, k, a, lda, tau);
		else if constexpr (std::is_same<T, double>::value)
			LAPACKE_dorgqr(matrix_layout, m, n, k, a, lda, tau);
	}

	///LAPACKE_?ungqr
	///Generates the complex unitary matrix Q of the QR factorization formed by geqrf.
	template<typename T> requires is_lapack_type<T>
	void ungqr(int matrix_layout, lapack_int m, lapack_int n, lapack_int k, T* a, lapack_int lda, const T* tau)
	{
		if constexpr (std::is_same<T, std::complex<float>>::value)
			LAPACKE_cungqr(matrix_layout, m, n, k, reinterpret_cast<MKL_Complex8*>(a),
				lda, reinterpret_cast<const MKL_Complex8*>(tau));
		else if constexpr (std::is_same<T, std::complex<double>>::value)
			LAPACKE_zungqr(matrix_layout, m, n, k, reinterpret_cast<MKL_Complex16*>(a),
				lda, reinterpret_cast<const MKL_Complex16*>(tau));

	}

	///LAPACKE_?getrf
	///Computes the LU factorization of a general m by n matrix. 
	///a overwritten by L and U. Used with getrs to solve linear systems.
	template<typename T> requires is_lapack_type<T>
	void getrf(int matrix_layout, lapack_int m, lapack_int n, T* a, lapack_int lda, lapack_int* ipiv)
	{
		if constexpr (std::is_same<T, float>::value)
			LAPACKE_sgetrf(matrix_layout, m, n, a, lda, ipiv);
		else if constexpr (std::is_same<T, double>::value)
			LAPACKE_dgetrf(matrix_layout, m, n, a, lda, ipiv);
		else if constexpr (std::is_same<T, std::complex<float>>::value)
			LAPACKE_cgetrf(matrix_layout, m, n, reinterpret_cast<MKL_Complex8*>(a), lda, ipiv);
		else if constexpr (std::is_same<T, std::complex<double>>::value)
			LAPACKE_zgetrf(matrix_layout, m, n, reinterpret_cast<MKL_Complex16*>(a), lda, ipiv);
		else {
			std::cout << "getrf is not defined for type " << typeid(T).name() << std::endl;
			std::terminate();
		}
	};

	///LAPACKE_?getrs
	///Solves a system of linear equations with an LU-factored square coefficient matrix.
	///Call getrf first to compute LU factorization of A.
	template<typename T> requires is_lapack_type<T>
	void getrs(int matrix_layout, char trans, lapack_int n, lapack_int nrhs, const T* a,
		lapack_int lda, const lapack_int* ipiv, T* b, lapack_int ldb)
	{
		if constexpr (std::is_same<T, float>::value)
			LAPACKE_sgetrs(matrix_layout, trans, n, nrhs, a, lda, ipiv, b, ldb);
		else if constexpr (std::is_same<T, double>::value)
			LAPACKE_dgetrs(matrix_layout, trans, n, nrhs, a, lda, ipiv, b, ldb);
		else if constexpr (std::is_same<T, std::complex<float>>::value)
			LAPACKE_cgetrs(matrix_layout, trans, n, nrhs, reinterpret_cast<const MKL_Complex8*>(a),
				lda, ipiv, reinterpret_cast<MKL_Complex8*>(b), ldb);
		else if constexpr (std::is_same<T, std::complex<double>>::value)
			LAPACKE_zgetrs(matrix_layout, trans, n, nrhs, reinterpret_cast<const MKL_Complex16*>(a),
				lda, ipiv, reinterpret_cast<MKL_Complex16*>(b), ldb);
		else {
			std::cout << "getrs is not defined for type " << typeid(T).name() << std::endl;
			std::terminate();
		}
	};

	///LAPACKE_?pptrf
	///Computes the Cholesky factorization of a symmetric (Hermitian) positive definite matrix.
	///In packed format (vector of length N*(N+1)/2). 
	///ap overwritten by L or U. Used with pptrf to solve linear systems.
	template<typename T> requires is_lapack_type<T>
	void pptrf(int matrix_layout, char uplo, lapack_int n, T* ap)
	{
		if constexpr (std::is_same<T, float>::value)
			LAPACKE_spptrf(matrix_layout, uplo, n, ap);
		else if constexpr (std::is_same<T, double>::value)
			LAPACKE_dpptrf(matrix_layout, uplo, n, ap);
		else if constexpr (std::is_same<T, std::complex<float>>::value)
			LAPACKE_cpptrf(matrix_layout, uplo, n, reinterpret_cast<MKL_Complex8*>(ap));
		else if constexpr (std::is_same<T, std::complex<double>>::value)
			LAPACKE_zpptrf(matrix_layout, uplo, n, reinterpret_cast<MKL_Complex16*>(ap));
		else {
			std::cout << "pptrf is not defined for type " << typeid(T).name() << std::endl;
			std::terminate();
		}
	}

	///LAPACKE_?pptrs
	///Solves a symmetric positive definite linear system with a Cholesky factored matrix.
	///Call potrs first to compute Cholesky factorization of A.
	template<typename T> requires is_lapack_type<T>
	void pptrs(int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
		const T* ap, T* b, lapack_int ldb)
	{
		if constexpr (std::is_same<T, float>::value)
			LAPACKE_spptrs(matrix_layout, uplo, n, nrhs, ap, b, ldb);
		else if constexpr (std::is_same<T, double>::value)
			LAPACKE_dpptrs(matrix_layout, uplo, n, nrhs, ap, b, ldb);
		else if constexpr (std::is_same<T, std::complex<float>>::value)
			LAPACKE_cpptrs(matrix_layout, uplo, n, nrhs, reinterpret_cast<MKL_Complex8*>(ap),
				reinterpret_cast<MKL_Complex8*>(b), ldb);
		else if constexpr (std::is_same<T, std::complex<double>>::value)
			LAPACKE_zpptrs(matrix_layout, uplo, n, nrhs, reinterpret_cast<MKL_Complex16*>(ap),
				reinterpret_cast<MKL_Complex16*>(b), ldb);
		else {
			std::cout << "pptrs is not defined for type " << typeid(T).name() << std::endl;
			std::terminate();
		}
	};

	///LAPACKE_?potrf
	///Computes the Cholesky factorization of a symmetric (Hermitian) positive definite matrix.
	///In full matrix format.
	///a overwritten by L or U. Use with potrs to solve linear systems.
	template<typename T> requires is_lapack_type<T>
	void potrf(int matrix_layout, char uplo, lapack_int n, T* a, lapack_int lda)
	{
		if constexpr (std::is_same<T, float>::value)
			LAPACKE_spotrf(matrix_layout, uplo, n, a, lda);
		else if constexpr (std::is_same<T, double>::value)
			LAPACKE_dpotrf(matrix_layout, uplo, n, a, lda);
		else if constexpr (std::is_same<T, std::complex<float>>::value)
			LAPACKE_cpotrf(matrix_layout, uplo, n, reinterpret_cast<MKL_Complex8*>(a), lda);
		else if constexpr (std::is_same<T, std::complex<double>>::value)
			LAPACKE_zpotrf(matrix_layout, uplo, n, reinterpret_cast<MKL_Complex16*>(a), lda);
		else {
			std::cout << "potrf is not defined for type " << typeid(T).name() << std::endl;
			std::terminate();
		}
	}

	///LAPACKE_?potrs
	///Solves a symmetric positive definite linear system with a Cholesky factored matrix.
	///Call potrf first to compute Cholesky factorization of A.
	template<typename T> requires is_lapack_type<T>
	void potrs(int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
		const T* a, lapack_int lda, T* b, lapack_int ldb)
	{
		if constexpr (std::is_same<T, float>::value)
			LAPACKE_spotrs(matrix_layout, uplo, n, nrhs, a, lda, b, ldb);
		else if constexpr (std::is_same<T, double>::value)
			LAPACKE_dpotrs(matrix_layout, uplo, n, nrhs, a, lda, b, ldb);
		else if constexpr (std::is_same<T, std::complex<float>>::value)
			LAPACKE_cpotrs(matrix_layout, uplo, n, nrhs, reinterpret_cast<MKL_Complex8*>(a),
				lda, reinterpret_cast<MKL_Complex8*>(b), ldb);
		else if constexpr (std::is_same<T, std::complex<double>>::value)
			LAPACKE_zpotrs(matrix_layout, uplo, n, nrhs, reinterpret_cast<MKL_Complex16*>(a),
				lda, reinterpret_cast<MKL_Complex16*>(b), ldb);
		else {
			std::cout << "potrs is not defined for type " << typeid(T).name() << std::endl;
			std::terminate();
		}
	};

	///LAPACKE_?trtrs
	///Solves a system of linear equations with a triangular coefficient matrix
	template<typename T> requires is_lapack_type<T>
	void trtrs(int matrix_layout, char uplo, char trans, char diag, lapack_int n, 
		lapack_int nrhs, const T* a, lapack_int lda, T* b, lapack_int ldb)
	{

		if constexpr (std::is_same<T, float>::value)
			LAPACKE_strtrs(matrix_layout, uplo, trans, diag, n, nrhs, a, lda, b, ldb);
		else if constexpr (std::is_same<T, double>::value)
			LAPACKE_dtrtrs(matrix_layout, uplo, trans, diag, n, nrhs, a, lda, b, ldb);
		else if constexpr (std::is_same<T, std::complex<float>>::value)
			LAPACKE_ctrtrs(matrix_layout, uplo, trans, diag, n, nrhs, a, lda, b, ldb);
		else if constexpr (std::is_same<T, std::complex<double>>::value)
			LAPACKE_ztrtrs(matrix_layout, uplo, trans, diag, n, nrhs, a, lda, b, ldb);
		else {
			std::cout << "trtrs is not defined for type " << typeid(T).name() << std::endl;
			std::terminate();
		}
	}

	///LAPACKE_?sytrd
	///Reduces a real symmetric matrix to tridiagonal form. 
	///Used with sytrd to compute eigenvalues.
	template<typename T> requires is_lapack_type<T>
	void sytrd(int matrix_layout, char uplo, lapack_int n, T* a, lapack_int lda,
		T* d, T* e, T* tau)
	{
		if constexpr (std::is_same<T, float>::value)
			LAPACKE_ssytrd(matrix_layout, uplo, n, a, lda, d, e, tau);
		else if constexpr (std::is_same<T, double>::value)
			LAPACKE_dsytrd(matrix_layout, uplo, n, a, lda, d, e, tau);
		else {
			std::cout << "sytrd is not defined for type " << typeid(T).name() << std::endl;
			std::terminate();
		}
	};

	///LAPACKE_?sterf
	///Computes all eigenvalues of a real symmetric tridiagonal matrix using QR algorithm. 
	///Use with sytrd.
	template<typename T> requires is_lapack_type<T>
	void sterf(lapack_int n, T* d, T* e)
	{
		if constexpr (std::is_same<T, float>::value)
			LAPACKE_ssterf(n, d, e);
		else if constexpr (std::is_same<T, double>::value)
			LAPACKE_dsterf(n, d, e);
		else {
			std::cout << "sterf is not defined for type " << typeid(T).name() << std::endl;
			std::terminate();
		}
	};

	///LAPACKE_?syevr
	///Computes selected eigenvalues and, optionally, eigenvectors of a real symmetric matrix.
	///a is overwritten on either the upper or lower triangular section and diagonal.
	template<typename T> requires is_lapack_type<T>
	void syevr(int matrix_layout, char jobz, char range, char uplo, lapack_int n, T* a, lapack_int lda,
		T vl, T vu, lapack_int il, lapack_int iu, T abstol, lapack_int* m,
		T* w, T* z, lapack_int ldz, lapack_int* isuppz)
	{
		if constexpr (std::is_same<T, float>::value)
			LAPACKE_ssyevr(matrix_layout, jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz, isuppz);
		else if constexpr (std::is_same<T, double>::value)
			LAPACKE_dsyevr(matrix_layout, jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz, isuppz);
		else {
			std::cout << "syevr is not defined for type " << typeid(T).name() << std::endl;
			std::terminate();
		}
	};

	///LAPACKE_?gehrd 
	///Reduces a general matrix to upper Hessenberg form.
	///a replaced by upper Hessenburg matrix H along with reflectors on below subdiagonal.
	///Use with gehrd and hseqr to compute eigenvalues of asymmetric matrices.
	template<typename T> requires is_lapack_type<T>
	void gehrd(int matrix_layout, lapack_int n, lapack_int ilo, lapack_int ihi, T* a, lapack_int lda, T* tau)
	{
		if constexpr (std::is_same<T, float>::value)
			LAPACKE_sgehrd(matrix_layout, n, ilo, ihi, a, lda, tau);
		else if constexpr (std::is_same<T, double>::value)
			LAPACKE_dgehrd(matrix_layout, n, ilo, ihi, a, lda, tau);
		else if constexpr (std::is_same<T, std::complex<float>>::value)
			LAPACKE_cgehrd(matrix_layout, n, ilo, ihi, reinterpret_cast<MKL_Complex8*>(a),
				lda, reinterpret_cast<MKL_Complex8*>(tau));
		else if constexpr (std::is_same<T, std::complex<double>>::value)
			LAPACKE_zgehrd(matrix_layout, n, ilo, ihi, reinterpret_cast<MKL_Complex16*>(a),
				lda, reinterpret_cast<MKL_Complex16*>(tau));
		else {
			std::cout << "gehrd is not defined for type " << typeid(T).name() << std::endl;
			std::terminate();
		}
	};

	///LAPACKE_?orghr
	///Generates the real orthogonal matrix Q determined by ? gehrd.
	///a overwritten by matrix Q.
	template<typename T> requires is_lapack_type<T>
	void orghr(int matrix_layout, lapack_int n, lapack_int ilo, lapack_int ihi, T* a, lapack_int lda, const T* tau)
	{
		if constexpr (std::is_same<T, float>::value)
			LAPACKE_sorghr(matrix_layout, n, ilo, ihi, a, lda, tau);
		else if constexpr (std::is_same<T, double>::value)
			LAPACKE_dorghr(matrix_layout, n, ilo, ihi, a, lda, tau);
		else {
			std::cout << "orghr is not defined for type " << typeid(T).name() << std::endl;
			std::terminate();
		}
	};

	///LAPACKE_?unghr
	///Generates the complex unitary matrix Q determined by ? gehrd.
	///a overwritten by matrix Q.
	template<typename T> requires is_lapack_type<T>
	void unghr(int matrix_layout, lapack_int n, lapack_int ilo, lapack_int ihi, T* a, lapack_int lda, const T* tau)
	{
		if constexpr (std::is_same<T, std::complex<float>>::value)
			LAPACKE_cunghr(matrix_layout, n, ilo, ihi, reinterpret_cast<MKL_Complex8*>(a),
				lda, reinterpret_cast<const MKL_Complex8*>(tau));
		else if constexpr (std::is_same<T, std::complex<double>>::value)
			LAPACKE_zunghr(matrix_layout, n, ilo, ihi, reinterpret_cast<MKL_Complex16*>(a),
				lda, reinterpret_cast<const MKL_Complex16*>(tau));
		else {
			std::cout << "unghr is not defined for type " << typeid(T).name() << std::endl;
			std::terminate();
		}
	};

	///LAPACKE_?hseqr
	///Computes all eigenvalues and (optionally) the Schur factorization of 
	/// matrix reduced to Hessenberg form.
	template<typename T> requires is_lapack_type<T>
	void hseqr(int matrix_layout, char job, char compz, lapack_int n, lapack_int ilo, lapack_int ihi,
		T* h, lapack_int ldh, ScalarType_t<T>* wr, ScalarType_t<T>* wi, T* z, lapack_int ldz)
	{
		if constexpr (std::is_same<T, float>::value)
			LAPACKE_shseqr(matrix_layout, job, compz, n, ilo, ihi, h, ldh, wr, wi, z, ldz);
		else if constexpr (std::is_same<T, double>::value)
			LAPACKE_dhseqr(matrix_layout, job, compz, n, ilo, ihi, h, ldh, wr, wi, z, ldz);
		else {
			std::cout << "hseqr is not defined for type " << typeid(T).name() << std::endl;
			std::terminate();
		}
	};
	template<typename T> requires is_lapack_type<T>&& is_complex_v<T>
	void hseqr(int matrix_layout, char job, char compz, lapack_int n, lapack_int ilo, lapack_int ihi,
		T* h, lapack_int ldh, T* w, T* z, lapack_int ldz)
	{
		if constexpr (std::is_same<T, std::complex<float>>::value)
			LAPACKE_chseqr(matrix_layout, job, compz, n, ilo, ihi, reinterpret_cast<MKL_Complex8*>(h),
				ldh, reinterpret_cast<MKL_Complex8*>(w), reinterpret_cast<MKL_Complex8*>(z), ldz);
		else if constexpr (std::is_same<T, std::complex<double>>::value)
			LAPACKE_zhseqr(matrix_layout, job, compz, n, ilo, ihi, reinterpret_cast<MKL_Complex16*>(h),
				ldh, reinterpret_cast<MKL_Complex16*>(w), reinterpret_cast<MKL_Complex16*>(z), ldz);
		else {
			std::cout << "hseqr is not defined for type " << typeid(T).name() << std::endl;
			std::terminate();
		}
	}

	///LAPACKE_?gees
	///Computes the eigenvalues and Schur factorization of a general matrix.
	template<typename T> requires is_lapack_type<T>
	void gees(int matrix_layout, char jobvs, char sort, char* select,
		lapack_int n, T* a, lapack_int lda, lapack_int* sdim, T* wr, T* wi,
		T* vs, lapack_int ldvs)
	{
		if constexpr (std::is_same<T, float>::value)
			LAPACKE_sgees(matrix_layout, jobvs, sort, reinterpret_cast<LAPACK_S_SELECT2>(select), n, a, lda, sdim, wr, wi, vs, ldvs);
		else if constexpr (std::is_same<T, double>::value)
			LAPACKE_dgees(matrix_layout, jobvs, sort, reinterpret_cast<LAPACK_D_SELECT2>(select), n, a, lda, sdim, wr, wi, vs, ldvs);
		else {
			std::cout << "gees is not defined for type " << typeid(T).name() << std::endl;
			std::terminate();
		}
	}
	template<typename T> requires is_lapack_type<T>
	void gees(int matrix_layout, char jobvs, char sort, char* select,
		lapack_int n, T* a, lapack_int lda, lapack_int* sdim, T* w,
		T* vs, lapack_int ldvs)
	{
		if constexpr (std::is_same<T, std::complex<float>>::value)
			LAPACKE_cgees(matrix_layout, jobvs, sort, reinterpret_cast<LAPACK_C_SELECT1>(select), n,
				reinterpret_cast<MKL_Complex8*>(a), lda, sdim, reinterpret_cast<MKL_Complex8*>(w),
				reinterpret_cast<MKL_Complex8*>(vs), ldvs);
		else if constexpr (std::is_same<T, std::complex<double>>::value)
			LAPACKE_zgees(matrix_layout, jobvs, sort, reinterpret_cast<LAPACK_Z_SELECT1>(select), n,
				reinterpret_cast<MKL_Complex16*>(a), lda, sdim, reinterpret_cast<MKL_Complex16*>(w),
				reinterpret_cast<MKL_Complex16*>(vs), ldvs);
		else {
			std::cout << "gees is not defined for type " << typeid(T).name() << std::endl;
			std::terminate();
		}
	}

	///LAPACKE_?trsen
	///Reorders the Schur factorization of a matrix
	template<typename T> requires is_lapack_type<T>
	void trsen(int matrix_layout, char job, char compq, const lapack_logical* select,
		lapack_int n, T* t, lapack_int ldt, T* q, lapack_int ldq,
		T* wr, T* wi, lapack_int* m, T* s, T* sep)
	{
		if constexpr (std::is_same<T, float>::value)
			LAPACKE_strsen(matrix_layout, job, compq, select, n, t, ldt, q, ldq, wr, wi, m, s, sep);
		else if constexpr (std::is_same<T, double>::value)
			LAPACKE_dtrsen(matrix_layout, job, compq, select, n, t, ldt, q, ldq, wr, wi, m, s, sep);
		else {
			std::cout << "trsen is not defined for type " << typeid(T).name() << std::endl;
			std::terminate();
		}
	}
	template<typename T> requires is_lapack_type<T>
	void trsen(int matrix_layout, char job, char compq, const lapack_logical* select,
		lapack_int n, T* t, lapack_int ldt, T* q, lapack_int ldq,
		T* w, lapack_int* m, ScalarType_t<T>* s, ScalarType_t<T>* sep)
	{
		if constexpr (std::is_same<T, std::complex<float>>::value)
			LAPACKE_ctrsen(matrix_layout, job, compq, select, n, reinterpret_cast<MKL_Complex8*>(t),
				ldt, reinterpret_cast<MKL_Complex8*>(q), ldq, reinterpret_cast<MKL_Complex8*>(w), m, s, sep);
		else if constexpr (std::is_same<T, std::complex<double>>::value)
			LAPACKE_ztrsen(matrix_layout, job, compq, select, n, reinterpret_cast<MKL_Complex16*>(t),
				ldt, reinterpret_cast<MKL_Complex16*>(q), ldq, reinterpret_cast<MKL_Complex16*>(w), m, s, sep);
		else {
			std::cout << "trsen is not defined for type " << typeid(T).name() << std::endl;
			std::terminate();
		}
	}

	///LAPACKE_?ggev
	///Computes the generalized eigenvalues, and the generalized eigenvectors for a pair of nonsymmetric matrices.
	template<typename T> requires is_lapack_type<T>
	void ggev(int matrix_layout, char jobvl, char jobvr, lapack_int n, T* a, lapack_int lda, T* b,
		lapack_int ldb, T* alphar, T* alphai, T* beta, T* vl, lapack_int ldvl, T* vr, lapack_int ldvr)
	{
		if constexpr (std::is_same<T, float>::value)
			LAPACKE_sggev(matrix_layout, jobvl, jobvr, n, a, lda, b, ldb, alphar, alphai, beta, vl, ldvl, vr, ldvr);
		else if constexpr (std::is_same<T, double>::value)
			LAPACKE_dggev(matrix_layout, jobvl, jobvr, n, a, lda, b, ldb, alphar, alphai, beta, vl, ldvl, vr, ldvr);
		else {
			std::cout << "ggev is not defined for type " << typeid(T).name() << std::endl;
			std::terminate();
		}
	}
	template<typename T> requires is_lapack_type<T>
	void ggev(int matrix_layout, char jobvl, char jobvr, lapack_int n, T* a, lapack_int lda,
		T* b, lapack_int ldb, T* alpha, T* beta, T* vl, lapack_int ldvl, T* vr, lapack_int ldvr)
	{
		if constexpr (std::is_same<T, std::complex<float>>::value)
			LAPACKE_cggev(matrix_layout, jobvl, jobvr, n, reinterpret_cast<MKL_Complex8*>(a), lda,
				reinterpret_cast<MKL_Complex8*>(b), ldb, reinterpret_cast<MKL_Complex8*>(alpha),
				reinterpret_cast<MKL_Complex8*>(beta), reinterpret_cast<MKL_Complex8*>(vl), ldvl,
				reinterpret_cast<MKL_Complex8*>(vr), ldvr);
		else if constexpr (std::is_same<T, std::complex<double>>::value)
			LAPACKE_zggev(matrix_layout, jobvl, jobvr, n, reinterpret_cast<MKL_Complex16*>(a), lda,
				reinterpret_cast<MKL_Complex16*>(b), ldb, reinterpret_cast<MKL_Complex16*>(alpha),
				reinterpret_cast<MKL_Complex16*>(beta), reinterpret_cast<MKL_Complex16*>(vl), ldvl,
				reinterpret_cast<MKL_Complex16*>(vr), ldvr);
		else {
			std::cout << "gees is not defined for type " << typeid(T).name() << std::endl;
			std::terminate();
		}
	}

	///LAPACKE_?lange 
	///Returns the 1-norm, Frobenius norm, or inf-norm of a general rectangular matrix.
	///set norm to
	///'M' or 'm' for largest absolute value of matrix a.
	///'1' or 'O' or 'o' for 1-norm.
	///'I' or 'i' for infinity-norm.
	///'F' or 'f' for Frobenius norm.
	template<typename T> requires is_lapack_type<T>
	ScalarType_t<T> lange(int matrix_layout, char norm, int m, int n, const T* a, int lda)
	{
		if constexpr (std::is_same<T, float>::value)
			return LAPACKE_slange(matrix_layout, norm, m, n, a, lda);
		if constexpr (std::is_same<T, double>::value)
			return LAPACKE_dlange(matrix_layout, norm, m, n, a, lda);
		else if constexpr (std::is_same<T, std::complex<float>>::value)
			return LAPACKE_clange(matrix_layout, norm, m, n, reinterpret_cast<const MKL_Complex8*>(a), lda);
		else if constexpr (std::is_same<T, std::complex<double>>::value)
			return LAPACKE_zlange(matrix_layout, norm, m, n, reinterpret_cast<const MKL_Complex16*>(a), lda);
		else {
			std::cout << "lang is not defined for type " << typeid(T).name() << std::endl;
			std::terminate();
		}
	}

}//end namespace BlockSp::lapack

#endif //end BLOCKSP_LAPACK_HPP