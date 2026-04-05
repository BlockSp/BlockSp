#ifndef BLOCKSP_DENSE_MULTIPLY_HPP
#define BLOCKSP_DENSE_MULTIPLY_HPP

#include <blitz/array.h>
#include "blocksp_lapack.hpp"
#include "blocksp_utility.hpp"

///blocksp_dense_multiply.hpp contains the dense linear algebra
/// multiplication functions used in BlockSp.
///These include dense matrix-matrix, matrix-vector products,
/// dense identity, and dense matrix transpose. 
/// 
///There are two versions of each: \n
/// - one is compile-time optimized, for TinyMatrices/TinyVectors. \n
/// - the other is for blitz::Arrays, the size of which is specified at run-time. 
namespace BlockSp::dense
{

	/////////////////////////////////
	//Dense matrix-vector multiply
	/////////////////////////////////

	constexpr int MATVEC_BLAS_THRESH{ 8 }; ///< The threshold between using naive nested loops (nnl)
	constexpr int MATMAT_BLAS_THRESH{ 8 }; ///  vs using Blas for dense matVec and matMat.
																				 ///  Testing indicates nnl is, in general, faster for small matrices below 
																				 ///  ~6x6-8x8 due to optimized loops over compile time constants.

	///Naive nested loop TinyMatrix-TinyVector product, use for small M, N.
	template<typename T, int M, int N>
	blitz::TinyVector<T, M> matVec_nnl(const blitz::TinyMatrix<T, M, N>& mat,
		const blitz::TinyVector<T, N>& vec)
	{
		if constexpr (M == 1 && N == 1) return blitz::TinyVector<T, M>(mat(0, 0) * vec(0));
		if constexpr (M == 1 && N == 2) return blitz::TinyVector<T, M>(mat(0, 0) * vec(0) + mat(0, 1) * vec(1));
		if constexpr (M == 2 && N == 1) return blitz::TinyVector<T, M>(mat(0, 0) * vec(0), mat(1, 0) * vec(0));
		if constexpr (M == 2 && N == 2)
			return blitz::TinyVector<T, M>(mat(0, 0) * vec(0) + mat(0, 1) * vec(1), mat(1, 0) * vec(0) + mat(1, 1) * vec(1));

		blitz::TinyVector<T, M> mv;
		for (int im = 0; im < M; ++im)
		{
			T val = 0;
			for (int in = 0; in < N; ++in)
				val += mat(im, in) * vec(in);
			mv(im) = val;
		}
		return mv;
	}

	///TinyMatrix-TinyVector product.
	///Using nested loops for small products, Blas for larger.
	///For float, double, std::complex<float>, std::complex<double>
	///For other types, user must provide implementation.
	template<typename T, int M, int N>
	blitz::TinyVector<T, M> matVec(const blitz::TinyMatrix<T, M, N>& mat,
		const blitz::TinyVector<T, N>& vec)
	{
		//compute small mat-vecs first
		if constexpr (M <= MATVEC_BLAS_THRESH && N <= MATVEC_BLAS_THRESH) //use naive nested loop (nnl)
			return matVec_nnl(mat, vec);
		else
		{
			//Use Blas for larger matrices
			blitz::TinyVector<T, M> mv;
			lapack::gemv<T>(CblasRowMajor, CblasNoTrans, M, N, 1, mat.data(), N, vec.data(), 1, 0, mv.data(), 1);
			return mv;
		}
	}

	///TinyMatrix-TinyVector product for integral types.
	///Using naive matVec.
	template<typename T, int M, int N, std::enable_if_t<std::is_integral<T>::value, bool> = true>
	blitz::TinyVector<T, M> matVec(const blitz::TinyMatrix<T, M, N>& mat,
		const blitz::TinyVector<T, N>& vec)
	{
		return matVec_nnl(mat, vec);
	}

	///Transpose matVec, naive nested loop (nnl). Required due to templated matrix sizes
	///NOTE: the conjugate transpose is used for complex types.
	template<typename T, int M, int N>
	blitz::TinyVector<T, N> matVec_nnl_transpose(const blitz::TinyMatrix<T, M, N>& mat,
		const blitz::TinyVector<T, M>& vec)
	{
		if constexpr (M == 1 && N == 1) return blitz::TinyVector<T, N>(mat(0, 0) * vec(0));
		blitz::TinyVector<T, N> mv; mv = 0.0;
		for (int im = 0; im < M; ++im)
			for (int in = 0; in < N; ++in)
				if constexpr (is_complex_v<T>) mv(in) += std::conj(mat(im, in)) * vec(im);
				else mv(in) += mat(im, in) * vec(im);
		return mv;
	}

	///Transpose matVec, required due to templated matrix sizes
	///NOTE: the conjugate transpose is used for complex types.
	template<typename T, int M, int N>
	blitz::TinyVector<T, N> matVec_transpose(const blitz::TinyMatrix<T, M, N>& mat,
		const blitz::TinyVector<T, M>& vec)
	{
		//compute small mat-vecs first
		if constexpr (M <= MATVEC_BLAS_THRESH && N <= MATVEC_BLAS_THRESH) //use naive nested loop (nnl)
			return matVec_nnl_transpose(mat, vec);
		else
		{
			//Use Blas for larger matrices
			blitz::TinyVector<T, N> mv;
			if constexpr (is_complex_v<T>)
				lapack::gemv<T>(CblasRowMajor, CblasConjTrans, M, N, 1, mat.data(), N, vec.data(), 1, 0, mv.data(), 1);
			else lapack::gemv<T>(CblasRowMajor, CblasTrans, M, N, 1, mat.data(), N, vec.data(), 1, 0, mv.data(), 1);
			return mv;
		}
	}

	///Transpose matVec - integral types.
	template<typename T, int M, int N, std::enable_if_t<std::is_integral<T>::value, bool> = true>
	blitz::TinyVector<T, N> matVec_transpose(const blitz::TinyMatrix<T, M, N>& mat,
		const blitz::TinyVector<T, M>& vec)
	{
		return matVec_nnl_transpose(mat, vec);
	}


	/////////////////////////////////
	//Dense matrix-matrix multiply
	/////////////////////////////////

	///TinyMatrix-TinyMatrix product
	///between MxK and KxN matrices : resulting MxN
	template<typename T, int M, int N, int K>
	blitz::TinyMatrix<T, M, N> matMat(const blitz::TinyMatrix<T, M, K>& mat1, const blitz::TinyMatrix<T, K, N>& mat2)
	{
		//small matrix-matrix products first
		if constexpr (M == 1 && K == 1 && N == 1) return blitz::TinyMatrix<T, M, N>(mat1(0, 0) * mat2(0, 0));
		else if constexpr (M <= MATMAT_BLAS_THRESH && K <= MATMAT_BLAS_THRESH &&
			N <= MATMAT_BLAS_THRESH) //faster to use nested loops
		{
			blitz::TinyMatrix<T, M, N> mm(0.0);
			for (int im = 0; im < M; ++im)
				for (int ik = 0; ik < K; ++ik)
					for (int in = 0; in < N; ++in)
						mm(im, in) += mat1(im, ik) * mat2(ik, in);
			return mm;
		}
		else
		{
			//Use Blas for larger matrix-matrix products
			blitz::TinyMatrix<T, M, N> mm;
			lapack::gemm<T>(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, mat1.data(), K,
				mat2.data(), N, 0, mm.data(), N);
			return mm;
		}
	}

	///Transpose matMat - first matrix only: A^T B - results in K x N matrix
	///NOTE: the conjugate transpose is used for complex types.
	template<typename T, int M, int N, int K>
	blitz::TinyMatrix<T, K, N> matMat_transpose(const blitz::TinyMatrix<T, M, K>& mat1, const blitz::TinyMatrix<T, M, N>& mat2)
	{
		//small matrix-matrix products first
		if constexpr (M <= MATMAT_BLAS_THRESH && K <= MATMAT_BLAS_THRESH &&
			N <= MATMAT_BLAS_THRESH)  //faster to use nested loops
		{
			blitz::TinyMatrix<T, K, N> mm(0.0);
			for (int im = 0; im < M; ++im)
				for (int ik = 0; ik < K; ++ik)
					for (int in = 0; in < N; ++in)
						if constexpr (is_complex_v<T>) mm(ik, in) += std::conj(mat1(im, ik)) * mat2(im, in);
						else mm(ik, in) += mat1(im, ik) * mat2(im, in);
			return mm;
		}
		else
		{
			//Use Blas for larger matrix-matrix products
			blitz::TinyMatrix<T, K, N> mm;
			if constexpr (is_complex_v<T>)
				lapack::gemm<T>(CblasRowMajor, CblasConjTrans, CblasNoTrans, K, N, M, 1, mat1.data(), K,
					mat2.data(), N, 0, mm.data(), N);
			else lapack::gemm<T>(CblasRowMajor, CblasTrans, CblasNoTrans, K, N, M, 1, mat1.data(), K,
				mat2.data(), N, 0, mm.data(), N);
			return mm;
		}
	}

	///Multiply two square matrices of the same size
	template<typename T, int M>
	blitz::TinyMatrix<T, M, M> matMat(const blitz::TinyMatrix<T, M, M>& mat1, const blitz::TinyMatrix<T, M, M>& mat2,
		bool transpose = false)
	{
		return transpose ? matMat_transpose<T, M, M, M>(mat1, mat2) : matMat<T, M, M, M>(mat1, mat2);
	}


	////////////////////////////
	//Matrix identity/transpose
	////////////////////////////

	///Identity matrix
	template<typename T, int N>
	blitz::TinyMatrix<T, N, N> identity()
	{
		blitz::TinyMatrix<T, N, N> id(0);
		for (int i = 0; i < N; ++i) id(i, i) = T(1);
		return id;
	}

	///Transpose of dense matrix, floating point numbers, via mkl_?omatcopy
	///NOTE: the conjugate transpose is used for complex types.
	template<typename T, int M, int N>
	blitz::TinyMatrix<T, N, M> transpose(const blitz::TinyMatrix<T, M, N>& mat)
	{
		blitz::TinyMatrix<T, N, M> tran;
		if constexpr (lapack::is_complex_v<T>)
			lapack::omatcopy<T>('r', 'c', M, N, T(1.0), mat.data(), N, tran.data(), M);		 //conjugate transpose
		else lapack::omatcopy<T>('r', 't', M, N, T(1.0), mat.data(), N, tran.data(), M); //transpose
		return tran;
	}

	//In place transpose of dense a matrix, via mkl?imatcopy
	///NOTE: the conjugate transpose is used for complex types.
	template<typename T, int N>
	void transpose_in_place(blitz::TinyMatrix<T, N, N>& mat)
	{
		if constexpr (lapack::is_complex_v<T>)
			lapack::imatcopy<T>('r', 'c', N, N, T(1.0), mat.data(), N, N);		//conjugate transpose
		else lapack::imatcopy<T>('r', 't', N, N, T(1.0), mat.data(), N, N); //transpose
	}

	///Transpose of dense matrix, integral types, naive.
	template<typename T, int M, int N, std::enable_if_t<std::is_integral<T>::value, bool> = true>
	blitz::TinyMatrix<T, N, M> transpose(const blitz::TinyMatrix<T, M, N>& mat)
	{
		blitz::TinyMatrix<T, N, M> tran;
		for (int i = 0; i < M; ++i)
			for (int j = 0; j < N; ++j)
				tran(j, i) = mat(i, j);
		return tran;
	}


	////////////////////////////////////
	//blitz::Array versions (Blas only)
	////////////////////////////////////

	///blitz::Array matrix-vector product.
	///For float, double, std::complex<float>, std::complex<double>
	///For other types, user must provide implementation.
	template<typename T>
	blitz::Array<T, 1> matVec(const blitz::Array<T, 2>& mat,
		const blitz::Array<T, 1>& vec)
	{
		int M = mat.extent(0); int N = mat.extent(1);
		assert(vec.extent(0) == N);
		blitz::Array<T, 1> mv(M);
		lapack::gemv<T>(CblasRowMajor, CblasNoTrans, M, N, 1, mat.data(), N, vec.data(), 1, 0, mv.data(), 1);
		return mv;
	}

	///blitz::Array transpose matVec
	///NOTE: the conjugate transpose is used for complex types.
	template<typename T>
	blitz::Array<T, 1 > matVec_transpose(const blitz::Array<T, 2>& mat,
		const blitz::Array<T, 1>& vec)
	{
		int M = mat.extent(0); int N = mat.extent(1);
		assert(vec.extent(0) == M);
		blitz::Array<T, 1> mv(N);
		if constexpr (is_complex_v<T>)
			lapack::gemv<T>(CblasRowMajor, CblasConjTrans, M, N, 1, mat.data(), N, vec.data(), 1, 0, mv.data(), 1);
		else lapack::gemv<T>(CblasRowMajor, CblasTrans, M, N, 1, mat.data(), N, vec.data(), 1, 0, mv.data(), 1);
		return mv;
	}

	///MxK and KxN blitz::Arrays : resulting MxN
	template<typename T>
	blitz::Array<T, 2> matMat(const blitz::Array<T, 2>& mat1, const blitz::Array<T, 2>& mat2)
	{
		assert(mat1.extent(1) == mat2.extent(0));
		int M = mat1.extent(0); int K = mat1.extent(1); int N = mat2.extent(1);
		blitz::Array<T, 2> mm(M, N);
		lapack::gemm<T>(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, mat1.data(), K,
			mat2.data(), N, 0, mm.data(), N);
		return mm;
	}

	///Transpose matMat blitz::Arrays - first matrix only: A^T B - results in K x N matrix
	///NOTE: the conjugate transpose is used for complex types.
	template<typename T>
	blitz::Array<T, 2> matMat_transpose(const blitz::Array<T, 2>& mat1, const blitz::Array<T, 2>& mat2)
	{
		assert(mat1.extent(0) == mat2.extent(0));
		int M = mat1.extent(0); int K = mat1.extent(1); int N = mat2.extent(1);
		blitz::Array<T, 2> mm(K, N);
		if constexpr (is_complex_v<T>)
			lapack::gemm<T>(CblasRowMajor, CblasConjTrans, CblasNoTrans, K, N, M, 1, mat1.data(), K,
				mat2.data(), N, 0, mm.data(), N);
		else lapack::gemm<T>(CblasRowMajor, CblasTrans, CblasNoTrans, K, N, M, 1, mat1.data(), K,
			mat2.data(), N, 0, mm.data(), N);
		return mm;
	}

	///Identity 2D blitz::Array
	template<typename T>
	blitz::Array<T, 2> identity_Arr(int N)
	{
		blitz::Array<T, 2> id(N); id = T(0);
		for (int i = 0; i < N; ++i) id(i, i) = T(1);
		return id;
	}

	///Transpose of 2D blitz::Array, floating point numbers, via mkl_?omatcopy
	///NOTE: the conjugate transpose is used for complex types.
	template<typename T>
	blitz::Array<T, 2> transpose(const blitz::Array<T, 2>& mat)
	{
		int M = mat.extent(0); int N = mat.extent(1);
		blitz::Array<T, 2> tran(N, M);
		if constexpr (lapack::is_complex_v<T>)
			lapack::omatcopy<T>('r', 'c', M, N, T(1.0), mat.data(), N, tran.data(), M);		 //conjugate transpose
		else lapack::omatcopy<T>('r', 't', M, N, T(1.0), mat.data(), N, tran.data(), M); //transpose
		return tran;
	}

}//end namespace BlockSp::dense

#endif //end BLOCKSP_DENSE_MULTIPLY_HPP