#ifndef BLOCKSP_UTILITY_HPP
#define BLOCKSP_UTILITY_HPP

#include "blocksp_containers.hpp"
#include "blocksp_lapack.hpp"

#include <algorithm>
#include <fstream>
#include <random>
#include <numeric>

///blocksp_utility.hpp contains several utility functions for the BlockSp/blitz data structures.
///These include:
///
///1) Basic arithmetic, timers, random number functions.												\n
///2) TinyVector/Matrix operations, array/matrix manipulations.									\n
///3) Random BlockSp matrices and solArrays.																		\n
///4) BlockSp to full dense matrix, solArray to full dense vector, and reverse. \n
///5) Scalar to Complex BlockSp/solArray converters.														\n
///6) .mtx file matrix/vector to BlockSp/solArray converters.
///7) Explicit matrix inverse calculator for very small matrices.								\n 
///8) Hash function for TinyVectors. Prefetching functions.
namespace BlockSp::util
{
	////////////////////////
	//Timer
	/////////////////////////
	/*
	std::clock_t start;
	double duration;
	start = std::clock();
	//code here
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	std::cout << duration << " seconds" << std::endl;
	*/


	///////////////////////
	//Arithmetic
	///////////////////////

	///Power: base^exp, for compile time powers.
	inline int constexpr power(int base, int exp)
	{
		return exp == 0 ? 1 : base * power(base, exp - 1);
	}

	///n!
	inline int factorial(int n)
	{
		assert(n >= 0);
		if (n == 0) return 1;
		int val = 1;
		for (int fct = 2; fct <= n; ++fct) val *= fct;
		return val;
	}

	///n choose k.
	inline int choose(int n, int k)
	{
		return factorial(n) / (factorial(k) * factorial(n - k));
	}

	///Square root of x.
	template<typename T>
	inline T sqrt(T x)
	{
		return std::sqrt(x);
	}

	///Square of x.
	template<typename T>
	inline T sqr(T x)
	{
		return x * x;
	}

	///Euclidean/L2 norm of x.
	template<typename T>
	inline T mag(T x)
	{
		return sqrt(x * x);
	}

	///Squared L2 norm of x.
	template<typename T>
	inline T magsqr(T x)
	{
		return x * x;
	}


	//////////////////////////////
	//TinyVector operations
	//////////////////////////////

	///Max of a TinyVector.
	template<typename T, int N>
	inline T tvMax(const blitz::TinyVector<T, N>& tv)
	{
		T mx = tv(0);
		for (const auto& val : tv)
			if (val > mx) mx = val;
		return mx;
	}

	///Min of a TinyVector.
	template<typename T, int N>
	inline T tvMin(const blitz::TinyVector<T, N>& tv)
	{
		T mn = tv(0);
		for (const auto& val : tv)
			if (val < mn) mn = val;
		return mn;
	}

	///Max absolute value of a TinyVector.
	template<typename T, int N>
	inline T tvMaxAbs(const blitz::TinyVector<T, N>& tv)
	{
		T mx = std::abs(tv(0));
		for (const auto& val : tv)
			if (val > mx) mx = std::abs(val);
		return mx;
	}

	///Min absolute value of a TinyVector.
	template<typename T, int N>
	inline T tvMinAbs(const blitz::TinyVector<T, N>& tv)
	{
		T mn = std::abs(tv(0));
		for (const auto& val : tv)
			if (val < mn) mn = std::abs(val);
		return mn;
	}

	///arg_max of a TinyVector.
	template<typename T, int N>
	inline int tvArgMax(const blitz::TinyVector<T, N>& tv)
	{
		T mx = tv(0); int i_arg = 0;
		for (int i = 1; i < N; ++i)
		{
			auto val = tv(i);
			if (val > mx)
			{
				mx = val;
				i_arg = i;
			}
		}
		return i_arg;
	}

	///arg_min of a TinyVector.
	template<typename T, int N>
	inline int tvArgMin(const blitz::TinyVector<T, N>& tv)
	{
		T mn = tv(0); int i_arg = 0;
		for (int i = 1; i < N; ++i)
		{
			auto val = tv(i);
			if (val < mn)
			{
				mn = val;
				i_arg = i;
			}
		}
		return i_arg;
	}

	///Euclidean/L2 norm of a TinyVector.
	template<typename T, int N>
	inline T tvMag(const blitz::TinyVector<T, N>& x)
	{
		T res = x(0) * x(0);
		for (int i = 1; i < N; ++i)
			res += x(i) * x(i);
		return sqrt(res);
	}

	///Squared L2 norm of a TinyVector.
	template<typename T, int N>
	inline T tvMagSqr(const blitz::TinyVector<T, N>& x)
	{
		T res = x(0) * x(0);
		for (int i = 1; i < N; ++i)
			res += x(i) * x(i);
		return res;
	}

	///Absolute value of each component of a TinyVector
	template<typename T, int N>
	inline blitz::TinyVector<T, N> tvAbs(const blitz::TinyVector<T, N>& x)
	{
		return TinyVector<T, N>(blitz::abs(x));
	}

	///Floor of each dimension of a TinyVector.
	template<typename T, int N> requires lapack::is_lapack_type<T>
	inline blitz::TinyVector<int, N> tvFloor(const blitz::TinyVector<ScalarType_t<T>, N>& tv)
	{
		blitz::TinyVector<int, N> fl;
		int cnt = 0;
		for (const auto& val : tv)
		{
			fl(cnt) = floor(val); ++cnt;
		}
		return fl;
	}

	///Reverse a TinyVector.
	template<typename T, int N>
	inline blitz::TinyVector<T, N> tvRev(const blitz::TinyVector<T, N>& tv)
	{
		blitz::TinyVector<T, N> rev;
		for (int i = 0; i < N; ++i)
			rev(N - i - 1) = tv(i);
		return rev;
	}

	///Raise a TinyVector to the power exp (each component seperately), 
	///ex: (x,y)^(a,b) = (x^a, y^b)
	template<typename T, typename S, int N>
	inline blitz::TinyVector<T, N> tvPow(const blitz::TinyVector<T, N>& x,
		const blitz::TinyVector<S, N>& exp)
	{
		blitz::TinyVector<T, N> p;
		for (int i = 0; i < N; ++i)
			p(i) = std::pow(x(i), exp(i));
		return p;
	}

	///Raise a TinyVector to the power exp, 
	///ex: (x,y)^a = (x^a, y^a)
	template<typename T, typename S, int N>
	inline blitz::TinyVector<T, N> tvPow(const blitz::TinyVector<T, N>& x, S exp)
	{
		blitz::TinyVector<T, N> p;
		for (int i = 0; i < N; ++i)
			p(i) = std::pow(x(i), exp);
		return p;
	}

	///Get a subset TinyVector of length B from a TinyVector of length B*D.
	template<typename T, int B, int D>
	inline blitz::TinyVector<T, B> tvSubset(const blitz::TinyVector<T, B* D>& tv, int dim)
	{
		assert(0 <= dim && dim < D);
		blitz::TinyVector<T, B> sub;
		int cnt = 0;
		for (int ib = B * dim; ib < B * (dim + 1); ++ib)
		{
			sub(cnt) = tv(ib);
			cnt++;
		}
		return sub;
	}

	///Concatenate a TinyVector<TinyVector> into a single TinyVector.
	template<typename T, int B, int D>
	inline blitz::TinyVector<T, B* D> tvConcat(const blitz::TinyVector<blitz::TinyVector<T, B>, D>& tv)
	{
		blitz::TinyVector<T, B* D> cat;
		for (int dim = 0; dim < D; ++dim)
			for (int ib = 0; ib < B; ++ib)
				cat(dim * B + ib) = tv(dim)(ib);
		return cat;
	}

	///Subtract off the last dimension of a TinyVector.
	template<typename T, int N>
	inline blitz::TinyVector<T, N - 1> tvM1(const blitz::TinyVector<T, N>& tv)
	{
		blitz::TinyVector<T, N - 1> tvM; tvM = tv;
		return tvM;
	}

	///blitz dot product of two TinyVectors, with complex conjugate.
	template<typename T, int N>
	inline T bdot(const blitz::TinyVector<T, N>& x, const blitz::TinyVector<T, N>& y)
	{
		T val(0.0);
		if constexpr (is_complex_v<T>) val = blitz::dot(blitz::conj(x), y);
		else val = blitz::dot(x, y);
		return val;
	}

	///blitz dot product of two 1D blitz::Arrays, with complex conjugate.
	template<typename T>
	inline T bdot(const blitz::Array<T, 1>& x, const blitz::Array<T, 1>& y)
	{
		assert(x.extent(0) == y.extent(0));
		T val(0.0);
		if constexpr (is_complex_v<T>) val = blitz::dot(blitz::conj(x), y);
		else val = blitz::dot(x, y);
		return val;
	}


	////////////////////////////////
	//TinyMatrix/BlockSp operations
	////////////////////////////////

	///Row of TinyMatrix.
	template<typename T, int M, int N>
	inline blitz::TinyVector<T, N> tmRow(const blitz::TinyMatrix<T, M, N> tm, int row)
	{
		blitz::TinyVector<T, N> r;
		for (int ic = 0; ic < N; ++ic)
			r(ic) = tm(row, ic);
		return r;
	}

	///Column of TinyMatrix.
	template<typename T, int M, int N>
	inline blitz::TinyVector<T, M> tmCol(const blitz::TinyMatrix<T, M, N> tm, int col)
	{
		blitz::TinyVector<T, M> c;
		for (int ir = 0; ir < M; ++ir)
			c(ir) = tm(ir, col);
		return c;
	}

	///Max absolution value of a TinyMatrix.
	template<typename T, int M, int N>
	inline ScalarType_t<T> tmMaxAbs(const blitz::TinyMatrix<T, M, N>& A)
	{
		return lapack::lange<T>(LAPACK_ROW_MAJOR, 'M', M, N, A.data(), N);
	}

	///Frobenius norm of a TinyMatrix.
	template<typename T, int M, int N>
	inline ScalarType_t<T> frobeniusNorm(const blitz::TinyMatrix<T, M, N>& A)
	{
		return lapack::lange<T>(LAPACK_ROW_MAJOR, 'F', M, N, A.data(), N);
	}

	///One norm of a TinyMatrix.
	template<typename T, int M, int N>
	inline ScalarType_t<T> oneNorm(const blitz::TinyMatrix<T, M, N>& A)
	{
		return lapack::lange<T>(LAPACK_ROW_MAJOR, '1', M, N, A.data(), N);
	}

	///Infinity norm of a TinyMatrix.
	template<typename T, int M, int N>
	inline ScalarType_t<T> infNorm(const blitz::TinyMatrix<T, M, N>& A)
	{
		return lapack::lange<T>(LAPACK_ROW_MAJOR, 'I', M, N, A.data(), N);
	}

	///Trace of a square TinyMatrix.
	template<typename T, int B>
	inline T trace(const blitz::TinyMatrix<T, B, B>& A)
	{
		T trace = 0.0;
		for (int i = 0; i < B; ++i) trace += A(i, i);
		return trace;
	}

	///Tangential matrix to normal vector n: I - n kron n.
	template<typename T, int N>
	blitz::TinyMatrix<T, N, N> tangentMat(const blitz::TinyVector<T, N>& n)
	{
		blitz::TinyMatrix<T, N, N> InKronn(0.0);
		for (int i = 0; i < N; ++i) InKronn(i, i) = T(1.0);
		for (int i = 0; i < N; ++i)
			for (int j = 0; j < N; ++j)
				InKronn(i, j) -= n(i) * n(j);
		return InKronn;
	}

	///Trace of a BlockSp matrix.
	template<typename T, int B>
	inline T trace(const BlockSp<T, B>& A)
	{
		T trace = 0.0;
		int end = std::min(A.m(), A.n());
		for (int ie = 0; ie < end; ++ie)
			for (int ib = 0; ib < B; ++ib)
				trace += A(ie, ie)(ib, ib);
		return trace;
	}

	///Frobenius norm of BlockSp = trace(A^* A).
	template<typename T, int B>
	inline ScalarType_t<T> frobeniusNorm(const BlockSp<T, B>& A)
	{
		auto ATA = bsmm<T, B>(A, A, true);
		return std::sqrt(std::abs(trace(ATA)));
	}

	///Calculate the determinant of a TinyMatrix.
	///Warning! Explicit determinant calculations are an O(N!), where ! is factorial.
	///Use for small N only.
	template<typename T, int N>
	T determinant(const blitz::TinyMatrix<T, N, N>& A)
	{
		if constexpr (N == 1) return A(0, 0);
		if constexpr (N == 2) return A(0, 0) * A(1, 1) - A(1, 0) * A(0, 1);
		else
		{
			T det = 0.0;
			for (int j = 0; j < N; ++j) //scan across column of the first row
			{
				blitz::TinyMatrix<T, N - 1, N - 1> ASub; //make submatrix 
				for (int is = 0; is < N - 1; ++is)
					for (int js = 0; js < N; ++js) if (js != j)
					{
						int col_s = js > j ? js - 1 : js;		 //find column for ASub
						ASub(is, col_s) = A(is + 1, js);
					}
				T val = A(0, j) * determinant<T, N - 1>(ASub); //recursive call to submatrix
				if (j % 2 == 1) val *= -1; //include the -1 every other column
				det += val;
			}
			return det;
		}
	}

	////////////////////////////////////
	//Random Numbers/Random BlockSp
	////////////////////////////////////
	/*
	//random double between 0 and 1, copy and paste where needed

	//same random numbers each run
	std::uniform_real_distribution<double> rnd(0.0, 1.0);
	std::default_random_engine gen;
	rnd(gen); // this rnd(gen) provides the number

	//different random numbers each run
	std::uniform_real_distribution<double> rnd(0.0, 1.0);
	std::random_device rd;
	std::default_random_engine gen(rd());
	rnd(gen);
	//*/

	///Return a randomly filled solArray. If normalized, L2 norm = 1,
	///random numbers from min to max.
	template<typename T, int B, int D = 1>
	solArray<T, B, D> rand_solArray(int l, bool normalized = true,
		ScalarType_t<T> min = 0.0, ScalarType_t<T> max = 1.0)
	{
		assert(max > min);
		std::uniform_real_distribution<ScalarType_t<T>> rnd(min, max);
		std::default_random_engine gen;
		solArray<T, B, D> x(l);
		for (int i = 0; i < l; ++i)
			for (int dim = 0; dim < D; ++dim)
			{
				blitz::TinyVector<T, B> x_d;
				if constexpr (is_complex_v<T>)
					for (int ib = 0; ib < B; ++ib)
						x_d(ib) = T(rnd(gen), rnd(gen));
				else
					for (int ib = 0; ib < B; ++ib)
						x_d(ib) = T(rnd(gen));
				x(i, dim) = x_d;
			}
		return normalized ? (x / twoNorm(x, x)) : x;
	}

	///Generate a random mxn BlockSp matrix with ~ncol columns per row.
	///Random numbers from min to max
	template<typename T, int B>
	BlockSp<T, B> rand_BlockSp(int m, int n, int nCol = 4, ScalarType_t<T> min = 0.0,
		ScalarType_t<T> max = 1.0)
	{
		assert(max > min);
		std::uniform_real_distribution<ScalarType_t<T>> rnd(min, max);
		std::default_random_engine gen;
		BlockSp<T, B> A(m, n);
		nCol = std::min(n, nCol);
		for (int im = 0; im < m; ++im)
			for (int in = 0; in < nCol; ++in)
			{
				int col = rand() % n;	//random column index
				//Note: it's better practice to make your Blocks/TinyMatrices locally on the stack,
				//then allocate them to the BlockSp matrix. As seen here.
				blitz::TinyMatrix <T, B, B> blk;
				if constexpr (is_complex_v<T>)
					for (int i = 0; i < B; ++i)
						for (int j = 0; j < B; ++j)
							blk(i, j) = T(rnd(gen), rnd(gen));
				else
					for (int i = 0; i < B; ++i)
						for (int j = 0; j < B; ++j)
							blk(i, j) = T(rnd(gen));
				A(im, col) = blk;
			}
		return A;
	}

	///Return a randomly filled std::vector. If normalized, L2 norm = 1.
	///Random numbers from min to max.
	template<typename T> requires lapack::is_lapack_type<T>
	std::vector<T> rand_vec(int l, bool normalized = true,
		ScalarType_t<T> min = 0.0, ScalarType_t<T> max = 1.0)
	{
		assert(max > min);
		std::vector<T> vec(l);
		std::uniform_real_distribution<ScalarType_t<T>> rnd(min, max);
		std::default_random_engine gen;
		for (int i = 0; i < l; ++i)
			if constexpr (is_complex_v<T>)
			{
				vec[i].real(rnd(gen));
				vec[i].imag(rnd(gen));
			}
			else vec[i] = rnd(gen);
		if (normalized)
		{
			T norm(0);
			for (int i = 0; i < l; ++i)
			{
				if constexpr (is_complex_v<T>)
					norm += std::conj(vec[i]) * vec[i];
				else norm += vec[i] * vec[i];
			}
			norm = std::sqrt(std::abs(norm));
			for (int i = 0; i < l; ++i)
				vec[i] /= norm;
		}
		return vec;
	}


	/////////////////////////////
	//Array Manipulation
	/////////////////////////////

	///Get vector of strides from extent of an N-D blitz::Array.
	///Row major is assumed: so a 2x2x2 array has stride (4,2,1).
	template<int N>
	inline blitz::TinyVector<int, N> getStride(const blitz::TinyVector<int, N>& extent)
	{
		blitz::TinyVector<int, N> stride;
		int strd = 1;
		for (int dim = N - 1; dim >= 0; dim--)
		{
			stride(dim) = strd;
			strd *= extent(dim);
		}
		return stride;
	}

	///Convert an N-D index to 1D index given a stride vector.
	template<int N>
	inline int convIndexNto1(const blitz::TinyVector<int, N>& indexND,
		const blitz::TinyVector<int, N>& stride)
	{
		int index1D = blitz::dot(indexND, stride);
		return index1D;
	}

	///Convert 1D index to N-D index given vector of extents.
	template<int N>
	inline blitz::TinyVector<int, N> convIndex1toN(int index1D,
		const blitz::TinyVector<int, N>& extent)
	{
		assert(0 <= index1D && index1D < blitz::product(extent));
		blitz::TinyVector<int, N> indexND;
		blitz::TinyVector<int, N> stride{ getStride<N>(extent) };
		for (int dim = 0; dim < N; ++dim)
			indexND(dim) = (index1D / stride(dim)) % extent(dim);
		return indexND;
	}

	///Extract a subset of a std::vector.
	template<typename T>
	inline std::vector<T> extractSubVec(const std::vector<T> v, int start, int end)
	{
		assert(start >= 0 && start < v.size() && end > 0 && end < v.size());
		typename std::vector<T>::const_iterator first = v.begin() + start;
		typename std::vector<T>::const_iterator last = v.begin() + end;
		return std::vector<T>(first, last);
	}

	///Get a reference to a MxM block of a D*MxD*M 2D blitz::Array.
	///Note: the block references the input blitz::Array, changing the block changes the array. 
	template<typename T, int B, int D>
	inline blitz::Array<T, 2> getBlock(const blitz::Array<T, 2>& arr, int i, int j)
	{
		assert(0 <= i && 0 <= j && i < D && j < D);
		assert(arr.extent(0) == B * D && arr.extent(1) == B * D);
		//blitz::Range extracts a reference to a slice of the original array 
		blitz::Array<T, 2> block = arr(blitz::Range(i * B, (i + 1) * B - 1),
			blitz::Range(j * B, (j + 1) * B - 1));
		return block;
	}

	///Get a reference to a M vec of a D*M length 1D blitz::Array.
	template<typename T, int B, int D>
	inline blitz::Array<T, 1> getBlock(const blitz::Array<T, 1>& arr, int i)
	{
		assert(0 <= i && i < D);
		assert(arr.extent(0) == B * D);
		blitz::Array<T, 1> block = arr(blitz::Range(i * B, (i + 1) * B - 1));
		return block;
	}

	///Set an BxB block of BDxBD TinyMatrix from an BxB TinyMatrix, at location/block index (i,j).
	template<typename T, int B, int D>
	inline void setBlock(blitz::TinyMatrix<T, B* D, B* D>& big,
		const blitz::TinyMatrix<T, B, B>& small, int i, int j)
	{
		assert(0 <= i && 0 <= j && i < D && j < D);
		for (int im = 0; im < B; ++im)
			for (int in = 0; in < B; ++in)
				big(i * B + im, j * B + in) = small(im, in);
	}

	///Insert a value into a BlockSp matrix from a global index (i, j).
	///Used in .mtx file conversion
	///Note: it is better practice to make Blocks/TinyMatrices 
	/// locally on the stack, then allocate them to the BlockSp matrix.
	template<typename T, int B>
	void setFromGlobal(BlockSp<T, B>& A, int i, int j, T val)
	{
		assert(i >= 0 && j >= 0);
		assert(i < A.m() * B && j < A.n() * B);
		int im = i / B;			//find the row and col of BlockSp matrix
		int in = j / B;
		int i_blk = i % B;  //find the row and col within the local block
		int j_blk = j % B;
		A(im, in)(i_blk, j_blk) = val; //fill value
	}

	///Insert a value into a solArray from a global index i.
	///Used in .mtx file conversion
	///Note: it is better practice to make TinyVectors 
	/// locally on the stack, then allocate them to the solArray.
	template<typename T, int B, int D = 1>
	void setFromGlobal(solArray<T, B, D>& x, int i, T val)
	{
		assert(i >= 0 && i < x.len() * B * D);
		int BD = B * D;
		int im = i / BD;			//find the index of the solArray
		int i_BD = i % BD;		//find the index within the local blocks
		int dim = i_BD / B;
		int ib = i_BD % B;
		x(im, dim)(ib) = val; //fill value
	}


	/////////////////////////////////
	//BlockSp to Full converters

	///Convert a BlockSp matrix to a full dense matrix.
	///Resulting matrix may be intractably large.
	template<typename T, int B>
	inline blitz::Array<T, 2> blkToFull(const BlockSp<T, B>& blk)
	{
		blitz::Array<T, 2> full(blk.m() * B, blk.n() * B);
		full = 0.0;
		for (int im = 0; im < blk.n(); ++im)
			for (int in = 0; in < blk.colSize(im); ++in)
			{
				for (int i = 0; i < B; ++i)
					for (int j = 0; j < B; ++j)
						full(im * B + i, blk.colInd(im, in) * B + j) = blk(im, blk.colInd(im, in))(i, j);
			}
		return full;
	}

	///Convert a solArray to a full length 1D blitz::Array.
	template<typename T, int B, int D>
	inline blitz::Array<T, 1> saToFull(const solArray<T, B, D>& x)
	{
		blitz::Array<T, 1> full(x.len() * B * D);
		for (int ie = 0; ie < x.len(); ++ie)
		{
			blitz::Array<T, 1> full_el{ full(blitz::Range(ie * B * D, (ie + 1) * B * D - 1)) };
			for (int dim = 0; dim < D; ++dim)
				getBlock<T, B, D>(full_el, dim) = x(ie, dim);
		}
		return full;
	}

	///Convert a solArray to a full length std::vector.
	template<typename T, int B, int D>
	inline std::vector<T> saToFull_vec(const solArray<T, B, D>& x)
	{
		std::vector<T> full(x.len() * B * D);
		for (int ie = 0; ie < x.len(); ++ie)
			for (int dim = 0; dim < D; ++dim)
			{
				int start = ie * B * D + dim * B;
				auto data = x(ie, dim).data();
				std::copy(data, data + B, full.begin() + start);
			}
		return full;
	}

	///Convert a full length 1D blitz::Array to a solArray.
	///Assume full layout is as given from saToFull.
	///el0(dim0 ... dimD) ... eln(dim0 ... dimD)
	template<typename T, int B, int D = 1>
	inline solArray<T, B, D> fullToSA(const blitz::Array<T, 1>& full, int numEls)
	{
		assert(full.size() == numEls * B * D);
		solArray<T, B, D> x{ numEls };
		for (int ie = 0; ie < numEls; ++ie)
		{
			blitz::Array<T, 1> full_el{ full(blitz::Range(ie * B * D, (ie + 1) * B * D - 1)) };
			for (int dim = 0; dim < D; ++dim)
				x(ie, dim) = getBlock<T, B, D>(full_el, dim);
		}
		return x;
	}

	///Convert a full length std::vector to a solArray.
	///Assume full layout is as given from saToFull_vec.
	///el0(dim0 ... dimD) ... eln(dim0 ... dimD)
	template<typename T, int B, int D = 1>
	inline solArray<T, B, D> fullToSA(const std::vector<T>& full, int numEls)
	{
		assert(full.size() == numEls * B * D);
		solArray<T, B, D> x{ numEls };
		for (int ie = 0; ie < numEls; ++ie)
			for (int dim = 0; dim < D; ++dim)
			{
				int start = ie * B * D + dim * B;
				std::copy(full.begin() + start, full.begin() + start + B, x(ie, dim).data());
			}
		return x;
	}

	///Convert a TinyMatrix to a 2D array (which has better cout<< readability).
	template<typename T, int M, int N>
	inline blitz::Array<T, 2> tnyMatToArr(const blitz::TinyMatrix<T, M, N>& mat)
	{
		blitz::Array<T, 2> arr(M, N);
		arr(blitz::Range::all(), blitz::Range::all()) = mat;
		return arr;
	}

	///Convert a 2D blitz::Array of size MxN to a TinyMatrix.
	template<typename T, int M, int N>
	inline blitz::TinyMatrix<T, M, N> arrToTnyMat(const blitz::Array<T, 2>& arr)
	{
		assert(arr.extent(0) == M && arr.extent(1) == N);
		blitz::TinyMatrix<T, M, N> tm;
		for (int i = 0; i < M; ++i)
			for (int j = 0; j < N; ++j)
				tm(i, j) = arr(i, j);
		return tm;
	}


	//////////////////////
	//Complex conversion

	///Make a complex BlockSp matrix from a scalar BlockSp matrix.
	///If input matrix is already complex, that matrix is returned.
	template<typename T, int B>
	BlockSp<ComplexType_t<T>, B> makeComplex(const BlockSp<T, B>& A)
	{
		if constexpr (is_complex_v<T>) return A;
		else
		{
			BlockSp<ComplexType_t<T>, B> C(A.m(), A.n());
			for (int im = 0; im < A.m(); ++im)
				for (const auto& col : A.colInd(im))
					C(im, col) = A(im, col);
			return C;
		}
	}

	///Make a complex solArray from a scalar solArray.
	///If input is already complex, that array is returned.
	template<typename T, int B, int D = 1>
	solArray<ComplexType_t<T>, B, D> makeComplex(const solArray<T, B, D>& x)
	{
		if constexpr (is_complex_v<T>) return x;
		else
		{
			int l = x.len();
			solArray<ComplexType_t<T>, B, D> x_c(l);
			for (int i = 0; i < l; ++i)
				for (int dim = 0; dim < D; ++dim)
					x_c(i, dim) = x(i, dim);
			return x_c;
		}
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
	template<typename T, int B = 1>
	BlockSp<T, B> mtxMatrix_to_BlockSp(const std::string& mat_loc,
		unsigned int start_line = 0, bool symmetric = false,
		bool base_zero = false, bool printMatSize = false)
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
				setFromGlobal(A, i, j, val);
				if (symmetric && i != j) //if symmetric, fill (j,i) with complex conjugate.
					setFromGlobal(A, j, i, std::conj(val));
			}
			else
			{
				ss2 >> val;
				setFromGlobal(A, i, j, val);
				if (symmetric && i != j) //if symmetric, fill (j,i)
					setFromGlobal(A, j, i, val);
			}
		}
		return A;
	}

	///Read in a matrix-market vector file (.mtx) and convert it to a solArray.
	template<typename T, int B = 1, int D = 1>
	solArray<T, B, D> mtxVector_to_solArray(const std::string& vec_loc,
		unsigned int start_line = 0, bool printVecSize = false)
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
		//mtx's number of rows M. N must = 1
		int constexpr BD = B * D;
		assert(M % BD == 0 && N == 1);
		int l = M / BD; //length of vector
		solArray<T, B, D> x(l);

		//Read from mtx array and set in solArray
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
			setFromGlobal<T, B, D>(x, im, val);
		}
		return x;
	}


	////////////////////////////
	//std::vector utility functions

	///Get a permutation vector from another vector and it's comparison function.
	template <typename T, typename Compare>
	std::vector<int> sort_permutation(const std::vector<T>& v, Compare compare)
	{
		std::vector<int> p(v.size());
		std::iota(p.begin(), p.end(), 0);
		std::sort(p.begin(), p.end(),
			[&](int i, int j) { return compare(v[i], v[j]); });
		return p;
	}
	template <typename T, typename Compare>
	std::vector<int> sort_permutation(const blitz::Array<T, 1>& v, Compare compare)
	{
		std::vector<int> p(v.size());
		std::iota(p.begin(), p.end(), 0);
		std::sort(p.begin(), p.end(),
			[&](int i, int j) { return compare(v(i), v(j)); });
		return p;
	}

	///Apply the permutation vector to another vector.
	template <typename T>
	std::vector<T> apply_permutation(const std::vector<T>& v, const std::vector<int>& p)
	{
		std::vector<T> sorted_v(v.size());
		std::transform(p.begin(), p.end(), sorted_v.begin(),
			[&](int i) { return v(i); });
		return sorted_v;
	}
	template <typename T>
	blitz::Array<T, 1> apply_permutation(const blitz::Array<T, 1>& v, const std::vector<int>& p)
	{
		blitz::Array<T, 1> sorted_v(v.size());
		std::transform(p.begin(), p.end(), sorted_v.begin(),
			[&](int i) { return v(i); });
		return sorted_v;
	}

	///Print std::vector.
	template<typename T>
	inline void printV(const std::vector<T>& v)
	{
		for (const auto& val : v)
			std::cout << val << " ";
		std::cout << std::endl;
	}


	/////////////////////////////
	//Explicit small inverses

	///Compute explicit inverse of a small 1x1, 2x2, and 3x3 matrix
	///A^-1 = 1/det * adj(A)
	///det = deteminant of A, adj(A) = Adjoint matrix of A.
	template<typename T, int N, std::enable_if_t<N <= 3, bool> = true>
	blitz::TinyMatrix<T, N, N> explicit_small_Inv(const blitz::TinyMatrix<T, N, N>& A)
	{
		if constexpr (N == 1) return blitz::TinyMatrix<T, N, N>(1.0 / A(0, 0));
		else if constexpr (N == 2)
		{
			//A = [a b] , A^-1 = 1/(ad - bc)[d  -b]
			//		[c d]											[-c  a]
			T det = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0); //Determinant of A
			blitz::TinyMatrix<T, N, N> A_Inv;
			A_Inv(0, 0) = A(1, 1);
			A_Inv(0, 1) = -A(0, 1);
			A_Inv(1, 0) = -A(1, 0);
			A_Inv(1, 1) = A(0, 0);
			A_Inv /= det;
			return A_Inv;
		}
		else if constexpr (N == 3)
		{
			blitz::TinyMatrix<T, N, N> A_Inv;

			//1) Explicitly compute Adjoint matrix of A. i.e. adj(A)
			//This is equal to the transpose of the Cofactor matrix.
			A_Inv(0, 0) = A(1, 1) * A(2, 2) - A(2, 1) * A(1, 2);
			A_Inv(1, 0) = -(A(1, 0) * A(2, 2) - A(2, 0) * A(1, 2));
			A_Inv(2, 0) = A(1, 0) * A(2, 1) - A(2, 0) * A(1, 1);

			A_Inv(0, 1) = -(A(0, 1) * A(2, 2) - A(2, 1) * A(0, 2));
			A_Inv(1, 1) = A(0, 0) * A(2, 2) - A(2, 0) * A(0, 2);
			A_Inv(2, 1) = -(A(0, 0) * A(2, 1) - A(2, 0) * A(0, 1));

			A_Inv(0, 2) = A(0, 1) * A(1, 2) - A(1, 1) * A(0, 2);
			A_Inv(1, 2) = -(A(0, 0) * A(1, 2) - A(1, 0) * A(0, 2));
			A_Inv(2, 2) = A(0, 0) * A(1, 1) - A(1, 0) * A(0, 1);

			//2) Calculate determinant of matrix A
			T det = A(0, 0) * (A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1))
				- A(0, 1) * (A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0))
				+ A(0, 2) * (A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0));

			//3) Return A^-1 = 1/det * adj(A)
			A_Inv /= det;
			return A_Inv;
		}
		else
		{
			std::cout << "N = " << N << " is too large for explicit_small_Inv. A returned" << std::endl;
			return A;
		}
	}

	///Compute explicit inverse of a small symmetic matrix A stored in compact vector form.
	///A length N*(N+1)/2, stored as if spanning the LOWER triangular portion of a matrix.
	///A^-1 = 1/det * adj(A)
	///det = deteminant of A, adj(A) = Adjoint matrix of A.
	///For N = 1, 2, and 3 only.
	template<typename T, int N, std::enable_if_t<N <= 3, bool> = true>
	blitz::TinyMatrix<T, N, N> explicit_small_Inv(const blitz::TinyVector<T, (N* (N + 1)) / 2>& A)
	{
		if constexpr (N == 1) return blitz::TinyMatrix<T, 1, 1>(1.0 / A(0));
		else if constexpr (N == 2)
		{
			//A = [a b] , A^-1 = 1/(ad - bc)[d  -b]
			//		[c d]											[-c  a]
			T det = A(0) * A(2) - A(1) * A(1); //Determinant of A
			blitz::TinyMatrix<T, N, N> A_Inv;
			A_Inv(0, 0) = A(2);
			A_Inv(0, 1) = -A(1);
			A_Inv(1, 0) = -A(1);
			A_Inv(1, 1) = A(0);
			A_Inv /= det;
			return A_Inv;
		}
		else if constexpr (N == 3)
		{
			blitz::TinyMatrix<T, 3, 3> A_Inv;

			//1) Explicitly compute Adjoint matrix of A. i.e. adj(A)
			//This is equal to the transpose of the Cofactor matrix.
			A_Inv(0, 0) = A(2) * A(5) - A(4) * A(4);
			A_Inv(1, 0) = -(A(1) * A(5) - A(3) * A(4));
			A_Inv(2, 0) = A(1) * A(4) - A(3) * A(2);
			A_Inv(1, 1) = A(0) * A(5) - A(3) * A(3);
			A_Inv(2, 1) = -(A(0) * A(4) - A(3) * A(1));
			A_Inv(2, 2) = A(0) * A(2) - A(1) * A(1);
			A_Inv(0, 1) = A_Inv(1, 0);
			A_Inv(0, 2) = A_Inv(2, 0);
			A_Inv(1, 2) = A_Inv(2, 1);

			//2) Calculate determinant of matrix A
			T det = A(0) * (A(2) * A(5) - A(4) * A(4))
				- A(1) * (A(1) * A(5) - A(4) * A(3))
				+ A(3) * (A(1) * A(4) - A(2) * A(3));

			//3) Return A^-1 = 1/det * adj(A)
			A_Inv /= det;
			return A_Inv;
		}
		else
		{
			std::cout << "N = " << N << " is too large for explicit_small_Inv. A returned" << std::endl;
			return A;
		}
	}


	//////////////////////////
	//Hashing for TinyVectors

	///std::pair hash (from boost, for TinyVectors).
	template <class T>
	inline void tv_hash_combine(std::size_t& seed, const T& v)
	{
		std::hash<T> hasher;
		seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
	}
	///Hash for TinyVectors to be used as keys for an unordered map/set.
	struct tv_Hash
	{
		inline std::size_t operator()(const blitz::TinyVector<int, 2>& tv) const
		{
			std::size_t seed = 0;
			for (int dim = 0; dim < 2; ++dim)
				tv_hash_combine(seed, tv(dim));
			return seed;
		}
	};
	///Equal for TinyVectors to be used as keys for an unordered map/set.
	struct tv_Equal {
		inline bool operator()(const blitz::TinyVector<int, 2>& lhs,
			const blitz::TinyVector<int, 2>& rhs) const
		{
			return lhs(0) == rhs(0) && lhs(1) == rhs(1);
		}
	};


	///Cache prefetcher, for Visual C++ and GCC/clang
	inline void prefetch_read(const void* addr)
	{
#if defined(_MSC_VER)
		_mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T1);
#elif defined(__GNUC__) || defined(__clang__)
		__builtin_prefetch(addr, 0, 1);
#endif
		// silently no-op on unknown compilers
	}


}//end namespace BlockSp::util

#endif // end BLOCKSP_UTILITY_HPP