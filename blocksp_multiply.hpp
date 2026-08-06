#ifndef	BLOCKSP_MULTIPLY_HPP
#define BLOCKSP_MULTIPLY_HPP

#include "blocksp_containers.hpp"
#include "blocksp_dense_multiply.hpp"
#include "blocksp_utility.hpp"

//blocksp_multiply.hpp provides BlockSp multiplication operations:
//1) Identity	 (bs_id)
//2) Transpose (bs_tr)
//3) Matrix-matrix multiplication	(bsmm) 
//4) Matrix-vector multiplication	(bsmv)
//Corresponding dense matrix operations are in blocksp_dense_multiply.hpp 

namespace BlockSp
{
	///Details of BlockSp multiplication operations.
	namespace mult_detail
	{
		///BlockSp identity matrix.
		template<typename T, int M>
		BlockSp<T, M> blkSpIdent(int size)
		{
			auto id{ dense::identity<T, M>() };
			BlockSp<T, M> blkId(size);
			for (int im = 0; im < size; ++im) blkId(im, im) = id;
			return blkId;
		}

		///Transpose a BlockSp matrix.
		template<typename T, int M>
		BlockSp<T, M> blkSpTranspose(const BlockSp<T, M>& other)
		{
			BlockSp<T, M> tran(other.n(), other.m());
			for (int im = 0; im < other.m(); ++im)
				for (int in = 0; in < other.colSize(im); ++in)
				{
					int colInd = other.colInd(im, in);
					tran(colInd, im) = dense::transpose<T, M, M>(other(im, colInd));
				}
			return tran;
		}


		///////////////////////////////////////
		//Block Sparse Matrix-Matrix Multiply
		///////////////////////////////////////

		///Block sparse matrix-matrix multiplication (bsmm). 
		///C = A * B or C = A^T * B. bool Atran transposes A. 
		template<typename T, int M>
		BlockSp<T, M> blkSpMatMat(const BlockSp<T, M>& A, const BlockSp<T, M>& B, bool Atran = false)
		{
			if (!Atran) assert(A.n() == B.m());
			else assert(A.m() == B.m());

			BlockSp<T, M> C;
			int m = A.m();
			int n = A.n();
			if (!Atran)	C = BlockSp<T, M>(m, B.n());
			else C = BlockSp<T, M>(n, B.n());

			if (!Atran)
			{
				//pre determine sparsity structure 
				std::vector<std::unordered_set<int>> sparsity(m);
				for (int imA = 0; imA < A.m(); ++imA) //scan each row of A 
				{	//scan each column in that row of A, its col number...
					for (int inA = 0; inA < A.colSize(imA); ++inA)
					{
						int colIndA = A.colInd(imA, inA);
						//...becomes the row number in B, scan each column in that row of B 
						for (int inB = 0; inB < B.colSize(colIndA); ++inB)
						{
							int colIndB = B.colInd(colIndA, inB);
							sparsity[imA].emplace(colIndB);
						}
					}
				}
				//reserve space in the BlockSp's vectors, matching the sparsity structure 
				for (int imA = 0; imA < m; ++imA)
					C.reserve(imA, sparsity[imA].size());

				//perform matrix-matrix product
				//scan each row of A 
				for (int imA = 0; imA < m; ++imA)
				{	//scan each column in that row of A, its col number...
					for (int inA = 0; inA < A.colSize(imA); ++inA)
					{
						int colIndA = A.colInd(imA, inA);
						//...becomes the row number in B, scan each column in that row of B 
						for (int inB = 0; inB < B.colSize(colIndA); ++inB)
						{
							int colIndB = B.colInd(colIndA, inB);
							C(imA, colIndB) += dense::matMat<T, M>(A(imA, colIndA), B(colIndA, colIndB));
						}
					}
				}
			}
			else
			{
				//pre determine sparsity structure 
				std::vector<std::unordered_set<int>> sparsity(n);
				for (int imA = 0; imA < A.m(); ++imA) //scan each row of A 
				{	//scan each column in that row of A, for transpose its row number is now col num which ...
					for (int inA = 0; inA < A.colSize(imA); ++inA)
					{
						int colIndA = A.colInd(imA, inA);
						//...becomes the row number in B, scan each column in that row of B 
						for (int inB = 0; inB < B.colSize(imA); ++inB)
						{
							int colIndB = B.colInd(imA, inB);
							sparsity[colIndA].emplace(colIndB);
						}
					}
				}
				//reserve space in the BlockSp's vectors, matching the sparsity structure 
				for (int inA = 0; inA < n; ++inA)
					C.reserve(inA, sparsity[inA].size());

				//perform matrix-matrix product
				//scan each row of A 
				for (int imA = 0; imA < A.m(); ++imA)
				{	//scan each column in that row of A, for trans its row number is now col num which ...
					for (int inA = 0; inA < A.colSize(imA); ++inA)
					{
						int colIndA = A.colInd(imA, inA);
						//...becomes the row number in B, scan each column in that row of B 
						for (int inB = 0; inB < B.colSize(imA); ++inB)
						{
							int colIndB = B.colInd(imA, inB);
							C(colIndA, colIndB) += dense::matMat<T, M>(A(imA, colIndA), B(imA, colIndB), true);
						}
					}
				}
			}
			return C;
		}

		///Triple blkSpMatMat D = A*B*C or A^T * B * C, the latter is often seen in FE problems.
		template<typename T, int M>
		BlockSp<T, M> blkSpMatMat(const BlockSp<T, M>& A, const BlockSp<T, M>& B, const BlockSp<T, M>& C, bool Atran = false)
		{
			if (!Atran) assert(A.n() == B.m() && B.n() == C.m());
			else assert(A.m() == B.m() && B.n() == C.m());
			BlockSp<T, M> D(A.m(), C.n());
			D = blkSpMatMat<T, M>(A, blkSpMatMat<T, M>(B, C), Atran);
			return D;
		}

		///Quadruple blkSpMatMat E = A*B*C*D  or A^T*B*C*D.
		template<typename T, int M>
		BlockSp<T, M> blkSpMatMat(const BlockSp<T, M>& A, const BlockSp<T, M>& B,
			const BlockSp<T, M>& C, const BlockSp<T, M>& D, bool Atran = false)
		{
			if (!Atran) assert(A.n() == B.m() && B.n() == C.m() && C.n() == D.m());
			else assert(A.m() == B.m() && B.n() == C.m() && C.n() == D.m());
			BlockSp<T, M> E(A.m(), D.n());
			E = blkSpMatMat<T, M>(A, blkSpMatMat<T, M>(B, C, D), Atran);
			return E;
		}


		///////////////////////////////////
		//BlockSp Matrix-Vector multiply
		///////////////////////////////////

		///Matrix vector product (bsmv) between BxB BlockSp matrix with solArray of size BxD.
		//Multiply either a single dimension (dim >= 0) or all dimensions (set dim < 0, default).
		///A^T if transpose. 
		///This function also encapsulates scalar cases, D = 1.
		template<typename T, int B, int D = 1, typename SpMat = BlockSp_Compr<T, B>>
		solArray<T, B, D> blkSpMatVec(const SpMat& A, const solArray<T, B, D>& x,
			int dim = -1, bool transpose = false)
		{
			if (!transpose) assert(A.n() == x.extent(0));
			else assert(A.m() == x.extent(0));
			assert(dim < D);

			if (!transpose)
			{
				solArray<T, B, D> b{ A.m() };

				if (dim >= 0)
					for (int im = 0; im < A.m(); ++im)
						A.for_each_row_entry(im, [&](int col, const auto& data)
							{
								b(im, dim) += dense::matVec<T, B, B>(data, x(col, dim));
							});
				else
					for (int im = 0; im < A.m(); ++im)
						A.for_each_row_entry(im, [&](int col, const auto& data)
							{
								for (int dim2 = 0; dim2 < D; ++dim2)
									b(im, dim2) += dense::matVec<T, B, B>(data, x(col, dim2));
							});
				return b;
			}
			else //transpose version
			{
				solArray<T, B, D> b{ A.n() };
				if (dim >= 0)
					for (int im = 0; im < A.m(); ++im)
						A.for_each_row_entry(im, [&](int col, const auto& data)
							{
								b(col, dim) += dense::matVec_transpose<T, B, B>(data, x(im, dim));
							});
				else
					for (int im = 0; im < A.m(); ++im)
						A.for_each_row_entry(im, [&](int col, const auto& data)
							{
								for (int dim2 = 0; dim2 < D; ++dim2)
									b(col, dim2) += dense::matVec_transpose<T, B, B>(data, x(im, dim2));
							});
				return b;
			}
		}

		///Matrix vector product (bsmv) between BDxBD BlockSp matrix with solArray of size BxD.
		///i.e. regular matVec. A^T if transpose
		template<typename T, int B, int D, typename SpMat = BlockSp_Compr<T, B* D>> requires (D > 1)
			solArray<T, B, D> blkSpMatVec(const SpMat& A, const solArray<T, B, D>& x, bool transpose = false)
		{
			if (!transpose) assert(A.n() == x.extent(0));
			else assert(A.m() == x.extent(0));

			if (!transpose)
			{
				solArray<T, B, D> b{ A.m() };
				for (int im = 0; im < A.m(); ++im)
					A.for_each_row_entry(im, [&](int col, const auto& data)
						{
							auto xLoc(util::tvConcat<T, B, D>(x(col)));
							blitz::TinyVector<T, B* D> ph{ dense::matVec<T, B * D, B * D>(data, xLoc) };
							for (int dim = 0; dim < D; ++dim)
								b(im, dim) += util::tvSubset<T, B, D>(ph, dim);
						});
				return b;
			}
			else //transpose version
			{
				solArray<T, B, D> b{ A.n() };
				for (int im = 0; im < A.m(); ++im)
				{
					auto xLoc(util::tvConcat<T, B, D>(x(im)));
					A.for_each_row_entry(im, [&](int col, const auto& data)
						{
							blitz::TinyVector<T, B* D> ph{ dense::matVec_transpose<T, B * D, B * D>(data, xLoc) };
							for (int dim = 0; dim < D; ++dim)
								b(col, dim) += util::tvSubset<T, B, D>(ph, dim);
						});
				}
				return b;
			}
		}

		///Void version - matrix vector product (bsmv) between BxB BlockSp matrix with solArray of size BxD.
		///Multiply either a single dimension (dim >= 0) or all dimensions (set dim < 0, default).
		///A^T if transpose. 
		///Adds to solArray b. This function also encapsulates scalar cases, D = 1.
		template<typename T, int B, int D = 1, typename SpMat = BlockSp_Compr<T, B>>
		void blkSpMatVec(solArray<T, B, D>& b, const SpMat& A, const solArray<T, B, D>& x,
			int dim = -1, bool transpose = false)
		{
			if (!transpose)
			{
				assert(A.n() == x.extent(0));
				assert(A.m() == b.extent(0));
			}
			else
			{
				assert(A.m() == x.extent(0));
				assert(A.n() == b.extent(0));
			}

			if (!transpose)
				if (dim >= 0)
					for (int im = 0; im < A.m(); ++im)
						A.for_each_row_entry(im, [&](int col, const auto& data)
							{
								b(im, dim) += dense::matVec<T, B, B>(data, x(col, dim));
							});
				else
					for (int im = 0; im < A.m(); ++im)
						A.for_each_row_entry(im, [&](int col, const auto& data)
							{
								for (int dim2 = 0; dim2 < D; ++dim2)
									b(im, dim2) += dense::matVec<T, B, B>(data, x(col, dim2));
							});

			else //transpose version
				if (dim >= 0)
					for (int im = 0; im < A.m(); ++im)
						A.for_each_row_entry(im, [&](int col, const auto& data)
							{
								b(col, dim) += dense::matVec_transpose<T, B, B>(data, x(im, dim));
							});
				else
					for (int im = 0; im < A.m(); ++im)
						A.for_each_row_entry(im, [&](int col, const auto& data)
							{
								for (int dim2 = 0; dim2 < D; ++dim2)
									b(col, dim2) += dense::matVec_transpose<T, B, B>(data, x(im, dim2));
							});
		}

		///Void-version matrix vector product (bsmv) between BDxBD BlockSp matrix with solArray of size BxD.
		///i.e. regular matVec. A^T if transpose. Adds to solArray b.
		template<typename T, int B, int D, typename SpMat = BlockSp_Compr<T, B* D>> requires (D > 1)
			void blkSpMatVec(solArray<T, B, D>& b, const SpMat& A, const solArray<T, B, D>& x, bool transpose = false)
		{
			if (!transpose)
			{
				assert(A.n() == x.extent(0));
				assert(A.m() == b.extent(0));
			}
			else
			{
				assert(A.m() == x.extent(0));
				assert(A.n() == b.extent(0));
			}

			if (!transpose)
				for (int im = 0; im < A.m(); ++im)
					A.for_each_row_entry(im, [&](int col, const auto& data)
						{
							auto xLoc(util::tvConcat<T, B, D>(x(col)));
							blitz::TinyVector<T, B* D> ph{ dense::matVec<T, B * D, B * D>(data, xLoc) };
							for (int dim = 0; dim < D; ++dim)
								b(im, dim) += util::tvSubset<T, B, D>(ph, dim);
						});
			else //transpose verision
				for (int im = 0; im < A.m(); ++im)
				{
					auto xLoc(util::tvConcat<T, B, D>(x(im)));
					A.for_each_row_entry(im, [&](int col, const auto& data)
						{
							blitz::TinyVector<T, B* D> ph{ dense::matVec_transpose<T, B * D, B * D>(data, xLoc) };
							for (int dim = 0; dim < D; ++dim)
								b(col, dim) += util::tvSubset<T, B, D>(ph, dim);
						});
				}
		}

	}//end namespace mult_detail


	/////////////////////////
	//Acronyms
	/////////////////////////

	///BlockSp identity matrix.
	template<typename T, int M>
	BlockSp<T, M> bs_id(int size) { return mult_detail::blkSpIdent<T, M>(size); }

	///BlockSp transpose.
	template<typename T, int M>
	BlockSp<T, M> bs_tr(const BlockSp<T, M>& other) { return mult_detail::blkSpTranspose<T, M>(other); }

	///BlockSp matrix-matrix multiply.
	template<typename T, int M>
	BlockSp<T, M> bsmm(const BlockSp<T, M>& A, const BlockSp<T, M>& B, bool Atran = false)
	{
		return mult_detail::blkSpMatMat<T, M>(A, B, Atran);
	}

	///BlockSp matrix-matrix multiply.
	template<typename T, int M>
	BlockSp<T, M> bsmm(const BlockSp<T, M>& A, const BlockSp<T, M>& B, const BlockSp<T, M>& C, bool Atran = false)
	{
		return mult_detail::blkSpMatMat<T, M>(A, B, C, Atran);
	}

	///BlockSp matrix-matrix multiply.
	template<typename T, int M>
	BlockSp<T, M> bsmm(const BlockSp<T, M>& A, const BlockSp<T, M>& B,
		const BlockSp<T, M>& C, const BlockSp<T, M>& D, bool Atran = false)
	{
		return mult_detail::blkSpMatMat<T, M>(A, B, C, D, Atran);
	}

	///BlockSp matrix-vector multiply.
	template<typename T, int B, int D = 1, typename SpMat = BlockSp_Compr<T, B>>
		requires (std::is_same_v<SpMat, BlockSp<T, B>> || std::is_same_v<SpMat, BlockSp_Compr<T, B>> ||
	std::is_same_v<SpMat, BlockSp_ComprSorted<T, B>>)
		solArray<T, B, D> bsmv(const SpMat& A, const solArray<T, B, D>& x, int dim = -1, bool transpose = false)
	{
		return mult_detail::blkSpMatVec<T, B, D>(A, x, dim, transpose);
	}

	///BlockSp matrix-vector multiply.
	template<typename T, int B, int D, typename SpMat = BlockSp_Compr<T, B* D>> requires (D > 1)
		solArray<T, B, D> bsmv(const SpMat& A, const solArray<T, B, D>& x, bool transpose = false)
	{
		return mult_detail::blkSpMatVec<T, B, D>(A, x, transpose);
	}

	///BlockSp matrix-vector multiply.
	template<typename T, int B, int D = 1, typename SpMat = BlockSp_Compr<T, B>>
		requires (std::is_same<SpMat, BlockSp<T, B>>::value || std::is_same<SpMat, BlockSp_Compr<T, B>>::value ||
	std::is_same_v<SpMat, BlockSp_ComprSorted<T, B>>)
		void bsmv(solArray<T, B, D>& b, const SpMat& A, const solArray<T, B, D>& x, int dim = -1, bool transpose = false)
	{
		mult_detail::blkSpMatVec<T, B, D>(b, A, x, dim, transpose);
	}

	///BlockSp matrix-vector multiply.
	template<typename T, int B, int D, typename SpMat = BlockSp_Compr<T, B* D>> requires (D > 1)
		void bsmv(solArray<T, B, D>& b, const SpMat& A, const solArray<T, B, D>& x, bool transpose = false)
	{
		mult_detail::blkSpMatVec<T, B, D>(b, A, x, transpose);
	}

}//end namespace BlockSp

#endif //end BLOCKSP_MULTIPLY_HPP