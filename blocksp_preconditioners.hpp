#ifndef BLOCKSP_PRECONDITIONERS_HPP
#define BLOCKSP_PRECONDITIONERS_HPP

#include "blocksp_containers.hpp"
#include "blocksp_dense_multiply.hpp"
#include "blocksp_multiply.hpp"
#include "blocksp_utility.hpp"

///blocksp_preconditioners.hpp defines BlockSp's preconditioners for the iterative solvers.
///Includes: \n
/// NoPre		- A struct for no preconditioning. Does nothing. \n
/// Diag		- A block diagonal preconditioner. \n
/// SOR			- A block successive over-relaxation preconditioner. \n
/// ILU0		- A block incomplete LU preconditioner with zero fill-in and no pivoting. \n
/// IChol0	- A block incomplete Cholesky preconditioner with zero fill-in and no pivoting. \n
///Additionally, the function makeADInv creates a vector of inverse solvers for the block diagonals.
namespace BlockSp::preConditioners
{

	constexpr bool test_zero_eigenvalues{ false }; ///< if true, test all diagonal blocks for zero eigenvalues.

	///Make a vector of inverses of the block diagonal of a BlockSp matrix
	template<typename Inv, typename T, int B>
	std::vector<Inv> makeADInv(const BlockSp<T, B>& A)
	{
		//Test all diagonal blocks for zero eigenvalues to ensure they are invertable.
		//Will break program if zero eigenvalue found.
		if constexpr (test_zero_eigenvalues)
		{
			ScalarType_t<T> mxc = 0.0;
			ScalarType_t<T> mx = 0.0;
			ScalarType_t<T> mn = std::numeric_limits<ScalarType_t<T>>::max();
			for (int im = 0; im < A.m(); ++im)
			{
				if constexpr (B == 1)
				{
					ScalarType_t<T> val = std::abs(A(im, im)(0, 0));
					mn = std::min(mn, val);
					mx = std::max(mx, val);
				}
				else
				{
					dense::NonSymEigenFactor<T, B> ee(A(im, im));
					mn = std::min(mn, ee.min_mag());
					mx = std::max(mx, ee.max_mag());
					mxc = std::max(mxc, ee.cond_number());
				}
				if (mn < 1.0e-15)
				{
					std::cout << "Zero eigenvalue on diagonal!" << std::endl;
					std::cout << "Cannot invert block " << im << ", goodbye." << std::endl;
					std::exit(EXIT_FAILURE);
				}
			}
			std::cout << "min block eval  " << mn << std::endl;
			std::cout << "max block eval  " << mx << std::endl;
			if (B != 1) std::cout << "max block cond# " << mxc << std::endl;
		}

		std::vector<Inv> AD_Inv(A.m());
		for (int im = 0; im < A.m(); ++im)
			AD_Inv[im] = Inv(A(im, im));
		return AD_Inv;
	}

	///No preconditioner, can also use the lambda [](const auto&){}
	template<typename T, int B, int D = 1>
	struct NoPre
	{
		NoPre() {}
		inline void operator()(solArray<T, B, D>&) const {}
	}; //end NoPre

	///Block diagonal preconditioner.
	///Inverts Ax = b assuming A is purely block diagonal.
	template<typename Inv, typename T, int B, int D = 1>
	struct Diag
	{
		int len;
		std::vector<Inv> AD_Inv;					///< Inverse of diagonal blocks of A.

		Diag() : len{ 0 } {}
		Diag(const BlockSp<T, B* D>& A) : AD_Inv{ makeADInv<Inv>(A) }
		{
			assert(A.m() == A.n());
			len = A.m();
		}

		//invert AD x = b, where AD = block diag of A.
		//inputs right-hand-side b, outputs x.
		void invert(solArray<T, B, D>& x) const
		{
			for (int i = 0; i < len; ++i)
			{
				if constexpr (D == 1) AD_Inv[i].invmul(x(i));
				else
				{
					auto x_loc{ util::tvConcat<T, B, D>(x(i)) };
					AD_Inv[i].invmul(x_loc);
					for (int dim = 0; dim < D; ++dim)
						x(i, dim) = util::tvSubset<T, B, D>(x_loc, dim);
				}
			}
		}

		inline void operator()(solArray<T, B, D>& r) const
		{
			invert(r);
		}

	}; //end Diag


	///Block successive over-relaxation (SOR) preconditioner.
	///Can also be used as a stand alone solver.
	template<typename Inv, typename T, int B, int D = 1>
	struct SOR
	{
		std::vector<Inv> AD_Inv;						///< Inverse of diagonal blocks of A.
		const BlockSp<T, B* D>* A_Ptr;			///< Pointer to matrix A.
		const BlockSp_Compr<T, B* D>* ACompr_Ptr; ///< Pointer to compressed matrix
		ScalarType_t<T> m_w;								///< SOR damping parameter.		
		int m_nu;														///< Default number of SOR iterations.
		int m_type;													///< SOR types: <0 backward only; >0 forward only, 0 symmetric.

		SOR() : A_Ptr{ nullptr }, ACompr_Ptr{ nullptr }, m_w{ 0.0 }, m_nu{ 0 }, m_type{ 0 } {}
		SOR(const BlockSp<T, B* D>& A, ScalarType_t<T> w, int nu, int type = 0) :
			AD_Inv{ makeADInv<Inv>(A) }, A_Ptr{ &A }, ACompr_Ptr{ nullptr }, m_w{ w }, m_nu{ nu }, m_type{ type }
		{
			assert(A.m() == A.n());
		}

		///One iteration of SOR in the forward direction.
		///Works to solve Ax = b, overwrites initial guess x.
		///w = SOR damping parameter.
		template<typename SpMat>
		void oneSORForw(const SpMat& A, solArray<T, B, D>& x, const solArray<T, B, D>& b, ScalarType_t<T> w) const
		{
			assert(A.m() == b.len() && A.n() == x.len());

			//forward SOR iterations
			for (int im = 0; im < A.m(); ++im)
			{
				blitz::TinyVector<T, B* D> ph(0.0);
				A.for_each_row_entry(im, [&](int col, const auto& data)
					{
						if (col != im)
						{
							if constexpr (D == 1) ph += dense::matVec<T, B* D, B* D>(data, x(col));
							else
							{
								auto xLoc{ util::tvConcat<T, B, D>(x(col)) };
								ph += dense::matVec<T, B* D, B* D>(data, xLoc);
							}
						}
					});
				if constexpr (D == 1) ph = b(im) - ph;
				else ph = util::tvConcat<T, B, D>(b(im)) - ph;
				AD_Inv[im].invmul(ph);
				if constexpr (D == 1)
					x(im) += -w * x(im) + w * ph;
				else for (int dim = 0; dim < D; ++dim)
					x(im, dim) += -w * x(im, dim) + w * util::tvSubset<T, B, D>(ph, dim);
			}
		};

		///One iteration of SOR in the backward direction.
		///Works to solve Ax = b, overwrites initial guess x.
		///w = SOR damping parameter.
		template<typename SpMat>
		void oneSORBack(const SpMat& A, solArray<T, B, D>& x, const solArray<T, B, D>& b, ScalarType_t<T> w) const
		{
			assert(A.m() == b.len() && A.n() == x.len());

			//backward SOR iterations
			for (int im = A.m() - 1; im >= 0; --im)
			{
				blitz::TinyVector<T, B* D> ph(0.0);					//ph stands for placeholder
				A.for_each_row_entry(im, [&](int col, const auto& data)
					{
						if (col != im)
						{
							if constexpr (D == 1) ph += dense::matVec<T, B* D, B* D>(data, x(col));
							else
							{
								auto xLoc{ util::tvConcat<T, B, D>(x(col)) };
								ph += dense::matVec<T, B* D, B* D>(data, xLoc);
							}
						}
					});
				if constexpr (D == 1) ph = b(im) - ph;
				else ph = util::tvConcat<T, B, D>(b(im)) - ph;
				AD_Inv[im].invmul(ph);
				if constexpr (D == 1)
					x(im) += -w * x(im) + w * ph;
				else for (int dim = 0; dim < D; ++dim)
					x(im, dim) += -w * x(im, dim) + w * util::tvSubset<T, B, D>(ph, dim);
			}
		};

		///Perform nu iterations of forward SOR. Overwrites initial guess x.
		template<typename SpMat>
		inline void runSORF(const SpMat& A, solArray<T, B, D>& x, const solArray<T, B, D>& b, ScalarType_t<T> w, int nu) const
		{
			for (int iter = 0; iter < nu; ++iter)
				oneSORForw(A, x, b, w);
		};

		///Perform nu iterations of backward SOR. Overwrites initial guess x.
		template<typename SpMat>
		inline void runSORB(const SpMat& A, solArray<T, B, D>& x, const solArray<T, B, D>& b, ScalarType_t<T> w, int nu) const
		{
			for (int iter = 0; iter < nu; ++iter)
				oneSORBack(A, x, b, w);
		};

		///Perform a cycle of forward and backward SOR. For nu iterations each, overwrites initial guess x.
		template<typename SpMat>
		inline void runSOR(const SpMat& A, solArray<T, B, D>& x, const solArray<T, B, D>& b, ScalarType_t<T> w, int nu) const
		{
			runSORF(A, x, b, w, nu);
			runSORB(A, x, b, w, nu);
		};

		///operator(), for use of SOR as a preconditioner for pCG or pGMRES.
		///Solves Ax_0 = r with initial guess x_0 = 0.
		inline void operator()(solArray<T, B, D>& r) const
		{
			if (ACompr_Ptr) //use compressed matrix if available
			{
				solArray<T, B, D> x{ r.len() };
				if (m_type < 0) runSORB(*ACompr_Ptr, x, r, m_w, m_nu);			//backward SOR only
				else if (m_type == 0) runSOR(*ACompr_Ptr, x, r, m_w, m_nu); //symmetric SOR
				else runSORF(*ACompr_Ptr, x, r, m_w, m_nu);								  //forward SOR only
				r = x;
			}
			else
			{
				solArray<T, B, D> x{ r.len() };
				if (m_type < 0) runSORB(*A_Ptr, x, r, m_w, m_nu);			 //backward SOR only
				else if (m_type == 0) runSOR(*A_Ptr, x, r, m_w, m_nu); //symmetric SOR
				else runSORF(*A_Ptr, x, r, m_w, m_nu);								 //forward SOR only
				r = x;
			}
		}

		inline void change_w(int w_new) { m_w = w_new; }
		inline void change_nu(int nu_new) { m_nu = nu_new; }
		inline void change_type(int type_new) { m_type = type_new; }
		inline void set_compr_ptr(const BlockSp_Compr<T, B* D>& ACompr) { ACompr_Ptr = &ACompr; }

	}; //end SOR


	///Block ILU0 preconditioner.
	///Approximates Ax = b by factoring A via ILU decomposition without fill-in. 
	///No pivoting.
	template<typename T, int B, int D = 1>
	struct ILU0
	{
		BlockSp_Compr<T, B* D> L; 										///< The L0 factorization. In compressed form.
		BlockSp_Compr<T, B* D> U; 										///< The U0 factorization. In compressed form.
		std::vector<std::unordered_set<int>> rowInd;	///< The nonzero row indices for each column
		std::vector<dense::LUFactor<T, B* D>> D_Inv;	///< The inverse of the Block diagonal of LU.
		int n;																				///< The BlockSp matrix size.

		ILU0() : n{ 0 } {}
		ILU0(const BlockSp<T, B* D>& A)
		{
			assert(A.m() == A.n());
			n = A.n(); D_Inv.resize(n);
			rowInd = A.rowIndicesByCol();		//non-zero row indices of A, sorted by column.
			factor(A);											//perform ILU0 factoriation		
		}
		ILU0& operator= (const ILU0& other)
		{
			L = other.L;
			U = other.U;
			rowInd = other.rowInd;
			D_Inv = other.D_Inv;
			n = other.n;
			return *this;
		}

		//Perform an inplace ILU0 factorization of matrix A
		void factor(const BlockSp<T, B* D>& A)
		{
			BlockSp<T, B* D> LU{ A };
			for (int i = 0; i < n; ++i)
			{
				//Factor below diagonal, only for A's non-zero entries.
				//Note: Gaussian elimination calls for application of the "right" inverse,
				//	i.e, L_ki = A_ki * A^{-1}_ii
				D_Inv[i] = dense::LUFactor<T, B* D>(LU(i, i));
				for (const auto& k : rowInd[i])
					if (k > i) D_Inv[i].r_invmul(LU(k, i));

				//Gaussian eliminate the remaining rows/columns
				for (const auto& k : rowInd[i]) if (k > i)
					for (const auto& j : LU.get_entries(k)) if (j.colInd() > i)
						if (rowInd[j.colInd()].contains(i))
							LU(k, j.colInd()) -= dense::matMat(LU(k, i), LU(i, j.colInd()));
			}
			L = std::move(LU.lowTri(-1).compress_and_move<false>()); //compress the LU factor
			U = std::move(LU.upTri(1).compress_and_move<false>());
		}

		///Solve LU x = b
		void invmul(solArray<T, B, D>& x) const
		{
			//Solve L^-1 via forward substitution
			for (int im = 1; im < n; ++im)
				L.for_each_row_entry(im, [&](int col, const auto& data)
					{
						if constexpr (D == 1) x(im) -= dense::matVec(data, x(col));
						else
						{
							auto x_loc{ util::tvConcat<T, B, D>(x(col)) };
							x_loc = dense::matVec(data, x_loc);
							for (int dim = 0; dim < D; ++dim)
								x(im, dim) -= util::tvSubset<T, B, D>(x_loc, dim);
						}
					});
			//Diagonals of L are identity matrices.

			int prefetch = 2;
			//Solve U^-1 via backward substitution
			for (int im = n - 1; im >= 0; --im)
			{
				//Prefetch x entries for the next rows of U
				if (im - prefetch >= 0)
					U.for_each_row_entry(im - prefetch, [&](int col, const auto&)
						{
							util::prefetch_read(&x(col));
						});

				U.for_each_row_entry(im, [&](int col, const auto& data)
					{
						if constexpr (D == 1) x(im) -= dense::matVec(data, x(col));
						else
						{
							auto x_loc{ util::tvConcat<T, B, D>(x(col)) };
							x_loc = dense::matVec(data, x_loc);
							for (int dim = 0; dim < D; ++dim)
								x(im, dim) -= util::tvSubset<T, B, D>(x_loc, dim);
						}
					});

				//Invert diagonals
				if constexpr (D == 1) D_Inv[im].invmul(x(im));
				else
				{
					auto x_loc{ util::tvConcat<T, B, D>(x(im)) };
					D_Inv[im].invmul(x_loc);
					for (int dim = 0; dim < D; ++dim)
						x(im, dim) = util::tvSubset<T, B, D>(x_loc, dim);
				}
			}
		}

		///operator(), for use of ILU0 as a preconditioner for pCG or pGMRES.
		inline void operator()(solArray<T, B, D>& r) const
		{
			invmul(r);
		}

	}; //end ILU0

	///Block Incomplete Cholesky(0) preconditioner.
	///Approximates Ax = b by factoring A via Cholesky decomposition LL^T without fill-in.
	///No pivoting. Using lower triangular matrices.
	///For symmetric positive definite (SPD) linear systems, 
	/// which are the user's responsibility to provide.
	///If LT, then a copy of the transpose of matrix L is stored.
	/// This will speed up the solver at the cost of extra memory usage.
	template<typename T, int B, int D = 1, bool LT = true>
	struct IChol0
	{
		BlockSp_ComprSorted<T, B* D> L;								///< The IChol0 factorization
		BlockSp_ComprSorted<T, B* D> Lt;							///< The transpose of L
		std::vector<blitz::TinyMatrix<T, B* D, B* D>> L_D;  ///< The diagonal of L
		std::vector<blitz::TinyMatrix<T, B* D, B* D>> Lt_D; ///< The diagonal of L^T
		std::vector<std::unordered_set<int>> rowInd;	///< The nonzero row indices for each column
		int n;																				///< The BlockSp matrix size.

		IChol0() : n{ 0 } {}

		///Construct IChol0 from an SPD matrix A. A is assumed to be full, not triangular. 
		IChol0(const BlockSp<T, B* D>& A)
		{
			static_assert(!is_complex_v<T>,
				"IChol0 is for real SPD systems only. Use ILU0 for complex Hermitian systems.");
			assert(A.m() == A.n());
			auto L_temp{ A.lowTri() };		//extract lower triangular of A.	
			n = A.n();
			L_D.resize(n);
			rowInd = A.rowIndicesByCol(); //non-zero row indices of A, sorted by column.
			factor(L_temp);								//create IChol0 factorization L.

			//Extract diagonals of L
			for (int i = 0; i < n; ++i)
			{
				L_D[i] = L_temp(i, i);
				L_temp.erase(i, i);
			}

			//If LT, store transpose of L
			BlockSp<T, B* D> Lt_temp;
			if constexpr (LT)							//store L^T
			{
				Lt_temp = bs_tr<T, B* D>(L_temp); //transpose L
				Lt_D.resize(n);
				for (int i = 0; i < n; ++i)
					Lt_D[i] = dense::transpose(L_D[i]);
			}

			//Create row indices of L. Compress L
			rowInd = L_temp.rowIndicesByCol(true); //update row indices to be lower triangular L.
			L = std::move(L_temp.compress_and_move<true>()); //compress and sort the matrix L
			if constexpr (LT) Lt = std::move(Lt_temp.compress_and_move<true>());
		}
		IChol0& operator= (const IChol0& other)
		{
			L = other.L;
			L_D = other.L_D;
			if constexpr (LT)
			{
				Lt = other.Lt;
				Lt_D = other.Lt_D;
			}
			rowInd = other.rowInd;
			n = other.n;
			return *this;
		}

		///Perform an inplace IChol0 factorization of matrix A_l.
		///Assumes that A_l is the lower block triangular of an SPD matrix.
		void factor(BlockSp<T, B* D>& A_l)
		{
			int constexpr BD = B * D;
			for (int i = 0; i < n; ++i)
			{
				//Cholesky factor the diagonal L_ii
				if constexpr (BD == 1) A_l(i, i) = std::sqrt(A_l(i, i)(0, 0));
				else
				{
					lapack::potrf<T>(LAPACK_ROW_MAJOR, 'L', BD, A_l(i, i).data(), BD);
					for (int ib = 0; ib < BD; ++ib)
						for (int jb = ib + 1; jb < BD; ++jb)
							A_l(i, i)(ib, jb) = T(0.0); //set upper triangular of diag block to zero
				}

				//Factor below diagonal, only for A's non-zero entries.
				//Note: Gaussian elimination calls for application of the "right" inverse,
				//	i.e, L_ki = A_ki * L^{-T}_ii
				for (const auto& k : rowInd[i]) if (k > i)
				{
					if constexpr (BD == 1) A_l(k, i) /= A_l(i, i)(0, 0);
					else
					{
						//right inverse solved via A^T X^T = Y^T
						dense::transpose_in_place(A_l(k, i)); //make Y^T
						Lii_Inv_forward_subst(A_l(i, i), A_l(k, i));
						dense::transpose_in_place(A_l(k, i)); //get X
					}
				}

				//Gaussian eliminate the remaining rows/columns
				for (const auto& k : rowInd[i]) if (k > i)
					for (const auto& j : A_l.get_entries(k)) if (j.colInd() > i && j.colInd() <= k)
						if (rowInd[j.colInd()].contains(i))
							A_l(k, j.colInd()) -= dense::matMat(A_l(k, i), dense::transpose(A_l(j.colInd(), i)));
			}
		}

		///Solve LL^T x = b
		void invmul(solArray<T, B, D>& x) const
		{
			//Solve L^-1 via forward substitution
			for (int im = 0; im < n; ++im)
			{
				L.for_each_row_entry(im, [&](int col, const auto& data)
					{
						if constexpr (D == 1) x(im) -= dense::matVec(data, x(col));
						else
						{
							auto x_loc{ util::tvConcat<T, B, D>(x(col)) };
							x_loc = dense::matVec(data, x_loc);
							for (int dim = 0; dim < D; ++dim)
								x(im, dim) -= util::tvSubset<T, B, D>(x_loc, dim);
						}
					});

				//Invert diagonals L_ii, triangular solve via forward substitution
				if constexpr (D == 1)	Lii_Inv_forward_subst(L_D[im], x(im));
				else
				{
					auto x_loc{ util::tvConcat<T, B, D>(x(im)) };
					Lii_Inv_forward_subst(L_D[im], x_loc);
					for (int dim = 0; dim < D; ++dim)
						x(im, dim) = util::tvSubset<T, B, D>(x_loc, dim);
				}
			}

			int prefetch = 2;
			//Solve L^T^-1 via backward substitution
			for (int im = n - 1; im >= 0; --im)
			{
				if constexpr (LT) //If L^T is stored, contiguous row storage (faster)
				{
					// Prefetch x entries for the next row of Lt
					if (im - prefetch >= 0)
						Lt.for_each_row_entry(im - prefetch, [&](int col, const auto&)
							{
								util::prefetch_read(&x(col));
							});

					Lt.for_each_row_entry(im, [&](int col, const auto& data)
						{
							if constexpr (D == 1) x(im) -= dense::matVec(data, x(col));
							else
							{
								auto x_loc{ util::tvConcat<T, B, D>(x(col)) };
								x_loc = dense::matVec(data, x_loc);
								for (int dim = 0; dim < D; ++dim)
									x(im, dim) -= util::tvSubset<T, B, D>(x_loc, dim);
							}
						});
				}
				else //use a scattered indexing approach (slower) 
					for (const auto& row : rowInd[im]) // if (row != im)
					{
						if constexpr (D == 1) x(im) -= dense::matVec_transpose(L(row, im), x(row));
						else
						{
							auto x_loc{ util::tvConcat<T, B, D>(x(row)) };
							x_loc = dense::matVec_transpose(L(row, im), x_loc);
							for (int dim = 0; dim < D; ++dim)
								x(im, dim) -= util::tvSubset<T, B, D>(x_loc, dim);
						}
					}

				//Invert diagonals, L_ii^T, triangular solve via backward substitution
				if constexpr (D == 1)
					if constexpr (LT) LiiT_Inv_backward_subst_versionLT(Lt_D[im], x(im));
					else LiiT_Inv_backward_subst(L_D[im], x(im));
				else
				{
					auto x_loc{ util::tvConcat<T, B, D>(x(im)) };
					if constexpr (LT) LiiT_Inv_backward_subst_versionLT(Lt_D[im], x_loc);
					else LiiT_Inv_backward_subst(L_D[im], x_loc);
					for (int dim = 0; dim < D; ++dim)
						x(im, dim) = util::tvSubset<T, B, D>(x_loc, dim);
				}
			}
		}

		///operator(), for use of IChol0 as a preconditioner for pCG.
		inline void operator()(solArray<T, B, D>& r) const
		{
			invmul(r);
		}

		//Solve L_ii X = Y via forward substitution, where L_ii is a Cholesky factored block diagonal.
		//X modified in place
		template<int N>
		void Lii_Inv_forward_subst(const blitz::TinyMatrix<T, N, N>& Lii,
			blitz::TinyMatrix<T, N, N>& X)
		{
			for (int i = 0; i < N; ++i)
				for (int k = 0; k < N; ++k)
				{
					for (int j = 0; j < i; ++j)
						X(i, k) -= Lii(i, j) * X(j, k);
					X(i, k) /= Lii(i, i);
				}
		}

		template<int N>
		void Lii_Inv_forward_subst(const blitz::TinyMatrix<T, N, N>& Lii,
			blitz::TinyVector<T, N>& x) const
		{
			for (int i = 0; i < N; ++i)
			{
				for (int j = 0; j < i; ++j)
					x(i) -= Lii(i, j) * x(j);
				x(i) /= Lii(i, i);
			}
		}

		//Solve L_ii^T x = t via backward substitution, 
		// where L_ii is a Cholesky factored block diagonal (lower triangular).
		//x modified in place
		template<int N>
		void LiiT_Inv_backward_subst(const blitz::TinyMatrix<T, N, N>& Lii,
			blitz::TinyVector<T, N>& x) const
		{
			for (int i = N - 1; i >= 0; --i)
			{
				for (int j = i + 1; j < N; ++j)
					x(i) -= Lii(j, i) * x(j);
				x(i) /= Lii(i, i);
			}
		}

		//Solve L_ii^T x = t via backward substitution, 
		//Used when LT is included. Here L^T_ii is a Cholesky factored block diagonal (upper triangular).
		//x modified in place
		template<int N>
		void LiiT_Inv_backward_subst_versionLT(const blitz::TinyMatrix<T, N, N>& LTii,
			blitz::TinyVector<T, N>& x) const
		{
			for (int i = N - 1; i >= 0; --i)
			{
				for (int j = i + 1; j < N; ++j)
					x(i) -= LTii(i, j) * x(j);
				x(i) /= LTii(i, i);
			}
		}

	}; //end IChol0


	///Try creating your own precondtioner.
	///Many applied math prizes can be won here.
	template<typename T, int B, int D = 1>
	struct YOUR_PRECONDITIONER
	{
		///////////////////////
		//Your member variables


		///Overload operator(), used in pCG and pGMRES to apply preconditioner.
		inline void operator()(const solArray<T, B, D>& r) const
		{
			//Your code here

		}

	}; //end YOUR_PRECONDITIONER

} //end namespace BlockSp::preConditioners

#endif //end BLOCKSP_PRECONDITIONERS_HPP