#ifndef BLOCKSP_SOLVERS_HPP
#define BLOCKSP_SOLVERS_HPP

#include "blocksp_containers.hpp"
#include "blocksp_deflated_restart.hpp"
#include "blocksp_dense_multiply.hpp"
#include "blocksp_dense_solvers.hpp"
#include "blocksp_preconditioners.hpp"
#include "blocksp_utility.hpp"

//blocksp_solvers.hpp defines BlockSp's fundamental linear system solvers.
//These include: 
//1) The solver engine functions:
//		Preconditioned block conjugate gradient (pCG).
//		Preconditioned block GMRES (pGMRES), with or without deflated restarting.
//		Arnoldi eigenvalue solver for preconditioned systems.	
// 
//2) Pre-built solvers for quick, easy use:
//		SOR_pGMRES_Solver   --  Successive over-relaxation preconditioned GMRES.
//		SOR_pCG_Solver      --  Successive over-relaxation preconditioned conjugate gradient.
//		ILU0_pGMRES_Solver  --  Incomplete LU with zero fill preconditioned GMRES.
//    IChol0_pCG_Solver 	--  Incomplete Cholesky with zero fill preconditioned conjugate gradient.
// 
//3) "ez" solver functions that solve Ax = b in one default function call.
//		Note: these solvers reconstruct the precondioner each call. The solvers of (2) may be more applicable.
//			ez::sor_pGMRES					--  SOR pGMRES with pre-packaged parameters for easy use. Not restarted.
//			ez::sor_pGMRES_robust   --  Lightly dampled SOR pGMRES. Not restarted.	
//			ez::sor_pGMRES_dr				--  SOR pGMRES. With deflated restart.
//			ez::ilu0_pGMRES					--  Incomplete LU (0) pGMRES. Not restarted
//			ez::ilu0_pGMRES_dr			--  Incomplete LU (0) pGMRES. With deflated restart.
//			ez::sor_pCG							--	Symmetric SOR pCG with pre-packaged parameters
// 			ez::iChol0_pCG					--  Incomplete Cholesky (0) pCG.
//		
//SOR, ILU0, and IChol0 preconditioners can be found in blocksp_preconditioners.hpp.
//deflated_restart for pGMRES can be found in blocksp_deflated_restart.hpp.

namespace BlockSp
{
	//////////////////////////
	//Iterative solvers
	//////////////////////////

	///Preconditioned block conjugate gradient (pCG) engine.
	///Solves Ax = b for symmetric positive definite systems, 
	/// which are the user's responsibility to provide. \n
	///tol = user defined tolerance for convergence (multiplied by initial residual),
	///maxIters = maximum iterations, printIters prints pCG iteration count. \n
	///preCond the preconditioner, a void function that inputs/modifies a solArray. \n
	///For T = floats and doubles only. 
	///See Y. Saad "Iterative Methods for Sparse Linear Systems" (chapter 9.2) for details. \n
	template<typename T, int B, int D, typename PreCond>
	solArray<T, B, D>	pCG(const BlockSp<T, B* D>& A, const solArray<T, B, D>& b,
		T tol, const PreCond& preCond, int maxIters = 1000, bool printIters = false)
	{
		static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value); //reals only

		//initial variables
		int len = b.len();
		solArray<T, B, D> x{ len };					  //solution
		solArray<T, B, D> r{ b };							//residual
		solArray<T, B, D> p{ r }; preCond(p);	//search direction
		solArray<T, B, D> z{ p };							//preconditioned residual
		std::vector<T> res_mag;								//magnitude of the residual
		res_mag.push_back(std::sqrt(dot<T, B, D>(p, p)));
		T tolerance = res_mag.back() * tol;
		//std::cout << res_mag.back() << ' ' << tolerance << std::endl;

		int iter = 0;
		while (res_mag.back() > tolerance)
		{
			//Compute Ap
			solArray<T, B, D> Ap{ len };
			bsmv<T, B, D>(Ap, A, p);
			T rz = dot<T, B, D>(r, z);
			T alpha = rz / dot<T, B, D>(p, Ap);	//step length

			//pCG algoritm for each component
			for (int ie = 0; ie < len; ++ie)
				for (int dim = 0; dim < D; ++dim)
				{
					x(ie, dim) += alpha * p(ie, dim);			//approx solution
					r(ie, dim) -= alpha * Ap(ie, dim);		//residual
				}
			res_mag.push_back(std::sqrt(dot<T, B, D>(r, r)));
			//std::cout << iter << ' ' << res_mag.back() << std::endl;

			//update if tolerance not reached
			if (res_mag.back() > tolerance)
			{
				z = r; preCond(z);											//precond resid
				T beta = dot<T, B, D>(r, z) / rz;
				for (int ie = 0; ie < len; ++ie)
					for (int dim = 0; dim < D; ++dim)
						p(ie, dim) = z(ie, dim) + beta * p(ie, dim); // update search direction
				++iter;
				if (iter > maxIters) break;
			}
		}
		if (printIters)
		{
			std::cout << "pCG iters " << iter << std::endl;
			if (iter > maxIters) std::cout << "maximum pCG iterations reached" << std::endl;
		}
		return x;
	}

	///One iteration of the Arnodi orthogonalization algorithm, for pGMRES.
	///Add to/modify upper Hessenburg matrix H and orthonormal basis Q
	template<typename T, int B, int D, typename PreCond>
	void one_arnoldi(int iter, blitz::Array<T, 2>& H, std::vector<solArray<T, B, D>>& Qn,
		const BlockSp<T, B* D>& A, const PreCond* preCond, ScalarType_t<T> zero_thresh)
	{
		//resize matrix H
		int m = iter + 2;
		int n = iter + 1;
		H.resizeAndPreserve(m, n);
		//remove nonsense on bottom row added by blitz's resize/preserve
		for (int i = 0; i < n; ++i)
			H(m - 1, i) = 0.0;

		//compute v = preCond A q_n
		solArray<T, B, D> v{ A.m() };
		bsmv<T, B, D>(v, A, Qn[iter]);
		if (preCond) preCond->operator()(v);

		//calculate H and v
		for (int j = 0; j <= iter; ++j)
		{
			H(j, iter) = dot<T, B, D>(Qn[j], v);
			axpy(v, -H(j, iter), Qn[j]);
		}
		H(iter + 1, iter) = twoNorm<T, B, D>(v, v);
		if (std::abs(H(iter + 1, iter)) < zero_thresh) return; //break if v is zero vector.
		for (auto& block : v)	block /= H(iter + 1, iter);			 //else normalize v, add to Qn.
		Qn.push_back(v);
	}

	///Left preconditioned block GMRES engine.
	///Solves Ax = b for indefinite nonsymmetric linear systems. \n
	///Given arnoldi basis Qn, find y to minimize the residual vector. 
	///Minimum found via QR factorization; then x = Qn y. \n
	///To avoid an intractably large Krylov subspace,
	/// pGMRES can be restarted via deflated_restart, see blocksp_deflated_restart.hpp. 
	/// 
	///tol = user defined tolerance for convergence (multiplied by initial residual),
	///preCond is the preconditioner, a void function that inputs/modifies a solArray. \n
	///maxIters = maximum number of pGMRES iterations.
	///printIters prints the pGMRES iteration count.
	///restart = the number of iterations before pGMRES is restarted via deflated_restart. 
	///					 pGMRES not restarted if restart < 0.
	///k = size of deflated restart space, typically k ~ 5-10.
	///printEigInfo gathers the preconditioned system's Ritz value info from Arnoldi. \n
	/// 
	///For T = float, double, std::complex<float>, and std::complex<double>. \n
	///See Y. Saad "Iterative Methods for Sparse Linear Systems" (see chapter 9.3) for more details.
	template<typename T, int B, int D, typename PreCond>
	solArray<T, B, D> pGMRES(const BlockSp<T, B* D>& A, const solArray<T, B, D>& b,
		ScalarType_t<T> tol, const PreCond& preCond, int maxIters = 500, int restart = -1,
		int k = -1, bool printIters = false, bool printEigInfo = false)
	{
		//initial variables
		using S = ScalarType_t<T>;
		std::vector<S> res_mag;										//magnitude of the residual
		std::vector<solArray<T, B, D>> Qn;				//orthonormal basis from Arnoldi
		blitz::Array<T, 2> H; H = 0.0;  					//upper Hessenburg from Arnoldi
		solArray<T, B, D> x_0{ b.len() };					//initial guess x_0 = 0
		solArray<T, B, D> r_0{ b }; preCond(r_0); //initial preConditioned residual
		res_mag.push_back(twoNorm<T, B, D>(r_0, r_0));
		{
			//Normalize r_0 and add it to Qn
			auto q_0{ r_0 };
			for (auto& block : q_0) block /= res_mag.back();
			Qn.push_back(q_0);
		}
		S zero_thresh = std::min(res_mag.back() * std::numeric_limits<S>::epsilon(),
			std::numeric_limits<S>::epsilon()); //threshold for zero. 
		S tolerance = res_mag.back() * tol;		//tolerance for pGMRES convergence.
		int iter = 0; int totIters = 0;
		std::vector<T> y; blitz::Array<T, 1> c_r; //small solution and residual coefficients
		bool restarted = false;	//has pGMRES been restarted? 

		//begin pGMRES
		while (res_mag.back() > tolerance)
		{
			//perform one arnoldi step
			one_arnoldi(iter, H, Qn, A, &preCond, zero_thresh);

			//Calculate residual vector and its magnitude.
			//Note: Restarting changes residual vector calcuation.
			int m = iter + 2;	int n = iter + 1;
			blitz::Array<T, 1> res_vec(m); res_vec = 0.0;
			if (!restarted) //loops before first restart. 
			{
				//find y via QR factorization to min|| ||b||_2 e1 - H y ||_2 
				//i.e. the magnitude of the non-restarted residual vectors.
				auto HCopy(H.copy());
				y.clear(); y.resize(m); y[0] = res_mag[0];
				lapack::gels<T>(LAPACK_ROW_MAJOR, 'N', m, n, 1, HCopy.data(), n, y.data(), 1);
				y.pop_back(); //remove last element(left over after gels)

				//calculate residual vector ||b||_2 e1 - H y and its magnitude.
				lapack::gemv<T>(CblasRowMajor, CblasNoTrans, m, n, -1.0, H.data(), n, y.data(), 1, 0, res_vec.data(), 1);
				res_vec(0) += res_mag[0];
				S r = 0.0;
				for (const auto& val : res_vec) r += is_complex_v<T> ? std::abs(std::conj(val) * val) : std::abs(val * val);
				res_mag.push_back(std::sqrt(r));
				//std::cout << totIters << ' ' << res_mag.back() << std::endl;
			}
			else //loops after restart
			{
				//find y via QR factorization min|| c_r - H y ||_2 = res
				//c_r = Q_{n+1}^T r_0, is a small residual coefficient vector, 
				//	where r_0 is the initial residual. 
				//Q_{n+1}^T is orthogonal to r_0 for each q after the kth.
				//c_r is calculated in the deflated_restart function.
				auto HCopy(H.copy());
				y.clear(); y.resize(m);
				int c_len = c_r.extent(0);
				for (int i = 0; i < c_len; ++i) y[i] = c_r(i);
				for (int i = c_len; i < m; ++i)	y[i] = T(0.0);
				std::vector<T> c{ y };
				lapack::gels<T>(LAPACK_ROW_MAJOR, 'N', m, n, 1, HCopy.data(), n, y.data(), 1);
				y.pop_back(); //remove last element(left over after gels)

				//calculate residual vector || c_r - H y ||_2 = res
				lapack::gemv<T>(CblasRowMajor, CblasNoTrans, m, n, -1.0, H.data(), n, y.data(), 1, 0, res_vec.data(), 1);
				for (int i = 0; i < m; ++i) res_vec(i) += c[i];
				S r = 0.0;
				for (const auto& val : res_vec) r += is_complex_v<T> ? std::abs(std::conj(val) * val) : std::abs(val * val);
				res_mag.push_back(std::sqrt(r));
				//std::cout << totIters << ' ' << res_mag.back() << std::endl;
			}
			++iter; ++totIters;
			if (totIters > maxIters) break;

			///pGMRES often requires a restart to avoid an intractably large Krylov subspace.
			///Using the deflated restart of Morgan. See blocksp_deflated_restart.hpp
			///Will construct a new H and Qn via the k-th smallest harmonic Ritz vectors.
			if (iter == restart && res_mag.back() > tolerance)
			{
				//compute approximate solution x = x_0 + Qy
				solArray<T, B, D> x{ x_0 };
				for (int iy = 0; iy < y.size(); ++iy)
					axpy(x, y[iy], Qn[iy]);
				x_0 = x; //set new initial solution x_0 = x

				//find new initial precondtioned residual r_0 
				r_0 = b - bsmv(A, x_0); preCond(r_0);

				//deflated restart -- create harmonic Ritz space of size k,
				//then restart pGMRES with this new space.
				//This function modifies all input variables except k.
				deflated_restart<T, B, D>(H, Qn, res_vec, c_r, res_mag, k);
				iter = k == 0 ? 0 : H.extent(1);
				if (k == 0) //if pure restart
				{
					auto q_0{ r_0 };
					res_mag.push_back(twoNorm<T, B, D>(q_0, q_0));
					c_r(0) = res_mag.back();
					for (auto& block : q_0) block /= res_mag.back();
					Qn.push_back(q_0);
				}
				restarted = k == 0 ? false : true; //change residual calculation
			}
		}
		if (printIters)
		{
			std::cout << "pGMRES iters " << totIters << std::endl;
			if (totIters > maxIters) std::cout << "maximum pGMRES iterations reached" << std::endl;
		}

		//Compute eigenvalues of H_s = H with last row removed (square portion).
		//Normal Ritz values approximate extreme eigenvalues of precond(A).
		//If restarted, print harmonic Ritz info, which approximate small eigenvalues.
		if (printEigInfo)
		{
			auto H_s(H.copy()); H_s.resizeAndPreserve(H.extent(0) - 1, H.extent(1));
			if (restarted)
			{
				dense::Eigenvalues_Arr<T> HEig(H_s, true, H(H.extent(0) - 1, H.extent(1) - 1));
				HEig.printEigInfo();
			}
			else
			{
				dense::Eigenvalues_Arr<T> HEig(H_s);
				HEig.printEigInfo();
			}
		}

		//compute x = Qn y
		assert(y.size() == Qn.size() - 1);
		solArray<T, B, D> x(x_0);
		for (int iy = 0; iy < y.size(); ++iy) x += y[iy] * Qn[iy];
		return x;
	}

	///Arnoldi eigenvalue solver.
	///Returns "iters" number of Ritz values, i.e. approximate eigenvalues. 
	///Regular Ritz values tend to approximate extreme eigenvalues.
	///If harmonic_Ritz, harmonic Ritz values are calculated instead of regular Ritz values.
	///Harmonic Ritz values tend to approximate small eigenvalues.
	///May return fewer values if the minimal polynomial of A is lower degree than "iters".
	///printEigInfo provides the min, max, smallest, and largest (by magnitude)
	/// eigenvalues and the matrix condition number.
	///printEigVals prints all eigenvalues.
	template<typename T, int M, typename PreCond>
	dense::Eigenvalues_Arr<T> arnoldi_eigenvalues(const BlockSp<T, M>& A, int iters = 30,
		const PreCond* preCond = nullptr, bool printEigInfo = false, bool printEigVals = false,
		bool harmonic_Ritz = false)
	{
		assert(A.m() == A.n());
		//initial variables
		std::vector<solArray<T, M>> Qn;			  	//orthonormal basis for Arnoldi
		blitz::Array<T, 2> H;							  		//upper Hessenburg for Arnoldi
		H = 0.0;
		{
			solArray<T, M> q_0{ A.n() }; q_0(0, 0)(0) = T(1.0); //make first normal vector
			Qn.push_back(q_0);
		}
		ScalarType_t<T> zero_thresh = 10 * M * M * std::numeric_limits<T>::epsilon(); //threshold for zero. 
		iters = std::min(iters, A.m() * M);

		//begin arnoldi
		for (int iter = 0; iter < iters; ++iter)
			one_arnoldi(iter, H, Qn, A, preCond, zero_thresh);

		//compute eigenvalues of H_s = H with last row removed (square portion)
		//Ritz values approximate extreme eigenvalues of A
		auto H_s(H.copy()); H_s.resizeAndPreserve(H.extent(0) - 1, H.extent(1));
		dense::Eigenvalues_Arr<T> HEig(H_s, harmonic_Ritz, H(H.extent(0) - 1, H.extent(1) - 1));
		if (printEigInfo) HEig.printEigInfo();
		if (printEigVals) HEig.printEigVals();
		return HEig;
	}


	/////////////////////////////
	//Pre-packaged solvers
	/////////////////////////////

	//See ez solves at the bottom!

	///pGMRES with block SOR preconditioner/LU block inversion.
	template<typename T, int B, int D = 1>
	struct SOR_pGMRES
	{
		preConditioners::SOR<dense::LUFactor<T, B* D>, T, B, D> SOR_pre; ///< SOR preconditioner
		ScalarType_t<T> m_w;				///< SOR damping parameter.
		int m_nu;										///< Default number of SOR iterations.
		int m_SORtype;							///< SOR types: <0 backward only; >0 forward only, 0 symmetric.
		ScalarType_t<T> m_tol;			///< Tolerance for pGMRES convergence.
		int maxIters;								///< Maximum number of pGMRES iterations. 
		int restart;								///< Number of iterations before deflated restart of pGMRES. not restarted if < 0.
		int m_k;										///< Size of deflated restart.
		bool printIters;						///< Print pGMRES iteration count.
		bool printEigInfo;					///< Print eigenvalue info from arnoldi.

		///pGMRES with block SOR preconditioner/LU block inversion.
		///SOR damping parameter of 1.6, 3 iterations per direction (6 total), symmetric SOR.
		///Note: The SOR damping parameter w may greatly impact solver speed.
		///w = (0, 2), larger parameters are often faster but smaller are more robust.
		///Note: Not restarted by default. 
		SOR_pGMRES(const BlockSp<T, B* D>& A, ScalarType_t<T> w = 1.6, int nu = 3, int SORtype = 0, ScalarType_t<T> tol = 1.0e-8,
			int maxIters = 5000, int restart = -1, int k = -1, bool printIters = false, bool printEigInfo = false) :
			m_w{ w }, m_nu{ nu }, m_SORtype{ SORtype }, m_tol{ tol }, maxIters{ maxIters },
			restart{ restart }, m_k{ k }, printIters{ printIters }, printEigInfo{ printEigInfo }
		{
			SOR_pre = preConditioners::SOR<dense::LUFactor<T, B* D>, T, B, D>(A, m_w, m_nu, m_SORtype);
		}

		///Solve Ax = b, for x
		solArray<T, B, D> solve(const BlockSp<T, B* D>& A, const solArray<T, B, D>& b) const
		{
			return pGMRES<T, B, D>(A, b, m_tol, SOR_pre, maxIters, restart, m_k, printIters, printEigInfo);
		}

		inline preConditioners::SOR<dense::LUFactor<T, B* D>, T, B, D>& access_SOR_pre() { return SOR_pre; }
		inline void change_SOR_pre(const preConditioners::SOR<dense::LUFactor<T, B* D>, T, B, D>& SOR_new)
		{
			SOR_pre = SOR_new;
		}
		inline void change_w(ScalarType_t<T> w) { m_w = w; }
		inline void change_nu(int nu) { m_nu = nu; }
		inline void change_SORtype(int type) { m_SORtype = type; }
		inline void change_tol(ScalarType_t<T> tol) { m_tol = tol; }
		inline void change_maxIters(int iters) { maxIters = iters; }
		inline void change_restart(int re) { restart = re; }
		inline void change_k(int k) { m_k = k; }
		inline void change_printIters(bool pr) { printIters = pr; }
		inline void change_printEigInfo(bool pei) { printEigInfo = pei; }
	};

	///pCG with block SOR preconditioner/Cholesky block inversion.
	///For symmetric positive definite (SPD) linear systems. 
	template<typename T, int B, int D = 1>
	struct SOR_pCG
	{
		preConditioners::SOR<dense::CholeskyFactor<T, B* D>, T, B, D> SOR_pre; ///< SOR preconditioner
		ScalarType_t<T> m_w;				///< SOR damping parameter		
		int m_nu;										///< Default number of SOR iterations
		ScalarType_t<T> m_tol;			///< Tolerance for pCG convergence.
		int maxIters;								///< Maximum number of pCG iterations. 
		int printIters;							///< Print pCG iteration count.

		///SPD Solver - pCG with SOR preconditioner / Cholesky block inversion.
		///Defaults - SOR damping parameter of 1.6, 3 iterations per direction (6 total), symmetric SOR required.
		SOR_pCG(const BlockSp<T, B* D>& A, ScalarType_t<T> w = 1.6, int nu = 3, ScalarType_t<T> tol = 1.0e-8,
			int maxIters = 5000, bool printIters = false) : m_w{ w }, m_nu{ nu },
			m_tol{ tol }, maxIters{ maxIters }, printIters{ printIters }
		{
			static_assert(!is_complex_v<T>); // reals only
			SOR_pre = preConditioners::SOR<dense::CholeskyFactor<T, B* D>, T, B, D>(A, m_w, m_nu);
		}

		///Solve Ax = b, for x
		solArray<T, B, D>	solve(const BlockSp<T, B* D>& A, const solArray<T, B, D>& b)
		{
			return pCG<T, B, D>(A, b, m_tol, SOR_pre, maxIters, printIters);
		}

		inline preConditioners::SOR<dense::CholeskyFactor<T, B* D>, T, B, D>& access_SOR_pre() { return SOR_pre; }
		inline void change_SOR_pre(const preConditioners::SOR<dense::CholeskyFactor<T, B* D>, T, B, D>& SOR_new)
		{
			SOR_pre = SOR_new;
		}
		inline void change_w(ScalarType_t<T> w) { m_w = w; }
		inline void change_nu(int nu) { m_nu = nu; }
		inline void change_tol(ScalarType_t<T> tol) { m_tol = tol; }
		inline void change_maxIters(int iters) { maxIters = iters; }
		inline void change_printIters(bool pr) { printIters = pr; }
	};

	///pGMRES with block incomplete LU with zero fill preconditioner.
	template<typename T, int B, int D = 1>
	struct ILU0_pGMRES
	{
		preConditioners::ILU0<T, B, D> ILU0_pre; ///< ILU0 preconditioner
		ScalarType_t<T> m_tol;			///< Tolerance for pGMRES convergence.
		int maxIters;								///< Maximum number of pGMRES iterations. 
		int restart;								///< Number of iterations before deflated restart of pGMRES. not restarted if < 0.
		int m_k;										///< Size of deflated restart.
		bool printIters;						///< Print pGMRES iteration count.
		bool printEigInfo;					///< Print eigenvalue info from arnoldi.

		///pGMRES with ILU0 preconditioner.
		///Note: Not restarted by default. 
		ILU0_pGMRES(const BlockSp<T, B* D>& A, ScalarType_t<T> tol = 1.0e-8, int maxIters = 5000,
			int restart = -1, int k = -1, bool printIters = false, bool printEigInfo = false) :
			m_tol{ tol }, maxIters{ maxIters }, restart{ restart }, m_k{ k },
			printIters{ printIters }, printEigInfo{ printEigInfo },
			ILU0_pre{ preConditioners::ILU0<T, B, D>(A) } {
		}

		///Solve Ax = b, for x
		solArray<T, B, D> solve(const BlockSp<T, B* D>& A, const solArray<T, B, D>& b) const
		{
			return pGMRES<T, B, D>(A, b, m_tol, ILU0_pre, maxIters, restart, m_k, printIters, printEigInfo);
		}

		inline preConditioners::ILU0<T, B, D>& access_ILU0_pre() { return ILU0_pre; }
		inline void change_tol(ScalarType_t<T> tol) { m_tol = tol; }
		inline void change_maxIters(int iters) { maxIters = iters; }
		inline void change_restart(int re) { restart = re; }
		inline void change_k(int k) { m_k = k; }
		inline void change_printIters(bool pr) { printIters = pr; }
		inline void change_printEigInfo(bool pei) { printEigInfo = pei; }
	};

	///pCG with block incomplete Cholesky with zero fill preconditioner.
	///For symmetric positive definite (SPD) linear systems. 
	template<typename T, int B, int D = 1>
	struct IChol0_pCG
	{
		preConditioners::IChol0<T, B, D> IChol0_pre; ///< IChol0 preconditioner
		ScalarType_t<T> m_tol;			///< Tolerance for pCG convergence.
		int maxIters;								///< Maximum number of pCG iterations. 
		int printIters;							///< Print pCG iteration count.

		///pCG with IChol0 preconditioner.
		///Note: not restarted by default.
		IChol0_pCG(const BlockSp<T, B* D>& A, ScalarType_t<T> tol = 1.0e-8,
			int maxIters = 5000, bool printIters = false) : m_tol{ tol }, maxIters{ maxIters },
			printIters{ printIters }, IChol0_pre{ preConditioners::IChol0<T, B, D>(A) }
		{
			static_assert(!is_complex_v<T>); // reals only
		}

		///Solve Ax = b, for x
		solArray<T, B, D>	solve(const BlockSp<T, B* D>& A, const solArray<T, B, D>& b)
		{
			return pCG<T, B, D>(A, b, m_tol, IChol0_pre, maxIters, printIters);
		}

		inline preConditioners::IChol0<T, B, D>& access_IChol0_pre() { return IChol0_pre; }
		inline void change_tol(ScalarType_t<T> tol) { m_tol = tol; }
		inline void change_maxIters(int iters) { maxIters = iters; }
		inline void change_printIters(bool pr) { printIters = pr; }
	};


	//////////////////////
	//ez solves!
	//////////////////////

	///"ez" solver functions solve Ax = b in one default function call.
	///Note: these solvers reconstruct the precondioner each call.
	///Other solvers may be more applicable.
	namespace ez
	{
		///////////
		//pGMRES

		///ez::sor_pGMRES! pGMRES with SOR preconditioner/LU block inversion,
		/// with pre-packaged parameters for easy use.
		///The pGMRES in this function does not restart.
		///Call: ez::sor_pGMRES(A, b) to solve Ax = b. Returns x. 
		template<typename T, int B, int D = 1>
		solArray<T, B, D> sor_pGMRES(const BlockSp<T, B* D>& A, const solArray<T, B, D>& b,
			bool printIters = false, bool printEigInfo = false)
		{
			ScalarType_t<T> w = 1.6;		//SOR damping parameter. 1.6 is relatively high for speed.
			int nu = 3;									//Default number of SOR iterations (2x total for symmetric).
			int SORtype = 0;						//SOR types: <0 backward only; >0 forward only, 0 symmetric.
			ScalarType_t<T> tol;				//Tolerance for pGMRES convergence.
			int maxIters = 5000;				//Maximum number of pGMRES iterations. 
			int restart = -1;						//Number of iterations before deflated restart of pGMRES.
			//negative = no restart.
			int k = -1;									//Size of deflated restart.
			tol = (std::is_same<T, float>::value || std::is_same<T, std::complex<float>>::value) ?
				1.0e-5 : 1.0e-8;
			SOR_pGMRES<T, B, D> solver(A, w, nu, SORtype, tol, maxIters, restart, k, printIters, printEigInfo);
			return solver.solve(A, b);
		};

		///ez::sor_pGMRES_robust! pGMRES with SOR preconditioner/LU block inversion.
		///Same as sor_pGMRES but with a smaller SOR damping parameter w = 0.6 for stability.
		///The pGMRES in this function does not restart. 
		///Call: ez::sor_pGMRES_robust(A, b) to solve Ax = b. Returns x. 
		template<typename T, int B, int D = 1>
		solArray<T, B, D> sor_pGMRES_robust(const BlockSp<T, B* D>& A, const solArray<T, B, D>& b,
			bool printIters = false, bool printEigInfo = false)
		{
			ScalarType_t<T> w = 0.6;		//SOR damping parameter. 0.6 is relatively low
			int nu = 3;									//Default number of SOR iterations (2x total for symmetric).
			int SORtype = 0;						//SOR types: <0 backward only; >0 forward only, 0 symmetric.
			ScalarType_t<T> tol;				//Tolerance for pGMRES convergence.
			int maxIters = 5000;				//Maximum number of pGMRES iterations. 
			int restart = -1;						//Number of iterations before deflated restart of pGMRES.
			//negative = no restart.
			int k = -1;									//Size of deflated restart.
			tol = (std::is_same<T, float>::value || std::is_same<T, std::complex<float>>::value) ?
				1.0e-5 : 1.0e-8;
			SOR_pGMRES<T, B, D> solver(A, w, nu, SORtype, tol, maxIters, restart, k, printIters, printEigInfo);
			return solver.solve(A, b);
		};

		///ez::sor_pGMRES_dr! pGMRES with SOR preconditioner/LU block inversion and deflated restart.
		///The pGMRES in this function is restarted via deflated_restart. 
		///The maximum Krylov subspace size is 30 and the deflated restart space size is 10. 
		///Call: ez::sor_pGMRES_dr(A, b) to solve Ax = b. Returns x. 
		template<typename T, int B, int D = 1>
		solArray<T, B, D> sor_pGMRES_dr(const BlockSp<T, B* D>& A, const solArray<T, B, D>& b,
			bool printIters = false, bool printEigInfo = false)
		{
			ScalarType_t<T> w = 1.6;		//SOR damping parameter.
			int nu = 3;									//Default number of SOR iterations (2x total for symmetric).
			int SORtype = 0;						//SOR types: <0 backward only; >0 forward only, 0 symmetric.
			ScalarType_t<T> tol;				//Tolerance for pGMRES convergence.
			int maxIters = 5000;				//Maximum number of pGMRES iterations. 
			int restart = 30;						//Number of iterations before deflated restart of pGMRES.
			int k = 10;									//Size of deflated restart.
			tol = (std::is_same<T, float>::value || std::is_same<T, std::complex<float>>::value) ?
				1.0e-5 : 1.0e-8;
			SOR_pGMRES<T, B, D> solver(A, w, nu, SORtype, tol, maxIters, restart, k, printIters, printEigInfo);
			return solver.solve(A, b);
		};

		///ez::ilu0_pGMRES! pGMRES with incomplete LU (0) preconditioner.
		///The pGMRES in this function does not restart. 
		///Call: ez::ilu0_pGMRES(A, b) to solve Ax = b. Returns x. 
		template<typename T, int B, int D = 1>
		solArray<T, B, D> ilu0_pGMRES(const BlockSp<T, B* D>& A, const solArray<T, B, D>& b,
			bool printIters = false, bool printEigInfo = false)
		{
			int maxIters = 5000;				//Maximum number of pGMRES iterations. 
			int restart = -1;						//Number of iterations before deflated restart of pGMRES.
			//negative = no restart.
			int k = -1;									//Size of deflated restart.
			ScalarType_t<T> tol;				//Tolerance for pGMRES convergence.
			tol = (std::is_same<T, float>::value || std::is_same<T, std::complex<float>>::value) ?
				1.0e-5 : 1.0e-8;
			ILU0_pGMRES<T, B, D> solver(A, tol, maxIters, restart, k, printIters, printEigInfo);
			return solver.solve(A, b);
		};

		///ez::ilu0_pGMRES_dr! pGMRES with incomplete LU (0) preconditioner.
		///The pGMRES in this function is restarted via deflated_restart. 
		///The maximum Krylov subspace size is 30 and the deflated restart space size is 10. 
		///Call: ez::ilu0_pGMRES_dr(A, b) to solve Ax = b. Returns x. 
		template<typename T, int B, int D = 1>
		solArray<T, B, D> ilu0_pGMRES_dr(const BlockSp<T, B* D>& A, const solArray<T, B, D>& b,
			bool printIters = false, bool printEigInfo = false)
		{
			int maxIters = 5000;				//Maximum number of pGMRES iterations. 
			int restart = 30;						//Number of iterations before deflated restart of pGMRES.
			int k = 10;									//Size of deflated restart.
			ScalarType_t<T> tol;				//Tolerance for pGMRES convergence.
			tol = (std::is_same<T, float>::value || std::is_same<T, std::complex<float>>::value) ?
				1.0e-5 : 1.0e-8;
			ILU0_pGMRES<T, B, D> solver(A, tol, maxIters, restart, k, printIters, printEigInfo);
			return solver.solve(A, b);
		};

		///////////
		//pCG

		///ez::sor_pCG! Symmetric SOR pCG with Cholesky block inversion.
		///Call ez::sor_pCG(A, b) to solve Ax = b for symmetric positive definite systems (SPD).
		/// Returns x.
		template<typename T, int B, int D = 1>
		solArray<T, B, D> sor_pCG(const BlockSp<T, B* D>& A, const solArray<T, B, D>& b,
			bool printIters = false)
		{
			ScalarType_t<T> w = 1.6;		//SOR damping parameter. 1.6 is relatively high for speed.
			int nu = 3;									//Default number of SOR iterations (2x total for symmetric).
			ScalarType_t<T> tol;				//Tolerance for pGMRES convergence.
			int maxIters = 5000;				//Maximum number of pGMRES iterations. 
			tol = (std::is_same<T, float>::value || std::is_same<T, std::complex<float>>::value) ?
				1.0e-5 : 1.0e-8;
			SOR_pCG<T, B, D> solver(A, w, nu, tol, maxIters, printIters);
			return solver.solve(A, b);
		}

		///ez::ichol0_pCG! Incomplete Cholesky (0) pCG.
		///Call ez::ichol0_pCG(A, b) to solve Ax = b for symmetric positive definite systems (SPD).
		/// Returns x.
		template<typename T, int B, int D = 1>
		solArray<T, B, D> ichol0_pCG(const BlockSp<T, B* D>& A, const solArray<T, B, D>& b,
			bool printIters = false)
		{
			int maxIters = 5000;				//Maximum number of pGMRES iterations. 
			ScalarType_t<T> tol;				//Tolerance for pGMRES convergence.
			tol = (std::is_same<T, float>::value || std::is_same<T, std::complex<float>>::value) ?
				1.0e-5 : 1.0e-8;
			IChol0_pCG<T, B, D> solver(A, tol, maxIters, printIters);
			return solver.solve(A, b);
		}

	} // end namepace BlockSp::ez

} //end namespace BlockSp

#endif // end BLOCKSP_SOLVERS_HPP