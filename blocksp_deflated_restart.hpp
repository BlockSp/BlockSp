#ifndef BLOCKSP_DEFLATED_RESTART_HPP
#define BLOCKSP_DEFLATED_RESTART_HPP

#include "blocksp_containers.hpp"
#include "blocksp_dense_multiply.hpp"
#include "blocksp_dense_solvers.hpp"
#include "blocksp_preconditioners.hpp"
#include "blocksp_utility.hpp"

namespace BlockSp
{

	///Restart pGMRES with a size k+1 Krylov subspace that consists of k harmonic Ritz vectors.
	///Harmonic Ritz vectors are picked to approximate the smallest eigenvalues/vectors.
	///These are the eigenvalues that give pGMRES the most difficulty.
	///Having these vectors in the restart space "deflates" the space pGMRES must search,
	///leading to better performance. 
	///Non-restarted pGMRES may become intractably large, pure restarts may stall.
	///See "GMRES with Deflated Restarting" by Ronald B. Morgan for more details.
	/// 
	///Note: this implementation of deflated restart pGMRES may still have issues. 
	///For some problems, the pGMRES iteration count greatly increases and is slower than non-restarted.
	///For others, the increase is minimal and the deflated restart version is faster than non-restarted.
	///There are likely some issues with numerical stability.
	///Try pGMRES with and without restarts, see which works best for your problem.
	///BlockSp offers SOR and ILU0 pGMRES with and without restarting.
	/// 
	///There are a few commented out debugging protocols at the bottom of this file
	///If you find any bugs, please let me know :)
	/// 
	///This program inputs the (n+1) by n upper Hessenberg Arnoldi matrix H, 
	///the orthonormal Krylov basis Qn, the residual vector res_vec, 
	///the residual coefficient vector c_r, the residual magnitudes res_mag,
	///and the restart space size k. Each is modified during the restart.
	///Works for both reals and complex.
	/// 
	///AI disclaimer: deflated_restart was written with assistance from Anthropic's Claude Opus 4.5.  
	template<typename T, int B, int D>
	void deflated_restart(blitz::Array<T, 2>& H, std::vector<solArray<T, B, D>>& Qn,
		blitz::Array<T, 1>& res_vec, blitz::Array<T, 1>& c_r,
		std::vector<ScalarType_t<T>>& res_mag, int k)
	{
		using S = ScalarType_t<T>;
		int m = H.extent(0);      //number of rows of H, m = n + 1, 
		int n = H.extent(1);      //number of columns of H
		if (k > n) k = n;					//limit k to available eigenvalues
		S zero_thresh = 10 * n * n * std::numeric_limits<S>::epsilon();

		if (k == 0)  //pure restart
		{
			H.resize(0);
			Qn.clear();
			res_mag.clear();
			c_r.resize(1);
			return;
		}

		//////////////////////////////////////////////////////
		///(1) Compute H_tilde = H_n + |h_mn|^2 * H_n^{-T} * e_n * e_n^T
		///The eigenvalues of H_tilde are the harmonic Ritz values.
		///H_n is the nxn square portion of H.
		///h_mn the bottom right sub-diagonal element of H. 
		///H_n^{-T} is the inverse of the transpose of H_n.
		///e_n is the n-th standard basis vector.

		//Extract square portion H_n (n x n)
		blitz::Array<T, 2> H_n(n);
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < n; ++j)
				H_n(i, j) = H(i, j);

		//Get h_mn, the bottom right sub-diagonal element of H
		T h_mn = H(n, n - 1);

		//Solve H_n^T * w = e_n for w (compute H_n^{-T} * e_n), via LU factorization.
		blitz::Array<T, 1> w(n); w = T(0);
		w(n - 1) = T(1);  // e_n
		{
			dense::LUFactor_Arr<T> H_nT_LU(dense::transpose(H_n));
			H_nT_LU.invmul(w);
		}

		//Form H_tilde (stored in H_n) = H_n + |h_mn|^2 * w * e_n^T
		//The second rhs term only modifies the last column of H_n
		T scale; 
		if constexpr (is_complex_v<T>) scale = std::conj(h_mn) * h_mn;
		else scale = h_mn * h_mn;
		for (int i = 0; i < n; ++i)
			H_n(i, n - 1) += scale * w(i);

		//////////////////////////////////////////////////////
		///(2) Compute the Schur decomposition of H_tilde
		///Then reorder to put the k smallest eigenvalues first

		blitz::Array<T, 2> Z(n, n);  //Schur vectors
		lapack_int sdim = 0;
		if constexpr (is_complex_v<T>) //Complex case
		{
			//Compute for eigenvalues and Schur factorization
			blitz::Array<T, 1> eig_vals(n);
			lapack::gees<T>(LAPACK_ROW_MAJOR, 'V', 'N', nullptr, n,
				H_n.data(), n, &sdim, eig_vals.data(), Z.data(), n);

			//Create a permutation function to find smallest Ritz values
			auto compare = [&](const auto& a, const auto& b) { return std::abs(a) < std::abs(b); };
			auto perm{ util::sort_permutation(eig_vals, compare) };

			//Build select array for k smallest
			std::vector<lapack_logical> select(n, 0);
			for (int i = 0; i < k; ++i)
				select[perm[i]] = 1;

			//Reorder Schur form, sort eigenvalues by magnitude to find k smallest
			S s_out, sep_out;
			lapack::trsen<T>(LAPACK_ROW_MAJOR, 'V', 'V', select.data(), n,
				H_n.data(), n, Z.data(), n, eig_vals.data(), &sdim, &s_out, &sep_out);
		}
		else //Real case
		{
			blitz::Array<T, 1> wr(n), wi(n);  //real and imaginary parts of eigenvalues
			lapack::gees<T>(LAPACK_ROW_MAJOR, 'V', 'N', nullptr, n,
				H_n.data(), n, &sdim, wr.data(), wi.data(), Z.data(), n);

			//Sort eigenvalues by magnitude to find k smallest
			std::vector<int> perm(n);
			std::iota(perm.begin(), perm.end(), 0);
			std::sort(perm.begin(), perm.end(), [&](int i, int j) {
				S mag_1 = std::sqrt(wr(i) * wr(i) + wi(i) * wi(i));
				S mag_2 = std::sqrt(wr(j) * wr(j) + wi(j) * wi(j));
				return mag_1 < mag_2;
				});

			//Build the select array for k smallest eigenvalues.
			//If selecting a complex eigenvalue that's part of a 2x2 block,
			//both real and imaginary parts of the eigenvalue are selected.
			std::vector<lapack_logical> select(n, 0);
			int cnt = 0;
			for (int i = 0; i < n && cnt < k; ++i)
			{
				int ip = perm[i];
				if (select[ip]) continue;  //already selected as part of a pair
				select[ip] = 1;
				cnt++;

				//If this is a complex eigenvalue (wi != 0), 
				//find and select its conjugate pair, at ip+1 or ip-1.
				if (std::abs(wi(ip)) > zero_thresh)
				{
					//Find the conjugate
					for (int j = 0; j < n; ++j)
					{
						if (j != ip &&
							std::abs(wr(j) - wr(ip)) < zero_thresh &&
							std::abs(wi(j) + wi(ip)) < zero_thresh)
						{
							if (!select[j])
							{
								select[j] = 1;
								cnt++;
							}
							break;
						}
					}
				}
			}

			//Reorder Schur form to put selected eigenvalues first
			S s_out = 0; S sep_out = 0;
			lapack::trsen<T>(LAPACK_ROW_MAJOR, 'V', 'V', select.data(), n,
				H_n.data(), n, Z.data(), n, wr.data(), wi.data(), &sdim, &s_out, &sep_out);
		}

		//Update k, which may have changed in real systems due to complex eigenvalues
		k = sdim;

		//////////////////////////////////////////////////////
		///(3) Extract P_k (first k Schur vectors) and orthonormalize

		//P_k is the first k columns of Z
		blitz::Array<T, 2> P_k(n, k);
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < k; ++j)
				P_k(i, j) = Z(i, j);

		//Orthonormalize P_k via QR factorization
		//The Schur vectors are orthonormal, but this aids numerical stability
		{
			blitz::Array<T, 1> tau(k);
			lapack::geqrf<T>(LAPACK_ROW_MAJOR, n, k, P_k.data(), k, tau.data());
			if constexpr (is_complex_v<T>)
				lapack::ungqr<T>(LAPACK_ROW_MAJOR, n, k, k, P_k.data(), k, tau.data());
			else lapack::orgqr<T>(LAPACK_ROW_MAJOR, n, k, k, P_k.data(), k, tau.data());
		}
		
		//////////////////////////////////////////////////////
		///(4) Form P_{k+1} by extending P_k and orthonormalizing res_vec

		//P_{k+1} is (n+1)x(k+1)
		blitz::Array<T, 2> P_kp1(n + 1, k + 1);
		P_kp1 = T(0);

		//Copy P_k into first k columns (with zero in last row)
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < k; ++j)
				P_kp1(i, j) = P_k(i, j);

		//Put res_vec in last column
		for (int i = 0; i < n + 1; ++i)
			P_kp1(i, k) = res_vec(i);

		//Gram-Schmidt: orthogonalize last column against first k
		{
			for (int i = 0; i < k; ++i)
			{
				T d = T(0);
				for (int j = 0; j < n + 1; ++j)
					if constexpr (is_complex_v<T>) d += std::conj(P_kp1(j, i)) * P_kp1(j, k);
					else d += P_kp1(j, i) * P_kp1(j, k);
				for (int j = 0; j < n + 1; ++j)
					P_kp1(j, k) -= d * P_kp1(j, i);
			}
			//Normalize last column
			T norm = T(0);
			for (int j = 0; j < n + 1; ++j)
				if constexpr (is_complex_v<T>) norm += std::conj(P_kp1(j, k)) * P_kp1(j, k);
				else norm += P_kp1(j, k) * P_kp1(j, k);
			norm = std::sqrt(norm);
			for (int j = 0; j < n + 1; ++j)
				P_kp1(j, k) /= norm;
		}

		//////////////////////////////////////////////////////
		///(5) Compute new residual coefficient vector c_r = P_{k+1}^T * res_vec

		c_r.resize(k + 1);
		c_r = T(0);
		for (int i = 0; i < k + 1; ++i)
			for (int j = 0; j < n + 1; ++j)
				if constexpr(is_complex_v<T>) c_r(i) += std::conj(P_kp1(j, i)) * res_vec(j);
				else c_r(i) += P_kp1(j, i) * res_vec(j);

		//////////////////////////////////////////////////////
		///(6) Form the new H: H_new = P_{k+1}^T * H * P_k

		auto H_new{ dense::matMat_transpose(P_kp1, dense::matMat(H, P_k)) };

		//////////////////////////////////////////////////////
		///(7) Form the new basis: Qn_new = Qn * P_{k+1}

		int len = Qn[0].len();
		std::vector<solArray<T, B, D>> Qn_new(k + 1);
		for (int i = 0; i < k + 1; ++i)
			Qn_new[i].resize(len);

		for (int im = 0; im < m; ++im)
			for (int ik = 0; ik < k + 1; ++ik)
				for (int il = 0; il < len; ++il)
					Qn_new[ik](il) += Qn[im](il) * P_kp1(im, ik);

		//////////////////////////////////////////////////////
		///(8) Set new values

		H.resize(H_new.extent());
		H = H_new;
		Qn = Qn_new;
		std::vector<S> res_mag_new;
		res_mag_new.push_back(res_mag.back());
		res_mag = res_mag_new;
	}


	//deflated_restart pGMRES debugging protocols
	//place these in the pGMRES function or in deflated_restart

	/*
//compare dot(Qn[iy], r_0) and c_r
//place where y is calculated
std::vector<T> yy(m);
for (int iy = 0; iy < m; ++iy) yy[iy] = dot(Qn[iy], r_0);

T dif = 0.0;
for (int i = 0; i < m; ++i)
	dif = std::max(dif, std::abs(y[i] - yy[i]));
std::cout << "y diff " << dif << std::endl;

//if (iter + 1 == restart)
{
	std::cout << std::endl;
	std::cout << m << ' ' << n << ' ' << Qn.size() << std::endl;
	std::cout << "y" << std::endl;
	for (int i = 0; i < m; ++i)
		std::cout << i << ' ' << y[i] << std::endl;
	std::cout << std::endl;
}
//*/

/*
{
	//compare true residual b - Ax_0 with residual vector Qn * res_vec
	//place after res_vec calculation

	//compute approximate solution x = x_0 + Qy
	solArray<T, B, D> x{ x_0 };
	for (int iy = 0; iy < y.size(); ++iy)
		axpy(x, y[iy], Qn[iy]);

	//find new initial precondtioned residual q_0
	auto qq{ b - bsmv(A, x) }; preCond(qq);

	solArray<T, B, D> rr(r_0.len());
	for (int j = 0; j < res_vec.size(); ++j)
		for (int i = 0; i < rr.len(); ++i)
			rr(i) += Qn[j](i) * res_vec(j);
	//std::cout << "diff " << infNorm(qq - rr) << std::endl;
	//std::cout << std::endl;
}
//*/

/*
//After deflated_restart, verify orthogonality of new basis
//place after deflated_restart

std::cout << "Checking Qn orthogonality after restart:" << std::endl;
for (int i = 0; i < Qn.size(); ++i)
{
	for (int j = 0; j <= i; ++j)
	{
		T d = dot(Qn[i], Qn[j]);
		if (i == j)
			std::cout << "||Qn[" << i << "]||^2 = " << d << std::endl;
		else if (std::abs(d) > 1e-10)
			std::cout << "<Qn[" << i << "], Qn[" << j << "]> = " << d << " (should be 0)" << std::endl;
	}
}

// Also check that c_coeff has the right norm
T c_norm_sq = T(0);
for (int i = 0; i < c_r.extent(0); ++i)
	c_norm_sq += c_r(i) * c_r(i);

std::cout << "||c_coeff|| = " << std::sqrt(std::abs(c_norm_sq)) << ", ||res_vec|| = " << res_mag.back() << std::endl;
std::cout << "diff " << std::abs(std::sqrt(std::abs(c_norm_sq)) - res_mag.back()) << std::endl;
std::cout << "Checking Arnoldi relation after restart:" << std::endl;
for (int j = 0; j < H.extent(1); ++j)  // for each column
{
	// Compute A * Qn[j]
	solArray<T, B, D> Av(Qn[0].len());
	bsmv<T, B, D>(Av, A, Qn[j]);
	preCond(Av);

	// Compute sum_i H[i,j] * Qn[i]
	solArray<T, B, D> Hv(Qn[0].len());
	for (int i = 0; i < H.extent(0); ++i)
		if (i < Qn.size())
			for (int il = 0; il < Hv.len(); ++il)
				Hv(il) += H(i, j) * Qn[i](il);

	// Check difference
	auto diff = infNorm(Av - Hv);
	std::cout << "Column " << j << ": ||A*v_j - H*V||_inf = " << diff << std::endl;
}
//*/

/*
//orthonormality check P_k or Pkp1
double mx = 0.0;
for (int i = 0; i < k; ++i)
{
	auto v_i{ P_k(blitz::Range::all(), i) };
	for (int j = 0; j < k; ++j)
	{
		auto v_j{ P_k(blitz::Range::all(), j) };
		//auto d = blitz::dot(blitz::conj(v_i), v_j);
		auto d = blitz::dot(v_i, v_j);
		if (i != j) mx = std::max(mx, std::abs(d));
		std::cout << i << ' ' << j << ' ' << d << std::endl;
	}
}
std::cout << mx << std::endl;
//*/


} //end namespace BlockSp

#endif //end BLOCKSP_DEFLATED_RESTART_HPP