#ifndef BLOCKSP_CONTAINERS_HPP
#define BLOCKSP_CONTAINERS_HPP

#define BLOCKSP_VERSION_MAJOR 2
#define BLOCKSP_VERSION_MINOR 0

#include <blitz/array.h>

#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

///blocksp_containers.hpp defines the fundamental containers of BlockSp,
///a templated C++ library for block sparse linear algebra.
///These include:
/// 
///Sp::CSR		- General/templated Condensed Row Storage (CSR) sparse matrix. \n
///BlockSp		- Block CSR Matrix. \n
///solArray		- Multi-dimension block Solution Array/Vector. \n 
///	Compressed versions of CSR and BlockSp are available for faster solves.
///  
///Notation: B = block size of solArray, D = dimension of solArray. \n
///BlockSp matrices typically have block size of B or B*D. 
///B = D = 1 reduces to standard sparse linear algebra. \n
///Data type T, block size B, and dimension D are compile-time template parameters, 
///resulting in fast, optimized operations. 
/// 
///This file also includes dot product, infinity norm, magnitude, and two norm functions for solArrays. \n
///And also includes fundamental types ScalarType, ComplexType, and is_complex. \n
///BlockSp matrix-matrix (bsmm) and matrix-vector (bsmv) multiplication are in blocksp_multiply.hpp.		\n
///Preconditioned conjugate gradient, GMRES, and Arnoldi eigenvalue solvers are in blocksp_solvers.hpp. \n 
///The dense linear algebra routines, e.g. multiplication and solvers, are in blocksp_dense_multiply/solvers.hpp. \n 
///Preconditioners are in blocksp_preconditioners.hpp.
namespace BlockSp
{
	//////////////////////////////////
	//Forward declare data structures 

	///Sp sparse namespace: contains a general templated 
	///Condensed Sparse Row Storage (CSR) matrix and a compressed version.
	namespace Sp
	{
		template<typename T>
		class Entry;
		template<typename T>
		class CSR;
		template<typename T, bool Sorted>
		struct CSR_Compressed;
	}

	///@typedef BlockSp - a Sp::CSR matrix with dense square BxB 
	/// sized matrices/blocks as elements. i.e., the A in Ax = b.
	///BlockSp_Compr is the compressed version. The base version is used
	/// for matrix construction/input/modification. The compressed version
	/// speeds up the various algoithms (bsmv, preconditioners, iterative solvers, etc).
	/// 
	///Dense matrix data type templated on T.
	///The block size B is specified at compile time via template 
	/// for fast optimized stack-based operations.
	///Note: for problems with a D-dimension solArray<T,B,D>, 
	/// matrix block sizes are often B*D.
	template<typename T, int B>
	using BlockSp = Sp::CSR<blitz::TinyMatrix<T, B, B>>;
	template<typename T, int B>
	using BlockSp_Compr = Sp::CSR_Compressed<blitz::TinyMatrix<T, B, B>, false>;
	template<typename T, int B>
	using BlockSp_ComprSorted = Sp::CSR_Compressed<blitz::TinyMatrix<T, B, B>, true>;


	/////////////////////////////////
	//Primary types

	///Check if template type is complex.
	template <typename T>
	struct is_complex : std::false_type {};
	template <std::floating_point T>
	struct is_complex<std::complex<T>> : std::true_type {};
	template<typename T>
	inline constexpr bool is_complex_v = is_complex<T>::value;

	///Extract value type from std::complex
	//Primary template - for non-complex types, S = T
	template<typename T>
	struct ScalarType { using type = T; };
	// Specialization for std::complex<U> - extract U
	template<typename U>
	struct ScalarType<std::complex<U>> { using type = U; };
	template<typename T>
	using ScalarType_t = typename ScalarType<T>::type;

	///Create std::complex from scalar
	template<typename T>
	struct ComplexType { using type = std::complex<T>; };
	template<typename U>
	struct ComplexType<std::complex<U>> { using type = std::complex<U>; };
	template<typename T>
	using ComplexType_t = typename ComplexType<T>::type;


	///////////////////////////////////////
	//General Sparse Class
	///////////////////////////////////////
	namespace Sp
	{

		//Entries into the CSR sparse matrix include the column index and value at that index.
		template<typename T>
		class Entry
		{
		private:
			int m_colIndex;	///< the column index
			T m_data;				///< the data value: templated on type T.

		public:

			///Constructors
			Entry() : m_colIndex{ -1 }, m_data{} {}
			Entry(int colInd, T data) : m_colIndex{ colInd }, m_data{ data } {}

			///Destructor
			~Entry() {}

			///Copy constructor
			Entry(const Entry& other)
			{
				m_colIndex = other.m_colIndex;
				m_data = other.m_data;
			}

			///Copy assignment
			Entry& operator= (const Entry& other)
			{
				m_colIndex = other.m_colIndex;
				m_data = other.m_data;
				return *this;
			}

			///Move constructor
			Entry(Entry<T>&& other) : m_colIndex{ other.m_colIndex }, m_data{ std::move(other.m_data) } {}

			///Move assignment
			Entry& operator=(Entry<T>&& other)
			{
				m_colIndex = other.m_colIndex;
				std::swap(m_data, other.m_data);
				return *this;
			}

			///Member functions
			int& colInd() { return m_colIndex; }
			const int& colInd() const { return m_colIndex; }
			T& data() { return m_data; }
			const T& data() const { return m_data; }

		};//end Entry

		template<typename T>
		std::ostream& operator << (std::ostream& out, const Entry<T>& ent)
		{
			out << "(" << ent.colInd() << "," << ent.data() << "), ";
			return out;
		}


		///Sp::CSR - A general templated Condensed Sparse Row Storage (CSR) matrix
		template<typename T>
		class CSR
		{
		private:

			////////////////////////////////////
			//Member data
			////////////////////////////////////

			int m_m, m_n, nnz;   ///< m_m = # of rows, m_n = # of columns, nnz = # of non-zeros in sparse matrix
			T m_zero;						 ///< data type T's zero value

			///CSR entry storage vector, each entry[p][q] contains
			///the (p, colIndex[p][q]) element of the matrix and the column index.
			std::vector<std::vector<Entry<T>>>	m_entries;

		public:

			////////////////////////////////////
			//Constructors
			////////////////////////////////////

			///Matrix of zero size
			CSR() : m_zero{ 0.0 }, m_m{ 0 }, m_n{ 0 }, nnz{ 0 } {}

			///mxn matrix
			CSR(int m, int n) : m_entries(m),
				m_zero{ 0.0 }, m_m{ m }, m_n{ n }, nnz{ 0 } {
			}

			///Matrix size given by TinyVector
			CSR(const blitz::TinyVector<int, 2>& size) : m_entries(size(0)),
				m_zero{ 0.0 }, m_m{ size(0) }, m_n{ size(1) }, nnz{ 0 } {
			}

			///Square nxn matrix
			CSR(int n) : m_entries(n),
				m_zero{ 0.0 }, m_m{ n }, m_n{ n }, nnz{ 0 } {
			}

			///Copy assignment
			CSR& operator= (const CSR& other)
			{
				m_m = other.m_m;
				m_n = other.m_n;
				nnz = other.nnz;
				m_zero = other.m_zero;
				m_entries = other.m_entries;
				return *this;
			}

			////////////////////////////////////
			//Indexing, starts from 0

			///Non-const indexing: if value does not exist, create element at (i,j) and make it zero.
			T& operator()(int i, int j)
			{
				//check index bounds
				if (0 > i || i >= m_m || 0 > j || j >= m_n)
					std::cout << "index out of bounds: " << blitz::TinyVector<int, 2>(i, j) << ' '
					<< blitz::TinyVector<int, 2>(m_m, m_n) << std::endl;
				assert(0 <= i && i < m_m && 0 <= j && j < m_n);

				//if the index already exists, return its value
				for (int in = 0; in < m_entries[i].size(); in++)
					if (m_entries[i][in].colInd() == j)	return m_entries[i][in].data();

				//else add new index and make it zero
				m_entries[i].push_back(Entry(j, m_zero));
				++nnz;
				return m_entries[i].back().data();
			}

			///Const indexing (i,j)
			const T& operator()(int i, int j) const
			{
				//check index bounds
				if (0 > i || i >= m_m || 0 > j || j >= m_n)
					std::cout << "index out of bounds: " << blitz::TinyVector<int, 2>(i, j) << ' '
					<< blitz::TinyVector<int, 2>(m_m, m_n) << std::endl;
				assert(0 <= i && i < m_m && 0 <= j && j < m_n);

				//if the index already exists, return its value
				for (int in = 0; in < m_entries[i].size(); in++)
					if (m_entries[i][in].colInd() == j)	return m_entries[i][in].data();
				return m_zero; //else return zero
			}

			///Index via TinyVector
			T& operator()(const blitz::TinyVector<int, 2>& index)
			{
				return this->operator()(index(0), index(1));
			}

			const T& operator()(const blitz::TinyVector<int, 2>& index) const
			{
				return this->operator()(index(0), index(1));
			}

			///Value at (i,j)
			T& at(int i, int j)
			{
				assert(0 <= i && i < m_m && 0 <= j && j < m_n);
				//if the index is already in m_colindex return its value
				for (int in = 0; in < m_entries[i].size(); in++)
					if (m_entries[i][in].colInd() == j)	return m_entries[i][in].data();
				return m_zero; //else return zero
			}

			const T& at(int i, int j) const
			{
				assert(0 <= i && i < m_m && 0 <= j && j < m_n);
				for (int in = 0; in < m_entries[i].size(); in++)
					if (m_entries[i][in].colInd() == j)	return m_entries[i][in].data();
				return m_zero;
			}

			///See if matrix is nonzero at (i,j) 
			inline bool scan(int i, int j) const
			{
				for (int in = 0; in < m_entries[i].size(); ++in)
					if (m_entries[i][in].colInd() == j)	return true;
				return false;
			}

			///Access the entry vectors
			const std::vector<std::vector<Entry<T>>>& get_entries() const { return m_entries; }
			std::vector<std::vector<Entry<T>>>& get_entries() { return m_entries; }
			const std::vector<Entry<T>>& get_entries(int im) const { return m_entries[im]; }
			std::vector<Entry<T>>& get_entries(int im) { return m_entries[im]; }

			///Access the matrix sizes 
			const int& get_m() const { return m_m; }
			int& get_m() { return m_m; }
			const int& get_n() const { return m_n; }
			int& get_n() { return m_n; }
			const int& get_nnz() const { return nnz; }
			int& get_nnz() { return nnz; }

			///////////////////////////
			//Matrix information

			///Return size of matrix
			inline blitz::TinyVector<int, 2> size() const
			{
				return blitz::TinyVector<int, 2>{ m_m, m_n };
			}

			inline int size(int dim) const
			{
				return size()(dim);
			}

			///Return number of rows m_m / number of columns m_n / number of nonzeros nnz
			inline int m() const { return m_m; }
			inline int n() const { return m_n; }
			inline int nz() const { return nnz; }

			///Get size of a row's entry vector (number of cols in that row).
			inline size_t colSize(int im) const { return m_entries[im].size(); }

			///Get column index/indices
			inline int colInd(int im, int in) const { return m_entries[im][in].colInd(); }
			inline int colInd(const blitz::TinyVector<int, 2>& ind) const { return m_entries[ind(0)][ind(1)].colInd(); }
			inline std::vector<int> colInd(int im) const
			{
				std::vector<int> cols;
				cols.reserve(m_entries[im].size());
				for (const auto& ent : m_entries[im])
					cols.push_back(ent.colInd());
				return cols;
			}

			///Increase column size.
			///Note: this function is for bookkeeping and does not modify m_entries
			inline void addCol(int nCol = 1) { m_n += nCol; }

			////////////////////////////
			//Matrix manipulation

			///Loop through each entry of the sparse matrix.
			template<typename F>
			void for_each_entry(F&& f) const
			{
				for (int im = 0; im < m_m; ++im)
					for (const auto& ent : m_entries[im])
						f(ent.colInd(), ent.data());
			}

			///Loop through each entry of a row within the sparse matrix.
			template<typename F>
			void for_each_row_entry(int im, F&& f) const
			{
				for (const auto& ent : m_entries[im])
					f(ent.colInd(), ent.data());
			}

			///Create transpose matrix, for simple types
			///Note: conjugate transpose for complex types.
			CSR<T> transpose() const
			{
				CSR<T> trans(m_n, m_m);
				for (int im = 0; im < m_m; im++)
					for (int in = 0; in < m_entries[im].size(); in++)
						if constexpr (is_complex_v<T>) trans(m_entries[im][in].colInd(), im) = std::conj(m_entries[im][in].data());
						else trans(m_entries[im][in].colInd(), im) = m_entries[im][in].data();
				return trans;
			}

			///Lower triangular part: int i = number of diags above/below main diag
			CSR<T> lowTri(int i = 0) const
			{
				CSR<T> low(m_m, m_n);
				for (int im = 0; im < m_m; im++)
					for (int in = 0; in < m_entries[im].size(); in++)
						if (m_entries[im][in].colInd() - i <= im)
							low(im, m_entries[im][in].colInd()) = this ->operator()(im, m_entries[im][in].colInd());
				return low;
			}

			///Upper triangular part: int i = number of diags above/below main diag
			CSR<T> upTri(int i = 0) const
			{
				CSR<T> up(m_m, m_n);
				for (int im = 0; im < m_m; im++)
					for (int in = 0; in < m_entries[im].size(); in++)
						if (m_entries[im][in].colInd() - i >= im)
							up(im, m_entries[im][in].colInd()) = this->operator()(im, m_entries[im][in].colInd());
				return up;
			}

			/////////////////////////////
			//Operator overloading

			//Arithmetic
			CSR<T>& operator+= (const CSR<T>& other)
			{
				assert(other.m_m == m_m && other.m_n == m_n);
				for (int im = 0; im < m_m; ++im)
					for (int in = 0; in < other.m_entries[im].size(); ++in)
						this->operator()(im, other.m_entries[im][in].colInd()) += other(im, other.m_entries[im][in].colInd());
				return *this;
			}

			CSR<T>& operator-= (const CSR<T>& other)
			{
				assert(other.m_m == m_m && other.m_n == m_n);
				for (int im = 0; im < m_m; ++im)
					for (int in = 0; in < other.m_entries[im].size(); ++in)
						this->operator()(im, other.m_entries[im][in].colInd()) -= other(im, other.m_entries[im][in].colInd());
				return *this;
			}

			CSR<T> operator+ (const CSR<T>& other)
			{
				assert(other.m_m == m_m && other.m_n == m_n);
				CSR<T> plus = *this;
				for (int im = 0; im < m_m; ++im)
					for (int in = 0; in < other.m_entries[im].size(); ++in)
						plus(im, other.m_entries[im][in].colInd()) += other(im, other.m_entries[im][in].colInd());
				return plus;
			}

			CSR<T> operator- (const CSR<T>& other)
			{
				assert(other.m_m == m_m && other.m_n == m_n);
				CSR<T> minus = *this;
				for (int im = 0; im < m_m; ++im)
					for (int in = 0; in < other.m_entries[im].size(); ++in)
						minus(im, other.m_entries[im][in].colInd()) -= other(im, other.m_entries[im][in].colInd());
				return minus;
			}

			CSR<T>& operator*= (T val)
			{
				for (int im = 0; im < m_m; ++im)
					for (auto& ent : m_entries[im])
						ent.data() *= val;
				return *this;
			}

			CSR<T>& operator/= (T val)
			{
				for (int im = 0; im < m_m; ++im)
					for (auto& ent : m_entries[im])
						ent.data() /= val;
				return *this;
			}

			///////////////////////
			//Member functions

			///Add matrix with more rows, same columns, to my matrix. My rows only
			CSR<T>& addLarger(const CSR<T>& other)
			{
				assert(other.m_m >= m_m && other.m_n == m_n);
				for (int im = 0; im < m_m; ++im)
					for (int in = 0; in < other.m_entries[im].size(); ++in)
						this->operator()(im, other.m_entries[im][in].colInd()) += other(im, other.m_entries[im][in].colInd());
				return *this;
			}

			///Subtract matrix with more rows, same columns, to my matrix. My rows only
			CSR<T>& subtractLarger(const CSR<T>& other)
			{
				assert(other.m_m >= m_m && other.m_n == m_n);
				for (int im = 0; im < m_m; ++im)
					for (int in = 0; in < other.m_entries[im].size(); ++in)
						this->operator()(im, other.m_entries[im][in].colInd()) -= other(im, other.m_entries[im][in].colInd());
				return *this;
			}

			///Get a vector of all nonzero indices
			std::vector<blitz::TinyVector<int, 2>> getIndices() const
			{
				std::vector<blitz::TinyVector<int, 2>> indices;
				for (int im = 0; im < m_m; im++)
					for (int in = 0; in < m_entries[im].size(); in++)
						indices.push_back(blitz::TinyVector<int, 2>(im, m_entries[im][in].colInd()));
				return indices;
			}

			///Get the nonzero row indices in a column
			std::vector<int> getRowIndicesForCol(int iCol) const
			{
				std::vector<int> ind;
				for (int im = 0; im < m_m; ++im)
					for (const auto& ent : m_entries[im])
						if (ent.colInd() == iCol) ind.push_back(im);
				return ind;
			}

			///Get the nonzero row indices in a column, unordered_set version
			std::unordered_set<int> getRowIndicesForCol_us(int iCol) const
			{
				std::unordered_set<int> ind;
				for (int im = 0; im < m_m; ++im)
					for (const auto& ent : m_entries[im])
						if (ent.colInd() == iCol) ind.emplace(im);
				return ind;
			}

			///Get all nonzero row indices, sorted into columns.
			///If ignore_diags, the diagonal index is not stored.
			std::vector<std::unordered_set<int>> rowIndicesByCol(bool ignore_diag = false) const
			{
				std::vector<std::unordered_set<int>> ind(n());
				for (int im = 0; im < m_m; ++im)
					for (const auto& ent : m_entries[im])
						if (ignore_diag && im == ent.colInd()) {}
						else ind[ent.colInd()].emplace(im);
				return ind;
			}

			///matVec (for simple types)
			std::vector<T> operator*(const std::vector<T>& v) const
			{
				assert(v.size() == m_n);
				std::vector<T> product(m_m); //initialize product vector
				for (int im = 0; im < m_m; im++)
				{
					T prod = 0.0;
					for (int in = 0; in < m_entries[im].size(); in++) //sparse multiplication
						prod += m_entries[im][in].data() * v[m_entries[im][in].colInd()];
					product[im] = prod;
				}
				return product;
			}

			///Shrink the matrix's vectors' capacity to its size - reduces storage,
			///It is best practice to use this function once all matrix values are set.
			/// One should avoid modifying the matrix from this point on.
			inline void shrink_to_fit()
			{
				for (int im = 0; im < m_m; ++im)
					m_entries[im].shrink_to_fit();
			}

			///Reserve space in a row of the entry vectors
			inline void reserve(int im, int size)
			{
				m_entries[im].reserve(size);
			}

			///Erase the entry at index (i, j)
			inline void erase(int i, int j)
			{
				for (int in = 0; in < m_entries[i].size(); ++in)
					if (m_entries[i][in].colInd() == j)
					{
						m_entries[i].erase(m_entries[i].begin() + in);
						--nnz;
						return;
					}
			}

			///Clear the CSR matrix.
			///Warning! This will destroy your matrix!
			inline void clear()
			{
				m_m = m_n = nnz = 0;
				m_entries = std::vector<std::vector<Entry<T>>>();
			}

			///Compress the matrix into the CSR_Compressed format, copying the data.
			///The compressed format will be faster within bsmv and iterative solvers.
			///If sorted, then the entries within each row are sorted by column index.
			///This allows for faster binary search indexing.
			template<bool Sorted = true>
			inline CSR_Compressed<T, Sorted> compress() const
			{
				CSR_Compressed<T, Sorted> comp;
				comp.m_m = m_m;	comp.m_n = m_n;	comp.nnz = nnz;
				comp.outer.resize(m_m + 1);
				comp.inner.resize(nnz);
				int cnt = 0;
				for (int im = 0; im < m_m; ++im)
				{
					std::copy(m_entries[im].begin(), m_entries[im].end(), comp.inner.begin() + cnt);
					comp.outer[im] = cnt;
					cnt += m_entries[im].size();

					//If sorted, sort each row by column index
					if constexpr (Sorted)
						std::sort(comp.inner.begin() + comp.outer[im], comp.inner.begin() + cnt,
							[](const Entry<T>& a, const Entry<T>& b) { return a.colInd() < b.colInd(); });
				}
				comp.outer[m_m] = nnz;
				return comp;
			}

			///Compress the matrix into the CSR_Compressed format, moving the data
			///The compressed format will be faster within bsmv and iterative solvers
			///Warning! This function clears the current CSR matrix. 
			template<bool Sorted = true>
			inline CSR_Compressed<T, Sorted> compress_and_move()
			{
				CSR_Compressed<T, Sorted> comp;
				comp.m_m = m_m;
				comp.m_n = m_n;
				comp.nnz = nnz;
				comp.outer.resize(m_m + 1);
				comp.inner.resize(nnz);
				int cnt = 0;
				for (int im = 0; im < m_m; ++im)
				{
					std::move(m_entries[im].begin(), m_entries[im].end(), comp.inner.begin() + cnt);
					comp.outer[im] = cnt;
					cnt += m_entries[im].size();

					//If sorted, sort each row by column index
					if constexpr (Sorted)
						std::sort(comp.inner.begin() + comp.outer[im], comp.inner.begin() + cnt,
							[](const Entry<T>& a, const Entry<T>& b) { return a.colInd() < b.colInd(); });
				}
				comp.outer[m_m] = nnz;
				this->clear();
				return comp;
			}

			///Print sparse matrix. nRows is number of rows printed, default all.
			void print(int nRows = -1) const
			{
				int p_m = nRows < 0 ? m_m : nRows;
				for (int im = 0; im < p_m; im++)
				{
					for (int in = 0; in < m_entries[im].size(); in++)
					{
						std::cout << "(";
						std::cout << im << "," << m_entries[im][in].colInd() << ": " << m_entries[im][in].data();
						std::cout << ")";
					}
					std::cout << std::endl;
				}
			}

			///Print the sparsity structure. nRows is number of rows printed, default all
			void printSparsity(int nRows = -1) const
			{
				int p_m = nRows < 0 ? m_m : nRows;
				for (int im = 0; im < p_m; ++im)
				{
					for (int in = 0; in < m_entries[im].size(); in++)
					{
						std::cout << "(";
						std::cout << im << "," << m_entries[im][in].colInd() << ") ";
					}
					std::cout << std::endl;
				}
			}

		}; //end CSR

		///Arithmetic operator overloads
		template<typename T>
		CSR<T> operator* (T val, CSR<T> A) { return A *= val; }
		template<typename T>
		CSR<T> operator* (CSR<T> A, T val) { return A *= val; }
		template<typename T>
		CSR<T> operator/ (CSR<T> A, T val) { return A /= val; }

		///BlockSp specific arithmetic overloads
		///Plus +
		template<typename T, int B>
		BlockSp<T, B> operator+ (T val, BlockSp<T, B> A) { return A += val; }
		template<typename T, int B>
		BlockSp<T, B> operator+ (BlockSp<T, B> A, T val) { return A += val; }
		template<typename T, int B>
		BlockSp<T, B> operator+ (BlockSp<T, B> A1, const BlockSp<T, B>& A2) { return A1 += A2; }

		///Minus -
		template<typename T, int B>
		BlockSp<T, B> operator- (BlockSp<T, B> A, T val) { return A -= val; }
		template<typename T, int B>
		BlockSp<T, B> operator- (BlockSp<T, B> A1, const BlockSp<T, B>& A2) { return A1 -= A2; }

		///Prod/div */
		template<typename T, int B>
		BlockSp<T, B> operator* (T val, BlockSp<T, B> A) { return A *= val; }
		template<typename T, int B>
		BlockSp<T, B> operator* (BlockSp<T, B> A, T val) { return A *= val; }
		template<typename T, int B>
		BlockSp<T, B> operator/ (BlockSp<T, B> A, T val) { return A /= val; }

		//Complex with ScalarType
		template<typename T, int B, std::enable_if_t<!std::is_same<T, ScalarType_t<T>>::value, bool> = true>
		BlockSp<T, B> operator+ (ScalarType_t<T> val, BlockSp<T, B> A) { return A += T(val); }
		template<typename T, int B, std::enable_if_t<!std::is_same<T, ScalarType_t<T>>::value, bool> = true>
		BlockSp<T, B> operator+ (BlockSp<T, B> A, ScalarType_t<T> val) { return A += T(val); }
		template<typename T, int B, std::enable_if_t<!std::is_same<T, ScalarType_t<T>>::value, bool> = true>
		BlockSp<T, B> operator- (BlockSp<T, B> A, ScalarType_t<T> val) { return A -= T(val); }
		template<typename T, int B, std::enable_if_t<!std::is_same<T, ScalarType_t<T>>::value, bool> = true>
		BlockSp<T, B> operator* (ScalarType_t<T> val, BlockSp<T, B> A) { return A *= T(val); }
		template<typename T, int B, std::enable_if_t<!std::is_same<T, ScalarType_t<T>>::value, bool> = true>
		BlockSp<T, B> operator* (BlockSp<T, B> A, ScalarType_t<T> val) { return A *= T(val); }
		template<typename T, int B, std::enable_if_t<!std::is_same<T, ScalarType_t<T>>::value, bool> = true>
		BlockSp<T, B> operator/ (BlockSp<T, B> A, ScalarType_t<T> val) { return A /= T(val); }


		///Fully compressed CSR format
		///Use regular CSR/BlockSp for matrix creation/element insertion, then compress and solve.
		///Note: this format is read-only.
		/// 
		///Formatted via two vectors:
		///The inner vector contains the sparse matrix entries - the column index and data value.
		/// this vector has length equal to the number of nonzeros (nnz)
		///The outer vector stores the index within the inner vector where that row begins. 
		/// Length m_m+1, the final entry is nnz.
		/// e.g. [1  0  2]
		///			 [3  4  0]  has  inner: [(0,1), (2, 2), (0, 3), (1, 4), (2, 5)]
		///			 [0  0  5]	and	 outer  [0, 2, 4, 5]
		template<typename T, bool Sorted = true>
		struct CSR_Compressed
		{
			int m_m, m_n, nnz;
			std::vector<int> outer;
			std::vector<Entry<T>> inner;

			CSR_Compressed() : m_m{ 0 }, m_n{ 0 }, nnz{ 0 } {}

			///Return number of rows m_m / number of columns m_n / number of nonzeros nnz
			inline int m() const { return m_m; }
			inline int n() const { return m_n; }
			inline int nz() const { return nnz; }

			///Index into the compressed CSR matrix.
			///Note: this form is read-only and an exception will be thrown if index (i,j) is not found.
			const T& operator()(int i, int j) const
			{
				//check index bounds
				if (0 > i || i >= m_m || 0 > j || j >= m_n)
					std::cout << "index out of bounds: " << blitz::TinyVector<int, 2>(i, j) << ' '
					<< blitz::TinyVector<int, 2>(m_m, m_n) << std::endl;
				assert(0 <= i && i < m_m && 0 <= j && j < m_n);

				//If the index already exists, return its value, else throw exception.
				//If sorted, use binary seach; else linear
				if constexpr (Sorted)
				{
					//binary search
					int lo = outer[i], hi = outer[i + 1];
					while (lo < hi)
					{
						int mid = lo + (hi - lo) / 2;
						int col = inner[mid].colInd();
						if (col == j) return inner[mid].data();
						else if (col < j) lo = mid + 1;
						else hi = mid;
					}
				}
				else //linear
					for (int k = outer[i]; k < outer[i + 1]; ++k)
						if (inner[k].colInd() == j)
							return inner[k].data();
				throw std::out_of_range("CSR_Compressed: index not found");
			}

			///Loop through each entry of the compressed sparse matrix.
			template<typename F>
			void for_each_entry(F&& f) const
			{
				for (int im = 0; im < m_m; ++im)
					for (int k = outer[im]; k < outer[im + 1]; ++k)
						f(inner[k].colInd(), inner[k].data());
			}

			///Loop through each entry of a row within the compressed sparse matrix.
			template<typename F>
			void for_each_row_entry(int im, F&& f) const
			{
				for (int k = outer[im]; k < outer[im + 1]; ++k)
					f(inner[k].colInd(), inner[k].data());
			}

			///Create a vector<vector> CSR matrix from the compressed form, copying the data.
			///The CSR form is used for matrix modification/input.
			CSR<T> decompress() const
			{
				CSR<T> decomp(m_m, m_n);
				decomp.get_nnz() = nnz;
				for (int im = 0; im < m_m; ++im)
				{
					int start = outer[im];
					int len = outer[im + 1] - start;
					decomp.get_entries(im).resize(len);
					std::copy(inner.begin() + start, inner.begin() + start + len,
						decomp.get_entries(im).begin());
				}
				return decomp;
			}

			///Create a vector<vector> CSR matrix from the compressed form, moving the data.
			///This form is used for matrix modification/input.
			///Warning! This function clears the current CSR_Compressed matrix!
			CSR<T> decompress_and_move()
			{
				CSR<T> decomp(m_m, m_n);
				decomp.get_nnz() = nnz;
				for (int im = 0; im < m_m; ++im)
				{
					int start = outer[im];
					int len = outer[im + 1] - start;
					decomp.get_entries(im).resize(len);
					std::move(inner.begin() + start, inner.begin() + start + len,
						decomp.get_entries(im).begin());
				}
				this->clear();
				return decomp;
			}

			///Clear a CSR_Compressed matrix.
			///Warning! This will destroy your matrix!
			void clear()
			{
				m_m = m_n = nnz = 0;
				outer = std::vector<int>();
				inner = std::vector<Entry<T>>();
			}

		}; //end CSR_Compressed

	}//end namespace Sp


	/////////////////////////
	//solArray 
	/////////////////////////

	///Solution Arrays (solArray) are the block vectors used by BlockSp.
	///i.e. the x and b in Ax = b. 
	///solArrays have arbitrary dimension and block size.
	///They may be used to represent multi-dimensional solutions, e.g. for high-order finite element problems. 
	///For example, the velocity and pressure variables within a 3D Navier-Stokes system (u,v,w,p)
	///can be stored in a single solArray.
	/// 
	///Structured as blitz::Array<TinyVector<TinyVector<T, Blocksize>, Dimension>, 1>,
	///the length of the solArray is equal to m_m or m_n of the BlockSp matrix, i.e., number of rows or columns.
	///Within each "element" of a solArray is a Dimension TinyVector that contains the Block TinyVectors.
	///solArray u is indexed as u(ie, dim)(ib) when D > 1 and u(ie)(ib) when D=1 (default).
	///ie = element index, dim = dimension index, ib = block index. 
	/// 
	///For example, a solArray u of length 4, with B=3, D=2 will look like: u = \n
	///	[ ({x1,x2,x3},{y1,y2,y3}) ] \n
	///	[ ({x1,x2,x3},{y1,y2,y3}) ] \n
	///	[ ({x1,x2,x3},{y1,y2,y3}) ] \n
	///	[ ({x1,x2,x3},{y1,y2,y3}) ] 
	/// 
	///Notes: setting B = D = 1, reduces the problem to regular sparse matrix-vector operations.
	///Block sparse matrix-vector multiplication can be performed along each dimension seperately or all together.
	///The use of TinyVectors as blocks/dimensions allows for fast stack-based compile-time optimized operations.
	template<typename T, int B, int D = 1>
	class solArray
	{
	private:

		/////////////////////////
		//Member data
		/////////////////////////

		blitz::Array<blitz::TinyVector<blitz::TinyVector<T, B>, D>, 1> m_arr; ///< array data
		int m_len; ///< length

	public:

		/////////////////////////
		//Constructors
		/////////////////////////

		solArray() : m_len{ 0 } {}

		///solArray with length len, set to zero
		solArray(int len) : m_len{ len }
		{
			m_arr.resize(len);
			for (auto& el : m_arr) el = T(0.0);
		}

		///solArray with length len, set to val
		solArray(int len, T val) : m_len{ len }
		{
			m_arr.resize(len);
			for (auto& el : m_arr) el = val;
		}

		///Destructor
		~solArray() {}

		///Copy constructor, blitz::arrays must be resized before copy
		solArray(const solArray<T, B, D>& other) : m_len{ other.m_len }
		{
			m_arr.resize(other.m_len); m_arr = other.m_arr;
		}

		///Copy assignment
		solArray& operator=(const solArray<T, B, D>& other)
		{
			m_len = other.m_len;
			m_arr.resize(other.m_len); m_arr = other.m_arr;
			return *this;
		}

		///Move constructor
		solArray(solArray<T, B, D>&& other) : m_len{ other.m_len }, m_arr{ std::move(other.m_arr) } {}

		///Move assignment
		solArray& operator=(solArray<T, B, D>&& other)
		{
			m_len = other.m_len;
			std::swap(m_arr, other.m_arr);
			return *this;
		}

		///Make solArray from blitz operations nonsense.
		///For example, consider two blitz arrays x and y, x + y is type blitz::_bz_ArrayExpr
		template<typename R>
		solArray(const blitz::_bz_ArrayExpr<R>& other, int length) : m_len{ length }
		{
			m_arr.resize(length);
			m_arr = other;
		}

		//////////////////////////
		//Member functions
		//////////////////////////

		inline int len() const { return m_len; }
		inline void resize(int l, T val = 0.0)
		{
			m_len = l;
			m_arr.resize(l);
			for (auto& el : m_arr) el = val;
		}
		inline void resize_and_preserve(int l) { m_len = l; m_arr.resizeAndPreserve(l); }
		inline void reset() { m_len = 0; m_arr.reset(); }
		inline void set_to_zero() { for (auto& el : m_arr) el = T(0.0); }
		inline const auto& data() const { return m_arr.data(); }
		inline auto& data() { return m_arr.data(); }
		inline const auto& extent() const { return m_arr.extent(); }
		inline auto extent(int dim) const { return m_arr.extent(dim); }
		inline const blitz::Array<blitz::TinyVector<blitz::TinyVector<T, B>, D>, 1>& arr() const { return m_arr; }
		inline blitz::Array<blitz::TinyVector<blitz::TinyVector<T, B>, D>, 1>& arr() { return m_arr; }

		///////////////////////////
		//Operator overloading
		///////////////////////////

		///////////////////////////
		//Indexing, starts from 0

		///Index u(im) for D = 1 returns Block TinyVector
		template<int DD = D, std::enable_if_t<DD == 1, bool> = true>
		const blitz::TinyVector<T, B>& operator()(int im, int dim = 0) const { return m_arr(im)(0); }

		template<int DD = D, std::enable_if_t<DD == 1, bool> = true>
		blitz::TinyVector<T, B>& operator()(int im, int dim = 0) { return m_arr(im)(0); }

		///Index u(im) for D > 1 returns Dimension TinyVector of Block TinyVector
		template<int DD = D, std::enable_if_t<(DD > 1), bool> = true>
			const blitz::TinyVector<blitz::TinyVector<T, B>, D>& operator()(int im) const { return m_arr(im); }

		template<int DD = D, std::enable_if_t<(DD > 1), bool> = true>
			blitz::TinyVector<blitz::TinyVector<T, B>, D>& operator()(int im) { return m_arr(im); }

		///Index u(im, dim) for D > 1 returns Block TinyVector
		template<int DD = D, std::enable_if_t<(DD > 1), bool> = true>
			const blitz::TinyVector<T, B>& operator()(int im, int dim) const { return m_arr(im)(dim); }

		template<int DD = D, std::enable_if_t<(DD > 1), bool> = true>
			blitz::TinyVector<T, B>& operator()(int im, int dim) { return m_arr(im)(dim); }

		//Note: an overloaded operator() indexing into the Block TinyVectors is not provided,
		//i.e. u(im, ib) for D = 1, and u(im, dim, ib) for D > 1.
		//This indexing is not needed for block sparse operations. 
		//Generally, it is faster and recommended to construct the Block TinyVector locally,
		// then set in the solArray. 
		//i.e. use u(im, dim) = your_tv rather than u(im, dim, ib) = value. 
		//Indexing u(im, dim)(ib) will index into the Block TinyVectors if needed.

		///////////////
		//Arithmetic

		solArray& operator+=(const T& val) { m_arr += val; return *this; }
		solArray& operator-=(const T& val) { m_arr -= val; return *this; }
		solArray& operator*=(const T& val) { m_arr *= val; return *this; }
		solArray& operator/=(const T& val) { m_arr /= val; return *this; }

		solArray& operator+=(const solArray& other) {
			for (int ie = 0; ie < m_len; ++ie)
				for (int dim = 0; dim < D; ++dim) m_arr(ie)(dim) += other(ie, dim);
			return *this;
		}
		solArray& operator-=(const solArray& other) {
			for (int ie = 0; ie < m_len; ++ie)
				for (int dim = 0; dim < D; ++dim) m_arr(ie)(dim) -= other(ie, dim);
			return *this;
		}
		solArray& operator*=(const solArray& other) {
			for (int ie = 0; ie < m_len; ++ie)
				for (int dim = 0; dim < D; ++dim) m_arr(ie)(dim) *= other(ie, dim);
			return *this;
		}
		solArray& operator/=(const solArray& other) {
			for (int ie = 0; ie < m_len; ++ie)
				for (int dim = 0; dim < D; ++dim) m_arr(ie)(dim) /= other(ie, dim);
			return *this;
		}

		///begin/end for iterating
		auto begin() { return m_arr.begin(); }
		auto end() { return m_arr.end(); }
		auto begin() const { return m_arr.begin(); }
		auto end() const { return m_arr.end(); }

	}; //end solArray

	//io
	template<typename T, int B, int D>
	std::ostream& operator << (std::ostream& out, const solArray<T, B, D>& sa)
	{
		out << sa.arr();
		return out;
	}

	/////////////////
	//Arithmetic

	//Plus +
	template<typename T, int B, int D = 1>
	inline solArray<T, B, D> operator+ (const solArray<T, B, D>& x, T val)
	{
		int len = x.len();
		solArray<T, B, D> plus(len);
		for (int ie = 0; ie < len; ++ie) plus(ie) = x(ie) + val;
		return plus;
	}

	template<typename T, int B, int D = 1>
	inline solArray<T, B, D> operator+ (T val, const solArray<T, B, D>& x)
	{
		int len = x.len();
		solArray<T, B, D> plus(len);
		for (int ie = 0; ie < len; ++ie) plus(ie) = val + x(ie);
		return plus;
	}

	template<typename T, int B, int D = 1>
	inline solArray<T, B, D> operator+ (const solArray<T, B, D>& x1, const solArray<T, B, D>& x2)
	{
		assert(x1.len() == x2.len());
		int len = x1.len();
		solArray<T, B, D> plus(len);
		for (int ie = 0; ie < len; ++ie)
			for (int dim = 0; dim < D; ++dim)
				plus(ie, dim) = x1(ie, dim) + x2(ie, dim);
		return plus;
	}

	//Minus -
	template<typename T, int B, int D = 1>
	inline solArray<T, B, D> operator- (const solArray<T, B, D>& x, T val)
	{
		int len = x.len();
		solArray<T, B, D> minus(len);
		for (int ie = 0; ie < len; ++ie) minus(ie) = x(ie) - val;
		return minus;
	}

	template<typename T, int B, int D = 1>
	inline solArray<T, B, D> operator- (T val, const solArray<T, B, D>& x)
	{
		int len = x.len();
		solArray<T, B, D> minus(len);
		for (int ie = 0; ie < len; ++ie) minus(ie) = val - x(ie);
		return minus;
	}

	template<typename T, int B, int D = 1>
	inline solArray<T, B, D> operator- (const solArray<T, B, D>& x1, const solArray<T, B, D>& x2)
	{
		assert(x1.len() == x2.len());
		int len = x1.len();
		solArray<T, B, D> minus(len);
		for (int ie = 0; ie < len; ++ie)
			for (int dim = 0; dim < D; ++dim)
				minus(ie, dim) = x1(ie, dim) - x2(ie, dim);
		return minus;
	}

	//Product *
	template<typename T, int B, int D = 1>
	inline solArray<T, B, D> operator* (const solArray<T, B, D>& x, T val)
	{
		int len = x.len();
		solArray<T, B, D> mult(len);
		for (int ie = 0; ie < len; ++ie) mult(ie) = x(ie) * val;
		return mult;
	}

	template<typename T, int B, int D = 1>
	inline solArray<T, B, D> operator* (T val, const solArray<T, B, D>& x)
	{
		return operator*(x, val);
	}

	template<typename T, int B, int D = 1, std::enable_if_t<!std::is_same<T, ScalarType_t<T>>::value, bool> = true>
	inline solArray<T, B, D> operator* (const solArray<T, B, D>& x, ScalarType_t<T> val)
	{
		int len = x.len();
		solArray<T, B, D> mult(len);
		for (int ie = 0; ie < len; ++ie) mult(ie) = x(ie) * val;
		return mult;
	}

	template<typename T, int B, int D = 1, std::enable_if_t<!std::is_same<T, ScalarType_t<T>>::value, bool> = true>
	inline solArray<T, B, D> operator* (ScalarType_t<T> val, const solArray<T, B, D>& x)
	{
		int len = x.len();
		solArray<T, B, D> mult(len);
		for (int ie = 0; ie < len; ++ie) mult(ie) = val * x(ie);
		return mult;
	}

	template<typename T, int B, int D = 1>
	inline solArray<T, B, D> operator* (const solArray<T, B, D>& x1, const solArray<T, B, D>& x2)
	{
		assert(x1.len() == x2.len());
		int len = x1.len();
		solArray<T, B, D> mult(len);
		for (int ie = 0; ie < len; ++ie)
			for (int dim = 0; dim < D; ++dim)
				mult(ie, dim) = x1(ie, dim) * x2(ie, dim);
		return mult;
	}

	///Fused Multiply-Addition Operation (axpy) y = a*x
	template<typename T, int B, int D>
	inline void axpy(solArray<T, B, D>& y, T a, const solArray<T, B, D>& x)
	{
		assert(x.extent(0) == y.extent(0));
		for (int ie = 0; ie < x.len(); ++ie)
			y(ie) += a * x(ie);
	}

	//Division /
	template<typename T, int B, int D = 1>
	inline solArray<T, B, D> operator/ (const solArray<T, B, D>& x, T val)
	{
		int len = x.len();
		solArray<T, B, D> div(len);
		for (int ie = 0; ie < len; ++ie) div(ie) = x(ie) / val;
		return div;
	}

	template<typename T, int B, int D = 1, std::enable_if_t<!std::is_same<T, ScalarType_t<T>>::value, bool> = true>
	inline solArray<T, B, D> operator/ (const solArray<T, B, D>& x, ScalarType_t<T> val)
	{
		int len = x.len();
		solArray<T, B, D> div(len);
		for (int ie = 0; ie < len; ++ie) div(ie) = x(ie) / val;
		return div;
	}

	template<typename T, int B, int D = 1>
	inline solArray<T, B, D> operator/ (T val, const solArray<T, B, D>& x)
	{
		int len = x.len();
		solArray<T, B, D> div(len);
		for (int ie = 0; ie < len; ++ie) div(ie) = val / x(ie);
		return div;
	}

	template<typename T, int B, int D = 1>
	inline solArray<T, B, D> operator/ (solArray<T, B, D> x1, const solArray<T, B, D>& x2)
	{
		assert(x1.len() == x2.len());
		int len = x1.len();
		for (int ie = 0; ie < len; ++ie)
			for (int dim = 0; dim < D; ++dim)
				x1(ie, dim) /= x2(ie, dim);
		return x1;
	}


	/////////////////////////////
	//solArray operations
	/////////////////////////////

	//infNorm - infinity norm
	//mag			- magnitude at all points
	//dot			- dot product
	//twoNorm - two norm

	///Take the infinity norm of a solArray.
	///Along all dimensions if dim < 0 (default), specific dimension if dim >= 0.
	template<typename T, int B, int D = 1>
	inline ScalarType_t<T> infNorm(const solArray<T, B, D>& x, int dim = -1)
	{
		assert(dim < D);
		ScalarType_t<T> norm = 0;
		for (const auto& el : x)
		{
			if (dim >= 0) norm = std::max(norm, blitz::max(blitz::abs(el(dim))));
			else for (int dim2 = 0; dim2 < D; ++dim2)
				norm = std::max(norm, blitz::max(blitz::abs(el(dim2))));
		}
		return norm;
	}

	///Take the infinity norm of a solArray along the specified dimensions.
	template<typename T, int B, int D = 1>
	inline ScalarType_t<T> infNorm(const solArray<T, B, D>& x, const std::vector<int>& dims)
	{
		ScalarType_t<T> norm = 0;
		for (const auto& dim : dims)
		{
			assert(dim >= 0 && dim < D);
			norm = std::max(norm, infNorm<T, B, D>(x, dim));
		}
		return norm;
	}

	///Magnitude of solArray at each block point
	///i.e. - mag(x(im)) = (|x(im, dim1)|^2 + ... + |x(im, dimD)|)^2)^0.5
	template<typename T, int B, int D = 1>
	solArray<ScalarType_t<T>, B> mag(const solArray<T, B, D>& x)
	{
		solArray<ScalarType_t<T>, B> magn(x.len());
		for (int ie = 0; const auto& el : x)
		{
			blitz::TinyVector<ScalarType_t<T>, B> magB(0.0);
			for (int dim = 0; dim < D; ++dim)
				for (int ib = 0; ib < B; ++ib)
				{
					T val = el(dim)(ib);
					if constexpr (is_complex_v<T>)
					{
						T val_sqr = std::conj(val) * val;
						magB(ib) += val_sqr.real();
					}
					else magB(ib) += val * val;
				}
			for (int ib = 0; ib < B; ++ib)
				magB(ib) = std::sqrt(magB(ib));
			magn(ie) = magB;
			ie++;
		}
		return magn;
	}

	///Dot product of two solArrays.
	///Along all dimensions if dim < 0 (default), specific dimension if dim >= 0.
	template<typename T, int B, int D = 1>
	T dot(const solArray<T, B, D>& x1, const solArray<T, B, D>& x2, int dim = -1)
	{
		assert(dim < D);
		assert(x1.len() == x2.len());
		T dot = 0.0;
		for (int ie = 0; ie < x1.len(); ++ie)
		{
			if constexpr (is_complex_v<T>) //complex dot product requires complex conjugation
			{
				if (dim >= 0) dot += blitz::dot(blitz::conj(x1(ie, dim)), x2(ie, dim));
				else for (int dim2 = 0; dim2 < D; ++dim2)
					dot += blitz::dot(blitz::conj(x1(ie, dim2)), x2(ie, dim2));
			}
			else
			{
				if (dim >= 0) dot += blitz::dot(x1(ie, dim), x2(ie, dim));
				else for (int dim2 = 0; dim2 < D; ++dim2)
					dot += blitz::dot(x1(ie, dim2), x2(ie, dim2));
			}
		}
		return dot;
	}

	///Two norm function for two solArrays, i.e. square root of dot product: sqrt(x dot y).
	///Along all dimensions if dim < 0, specific dimension if dim >= 0.
	template<typename T, int B, int D = 1>
	inline ScalarType_t<T> twoNorm(const solArray<T, B, D>& x, const solArray<T, B, D>& y, int dim = -1)
	{
		return std::sqrt(std::abs(dot<T, B, D>(x, y, dim)));
	}

	///Two norm function for a solArray x: sqrt(x dot x).
	///Along all dimensions if dim < 0, specific dimension if dim >= 0.
	template<typename T, int B, int D = 1>
	inline ScalarType_t<T> twoNorm(const solArray<T, B, D>& x, int dim = -1)
	{
		return twoNorm<T, B, D>(x, x, dim);
	}

} //end namespace BlockSp

#endif //end BLOCKSP_CONTAINERS_HPP