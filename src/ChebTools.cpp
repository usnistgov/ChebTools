#include "ChebTools/ChebTools.h"
#include "Eigen/Dense"

#include <algorithm>
#include <functional>

#include <chrono>
#include <iostream>
#include <map>
#include <limits>

#ifndef DBL_EPSILON
#define DBL_EPSILON std::numeric_limits<double>::epsilon()
#endif

/// See python code in https://en.wikipedia.org/wiki/Binomial_coefficient#Binomial_coefficient_in_programming_languages
/// This is a direct translation of that code to C++
double binomialCoefficient(const double n, const double k) {
    if (k < 0 || k > n) {
        return 0;
    }
    if (k == 0 || k == n) {
        return 1;
    }
    double _k = std::min(k, n - k); //# take advantage of symmetry
    double c = 1;
    for (double i = 0; i < _k; ++i) {
        c *= (n - i) / (i + 1);
    }
    return c;
}



namespace ChebTools {

    inline bool ValidNumber(double x){
        // Idea from http://www.johndcook.com/IEEE_exceptions_in_cpp.html
        return (x <= DBL_MAX && x >= -DBL_MAX);
    };

    void balance_matrix(const Eigen::MatrixXd &A, Eigen::MatrixXd &Aprime, Eigen::MatrixXd &D) {
        // https://arxiv.org/pdf/1401.5766.pdf (Algorithm #3)
        const int p = 2;
        double beta = 2; // Radix base (2?)
        int iter = 0;
        Aprime = A;
        D = Eigen::MatrixXd::Identity(A.rows(), A.cols());
        bool converged = false;
        do {
            converged = true;
            for (Eigen::Index i = 0; i < A.rows(); ++i) {
                double c = Aprime.col(i).lpNorm<p>();
                double r = Aprime.row(i).lpNorm<p>();
                double s = pow(c, p) + pow(r, p);
                double f = 1;
                if (!ValidNumber(c)){ 
                    std::cout << A << std::endl;
                    throw std::range_error("c is not a valid number in balance_matrix"); }
                if (!ValidNumber(r)) { throw std::range_error("r is not a valid number in balance_matrix"); }
                while (c < r/beta) {
                    c *= beta;
                    r /= beta;
                    f *= beta;
                }
                while (c >= r*beta) {
                    c /= beta;
                    r *= beta;
                    f /= beta;
                }
                if (pow(c, p) + pow(r, p) < 0.95*s) {
                    converged = false;
                    D(i, i) *= f;
                    Aprime.col(i) *= f;
                    Aprime.row(i) /= f;
                }
            }
            iter++;
            if (iter > 50) {
                break;
            }
        } while (!converged);
    }

    class ChebyshevExtremaLibrary {
    private:
        std::map<std::size_t, Eigen::VectorXd> vectors;
        void build(std::size_t N) {
            vectors[N] = (Eigen::VectorXd::LinSpaced(N + 1, 0, N).array()*EIGEN_PI / N).cos();
        }
    public:
        const Eigen::VectorXd & get(std::size_t N) {
            auto it = vectors.find(N);
            if (it != vectors.end()) {
                return it->second;
            }
            else {
                build(N);
                return vectors.find(N)->second;
            }
        }
    };
    static ChebyshevExtremaLibrary extrema_library;
    const Eigen::VectorXd &get_extrema(std::size_t N){ 
        return extrema_library.get(N); 
    }

    class ChebyshevRootsLibrary {
    private:
        std::map<std::size_t, Eigen::VectorXd> vectors;
        void build(std::size_t N) {
            vectors[N] = ((Eigen::VectorXd::LinSpaced(N, 0, N - 1).array() + 0.5)*EIGEN_PI / N).cos();
        }
    public:
        const Eigen::VectorXd & get(std::size_t N) {
            auto it = vectors.find(N);
            if (it != vectors.end()) {
                return it->second;
            }
            else {
                build(N);
                return vectors.find(N)->second;
            }
        }
    };
    static ChebyshevRootsLibrary roots_library;

    class LMatrixLibrary {
    private:
        std::map<std::size_t, Eigen::MatrixXd> matrices;
        void build(std::size_t N) {
            Eigen::MatrixXd L(N + 1, N + 1); ///< Matrix of coefficients
            for (int j = 0; j <= N; ++j) {
                for (int k = j; k <= N; ++k) {
                    double p_j = (j == 0 || j == N) ? 2 : 1;
                    double p_k = (k == 0 || k == N) ? 2 : 1;
                    L(j, k) = 2.0 / (p_j*p_k*N)*cos((j*EIGEN_PI*k) / N);
                    // Exploit symmetry to fill in the symmetric elements in the matrix
                    L(k, j) = L(j, k);
                }
            }
            matrices[N] = L;
        }
    public:
        const Eigen::MatrixXd & get(std::size_t N) {
            auto it = matrices.find(N);
            if (it != matrices.end()) {
                return it->second;
            }
            else {
                build(N);
                return matrices.find(N)->second;
            }
        }
    };
    static LMatrixLibrary l_matrix_library;

    class UMatrixLibrary {
    private:
        std::map<std::size_t, Eigen::MatrixXd> matrices;
        void build(std::size_t N) {
            Eigen::MatrixXd U(N + 1, N + 1); ///< Matrix of coefficients
            for (int j = 0; j <= N; ++j) {
                for (int k = j; k <= N; ++k) {
                    U(j, k) = cos((j*EIGEN_PI*k) / N);
                    // Exploit symmetry to fill in the symmetric elements in the matrix
                    U(k, j) = U(j, k);
                }
            }
            matrices[N] = U;
        }
    public:
        const Eigen::MatrixXd & get(std::size_t N) {
            auto it = matrices.find(N);
            if (it != matrices.end()) {
                return it->second;
            }
            else {
                build(N);
                return matrices.find(N)->second;
            }
        }
    };
    static UMatrixLibrary u_matrix_library;

    // From CoolProp
    template<class T> bool is_in_closed_range(T x1, T x2, T x) { return (x >= std::min(x1, x2) && x <= std::max(x1, x2)); };

    ChebyshevExpansion ChebyshevExpansion::operator+(const ChebyshevExpansion &ce2) const {
        if (m_c.size() == ce2.coef().size()) {
            // Both are the same size, nothing creative to do, just add the coefficients
            return ChebyshevExpansion(std::move(ce2.coef() + m_c), m_xmin, m_xmax); 
        }
        else{
            if (m_c.size() > ce2.coef().size()) {
                Eigen::VectorXd c(m_c.size()); c.setZero(); c.head(ce2.coef().size()) = ce2.coef();
                return ChebyshevExpansion(c+m_c, m_xmin, m_xmax);
            }
            else {
                std::size_t n = ce2.coef().size();
                Eigen::VectorXd c(n); c.setZero(); c.head(m_c.size()) = m_c;
                return ChebyshevExpansion(c + ce2.coef(), m_xmin, m_xmax);
            }
        }
    };
    ChebyshevExpansion& ChebyshevExpansion::operator+=(const ChebyshevExpansion &donor) {
        std::size_t Ndonor = donor.coef().size(), N1 = m_c.size();
        std::size_t Nmin = std::min(N1, Ndonor), Nmax = std::max(N1, Ndonor);
        // The first Nmin terms overlap between the two vectors
        m_c.head(Nmin) += donor.coef().head(Nmin);
        // If the donor vector is longer than the current vector, resizing is needed
        if (Ndonor > N1) {
            // Resize but leave values as they were
            m_c.conservativeResize(Ndonor);
            // Copy the last Nmax-Nmin values from the donor
            m_c.tail(Nmax - Nmin) = donor.coef().tail(Nmax - Nmin);
        }
        return *this;
    }
    ChebyshevExpansion ChebyshevExpansion::operator*(double value) const {
        return ChebyshevExpansion(m_c*value, m_xmin, m_xmax);
    }
    ChebyshevExpansion ChebyshevExpansion::operator+(double value) const {
        Eigen::VectorXd c = m_c;
        c(0) += value;
        return ChebyshevExpansion(c, m_xmin, m_xmax);
    }
    ChebyshevExpansion ChebyshevExpansion::operator-(double value) const {
        Eigen::VectorXd c = m_c;
        c(0) -= value;
        return ChebyshevExpansion(c, m_xmin, m_xmax);
    }
    ChebyshevExpansion& ChebyshevExpansion::operator*=(double value) {
        m_c *= value;
        return *this;
    }
    ChebyshevExpansion ChebyshevExpansion::operator*(const ChebyshevExpansion &ce2) const {

        std::size_t len1 = this->m_c.size(), len2 = ce2.coef().size();
        std::size_t n = len1 + len2 - 2;

        // Create padded vectors, and copy into them the coefficients from this instance 
        // and that of the donor
        Eigen::VectorXd a = Eigen::VectorXd::Zero(n+1), b = Eigen::VectorXd::Zero(n+1);
        a.head(len1) = this->m_c; b.head(len2) = ce2.coef();

        // Get the matrices u and v
        Eigen::MatrixXd u = u_matrix_library.get(n);
        Eigen::MatrixXd v = l_matrix_library.get(n);
        
        // Carry out the calculation of the final coefficients
        return ChebyshevExpansion(v*((u*a).array()*(u*b).array()).matrix(), m_xmin, m_xmax);
    };
    ChebyshevExpansion ChebyshevExpansion::times_x() const {
        double scale_factor = (m_xmax - m_xmin)/2.0;
        Eigen::VectorXd c = m_c*(m_xmax + m_xmin)/2.0; 
        c.conservativeResize(m_c.size()+1); c.tail(1).setZero();
        c(1) += m_c(0)*scale_factor;
        for (std::size_t i = 1; i < m_c.size(); ++i) {
            c(i-1) += 0.5*m_c[i]*scale_factor;
            c(i+1) += 0.5*m_c[i]*scale_factor;
        }
        return ChebyshevExpansion(c, m_xmin, m_xmax);
    };

    const vectype &ChebyshevExpansion::coef() const {
        return m_c;
    };
    /**
    * @brief Do a single input/single output evaluation of the Chebyshev expansion with the inputs scaled in [xmin, xmax]
    * @param x A value scaled in the domain [xmin,xmax]
    */
    double ChebyshevExpansion::y_recurrence(const double x) {
        // Use the recurrence relationships to evaluate the Chebyshev expansion
        std::size_t Norder = m_c.size() - 1;
        // Scale x linearly into the domain [-1, 1]
        double xscaled = (2 * x - (m_xmax + m_xmin)) / (m_xmax - m_xmin);
        // Short circuit if not using recursive solution
        if (Norder == 0){ return m_c[0]; }
        if (Norder == 1) { return m_c[0] + m_c[1]*xscaled; }

        vectype &o = m_recurrence_buffer;
        o(0) = 1;
        o(1) = xscaled;
        for (int n = 1; n < Norder; ++n) {
            o(n + 1) = 2 * xscaled*o(n) - o(n - 1);
        }
        return m_c.dot(o);
    }
    double ChebyshevExpansion::y_Clenshaw(const double x) const {
        std::size_t Norder = m_c.size() - 1;
        // Scale x linearly into the domain [-1, 1]
        double xscaled = (2 * x - (m_xmax + m_xmin)) / (m_xmax - m_xmin);
        // Short circuit if not using recursive solution
        if (Norder == 0) { return m_c[0]; }
        if (Norder == 1) { return m_c[0] + m_c[1]*xscaled; }
        
        double u_k = 0, u_kp1 = m_c[Norder], u_kp2 = 0;
        for (int k = Norder - 1; k >= 1; --k) {
            u_k = 2.0*xscaled*u_kp1 - u_kp2 + m_c(k);
            // Update summation values for all but the last step
            if (k > 1) {
                u_kp2 = u_kp1; u_kp1 = u_k;
            }
        }
        return xscaled*u_k - u_kp1 + m_c(0);
    }
    /**
    * @brief Do a vectorized evaluation of the Chebyshev expansion with the inputs scaled in [xmin, xmax]
    * @param x A vectype of values in the domain [xmin,xmax]
    */
    vectype ChebyshevExpansion::y(const vectype &x) const {
        // Scale x linearly into the domain [-1, 1]
        const vectype xscaled = (2 * x.array() - (m_xmax + m_xmin)) / (m_xmax - m_xmin);
        // Then call the function that takes the scaled x values
        return y_recurrence_xscaled(xscaled);
    }
    /**
    * @brief Do a vectorized evaluation of the Chebyshev expansion with the input scaled in the domain [-1,1]
    * @param xscaled A vectype of values scaled to the domain [-1,1] (the domain of the Chebyshev basis functions)
    * @param y A vectype of values evaluated from the expansion
    *
    * By using vectorizable types like Eigen::MatrixXd, without
    * any additional work, "magical" vectorization is happening
    * under the hood, giving a significant speed improvement. From naive
    * testing, the increase was a factor of about 10x.
    */
    vectype ChebyshevExpansion::y_recurrence_xscaled(const vectype &xscaled) const {
        const std::size_t Norder = m_c.size() - 1;

        Eigen::MatrixXd A(xscaled.size(), Norder + 1);
        
        if (Norder == 0) { return m_c[0]*Eigen::MatrixXd::Ones(A.rows(), A.cols()); }
        if (Norder == 1) { return m_c[0] + m_c[1]*xscaled.array(); }

        // Use the recurrence relationships to evaluate the Chebyshev expansion
        // In this case we do column-wise evaluations of the recurrence rule
        A.col(0).fill(1);
        A.col(1) = xscaled;
        for (int n = 1; n < Norder; ++n) {
            A.col(n + 1).array() = 2 * xscaled.array()*A.col(n).array() - A.col(n - 1).array();
        }
        // In this form, the matrix-vector product will yield the y values
        return A*m_c;
    }
    vectype ChebyshevExpansion::y_Clenshaw_xscaled(const vectype &xscaled) const {
        const std::size_t Norder = m_c.size() - 1;
        vectype u_k, u_kp1(xscaled.size()), u_kp2(xscaled.size());
        u_kp1.fill(m_c[Norder]); u_kp2.fill(0);
        for (int k = Norder - 1; k >= 1; --k) {
            u_k = 2 * xscaled.array()*u_kp1.array() - u_kp2.array() + m_c(k);
            // Update summation values for all but the last step
            if (k > 1) {
                u_kp2 = u_kp1; u_kp1 = u_k;
            }
        }
        return xscaled.array()*u_k.array() - u_kp1.array() + m_c(0);
    }

    Eigen::MatrixXd ChebyshevExpansion::companion_matrix() const {
        std::size_t Norder = m_c.size() - 1;
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(Norder, Norder);  
        // c_wrap wraps the first 0...Norder elements of the coefficient vector
        Eigen::Map<const Eigen::VectorXd> c_wrap(&(m_c[0]), Norder);
        // First row
        A(0, 1) = 1;
        // Last row
        A.row(Norder - 1) = -c_wrap / (2.0*m_c(Norder)); 
        A(Norder - 1, Norder - 2) += 0.5;
        // All the other rows
        for (int j = 1; j < Norder - 1; ++j) {
            A(j, j - 1) = 0.5;
            A(j, j + 1) = 0.5;
        }
        return A;
    }
    std::vector<double> ChebyshevExpansion::real_roots(bool only_in_domain) const {
        std::vector<double> roots;

        // Roots of the Chebyshev expansion are eigenvalues of the companion matrix
        // obtained from the companion_matrix function
        Eigen::MatrixXd Abalanced, D;
        balance_matrix(companion_matrix(), Abalanced, D);
        Eigen::VectorXcd eigvals = Abalanced.eigenvalues();

        for (int i = 0; i < eigvals.size(); ++i) {
            double imag = eigvals(i).imag();
            double real = eigvals(i).real();
            if (std::abs(imag)/std::abs(real) < 10 * DBL_EPSILON) {
                // Rescale back into real-world values
                double x = ((m_xmax - m_xmin)*real + (m_xmax + m_xmin)) / 2.0;
                // Keep it if in the domain or if you want all real roots
                if (!only_in_domain || (x <= m_xmax && x >= m_xmin)) {
                    roots.push_back(x);
                }
            }
        }
        return roots;
    }
    std::vector<ChebyshevExpansion> ChebyshevExpansion::subdivide(std::size_t Nintervals, std::size_t Norder) const {

        if (Nintervals == 1) {
            return std::vector<ChebyshevExpansion>(1, *this);
        }

        std::vector<ChebyshevExpansion> segments;
        double deltax = (m_xmax - m_xmin) / (Nintervals - 1);

        // Vector of values in the range [-1,1] as roots of a high-order Chebyshev 
        Eigen::VectorXd xpts_n11 = (Eigen::VectorXd::LinSpaced(Norder + 1, 0, Norder)*EIGEN_PI / Norder).array().cos();

        for (std::size_t i = 0; i < Nintervals - 1; ++i) {
            double xmin = m_xmin + i*deltax, xmax = m_xmin + (i + 1)*deltax;
            Eigen::VectorXd xrealworld = ((xmax - xmin)*xpts_n11.array() + (xmax + xmin)) / 2.0;
            segments.push_back(factoryf(Norder, y(xrealworld), xmin, xmax));
        }
        return segments;
    }
    std::vector<double> ChebyshevExpansion::real_roots_intervals(const std::vector<ChebyshevExpansion> &segments, bool only_in_domain) {
        std::vector<double> roots;
        for (auto &seg : segments) {
            const auto segroots = seg.real_roots(only_in_domain);
            roots.insert(roots.end(), segroots.cbegin(), segroots.cend());
        }
        return roots;
    }
    std::vector<double> ChebyshevExpansion::real_roots_approx(long Npoints)
    {
        std::vector<double> roots;
        // Vector of values in the range [-1,1] as roots of a high-order Chebyshev 
        Eigen::VectorXd xpts_n11 = (Eigen::VectorXd::LinSpaced(Npoints + 1, 0, Npoints)*EIGEN_PI / Npoints).array().cos();
        // Scale values into real-world values
        Eigen::VectorXd ypts = y_recurrence_xscaled(xpts_n11);
        // Eigen::MatrixXd buf(Npoints+1, 2); buf.col(0) = xpts; buf.col(1) = ypts; std::cout << buf << std::endl;
        for (size_t i = 0; i < Npoints - 1; ++i) {
            // The change of sign guarantees at least one root between indices i and i+1
            double y1 = ypts(i), y2 = ypts(i + 1);
            bool signchange = (std::signbit(y1) != std::signbit(y2));
            if (signchange) {
                double xscaled = xpts_n11(i);

                // Fit a quadratic given three points; i and i+1 bracket the root, so need one more constraint
                // i0 is the leftmost of the three indices that will be used; when i == 0, use 
                // indices i,i+1,i+2, otherwise i-1,i,i+1
                size_t i0 = (i >= 1) ? i - 1 : i;
                Eigen::Vector3d r;
                r << ypts(i0), ypts(i0 + 1), ypts(i0 + 2);
                Eigen::Matrix3d A;
                for (std::size_t irow = 0; irow < 3; ++irow) {
                    double _x = xpts_n11(i0 + irow);
                    A.row(irow) << _x*_x, _x, 1;
                }
                // abc holds the coefficients a,b,c for y = a*x^2 + b*x + c
                Eigen::VectorXd abc = A.colPivHouseholderQr().solve(r);
                double a = abc[0], b = abc[1], c = abc[2];

                // Solve the quadratic and find the root you want
                double x1 = (-b + sqrt(b*b - 4 * a*c)) / (2 * a);
                double x2 = (-b - sqrt(b*b - 4 * a*c)) / (2 * a);
                bool x1_in_range = is_in_closed_range(xpts_n11(i), xpts_n11(i + 1), x1);
                bool x2_in_range = is_in_closed_range(xpts_n11(i), xpts_n11(i + 1), x2);

                // Double check that only one root is within the range
                if (x1_in_range && !x2_in_range) {
                    xscaled = x1;
                }
                else if (x2_in_range && !x1_in_range) {
                    xscaled = x2;
                }
                else {
                    xscaled = 1e99;
                }

                // Rescale back into real-world values
                double x = ((m_xmax - m_xmin)*xscaled + (m_xmax + m_xmin)) / 2.0;
                roots.push_back(x);
            }
            else {
                // TODO: locate other roots based on derivative considerations
            }
        }
        return roots;
    }

    /// Chebyshev-Lobatto nodes cos(pi*j/N), j = 0,..., N
    Eigen::VectorXd ChebyshevExpansion::get_nodes_n11() {
        std::size_t N = m_c.size()-1;
        return extrema_library.get(N);
    }
    /// Values of the function at the Chebyshev-Lobatto nodes 
    Eigen::VectorXd ChebyshevExpansion::get_node_function_values() {
        std::size_t N = m_c.size()-1; 
        return u_matrix_library.get(N)*m_c;
    }

    ChebyshevExpansion ChebyshevExpansion::factoryf(const int N, const Eigen::VectorXd &f, const double xmin, const double xmax) {
        // Step 3: Get coefficients for the L matrix from the library of coefficients
        const Eigen::MatrixXd &L = l_matrix_library.get(N);
        // Step 4: Obtain coefficients from vector - matrix product
        return ChebyshevExpansion(L*f, xmin, xmax);
    }
    ChebyshevExpansion ChebyshevExpansion::from_powxn(const std::size_t n, const double xmin, const double xmax) {
        Eigen::VectorXd c = Eigen::VectorXd::Zero(n + 1);
        for (std::size_t k = 0; k <= n / 2; ++k) {
            std::size_t index = n - 2 * k;
            double coeff = binomialCoefficient(static_cast<double>(n), static_cast<double>(k));
            if (index == 0) {
                coeff /= 2.0;
            }
            c(index) = coeff;
        }
        return pow(2, 1-static_cast<int>(n))*ChebyshevExpansion(c, xmin, xmax);
    }
    ChebyshevExpansion ChebyshevExpansion::deriv(std::size_t Nderiv) const {
        // See Mason and Handscomb, p. 34, Eq. 2.52
        // and example in https ://github.com/numpy/numpy/blob/master/numpy/polynomial/chebyshev.py#L868-L964
        vectype c = m_c;
        for (std::size_t deriv_counter = 0; deriv_counter < Nderiv; ++deriv_counter) {
            std::size_t N = c.size() - 1, Nd = N - 1;
            vectype cd(N);
            for (std::size_t r = 0; r < Nd + 1; ++r) {
                cd(r) = 0;
                for (std::size_t k = r + 1; k < Nd + 2; ++k) {
                    if ((k - r) % 2 == 1) {
                        cd(r) += 2*k*c(k);
                    }
                }
                if (r == 0) {
                    cd(r) /= 2;
                }
                // Rescale the values if the range is not [-1,1]
                cd(r) /= (m_xmax-m_xmin)/2.0;
            }
            if (Nderiv == 1) {
                return ChebyshevExpansion(std::move(cd), m_xmin, m_xmax);
            }
            else{
                c = cd;
            }
        }
        return ChebyshevExpansion(std::move(c), m_xmin, m_xmax);
    };
    
    /// Once you specify which variable will be given, you can build the independent variable matrix
    void ChebyshevSummation::build_independent_matrix() {
        if (matrix_indep_built){ return; } // no-op if matrix already built
        /// C is a matrix with as many rows as terms in the summation, and the coefficients for each term in increasing order in each row
        Eigen::Index Nrows = 0, Ncols = terms.size();
        // Determine how many rows are needed 
        for (auto &term : terms) { Nrows = (F_SPECIFIED) ? std::max(Nrows, term.G.coef().size()) : std::max(Nrows, term.F.coef().size()); }
        // Fill matrix with all zeros (padding for shorter entries)
        C = Eigen::MatrixXd::Zero(Nrows, Ncols);
        B = Eigen::MatrixXd::Zero(Nrows, Ncols);
        N.resize(Ncols);
        // Fill each column
        Eigen::Index i = 0;
        for (auto &term : terms) {
            // Get the appropriate set of coefficients (expansion in delta (G) when tau is given, or expansion in tau (F) when delta is given)
            const Eigen::VectorXd &coef = (F_SPECIFIED) ? term.G.coef() : term.F.coef();
            C.col(i).head(coef.size()) = coef;
            i++;
        }
        matrix_indep_built = true;
    };
    /// Once you specify which variable will be given, you can build the independent variable matrix
    void ChebyshevSummation::build_dependent_matrix() {
        if (matrix_dep_built) { return; } // no-op if matrix already built
                                      /// C is a matrix with as many rows as terms in the summation, and the coefficients for each term in increasing order in each row
        Eigen::Index Nrows = 0, Ncols = terms.size();
        // Determine how many rows are needed 
        for (auto &term : terms) { Nrows = (!F_SPECIFIED) ? std::max(Nrows, term.G.coef().size()) : std::max(Nrows, term.F.coef().size()); }
        // Fill matrix with all zeros (padding for shorter entries)
        B = Eigen::MatrixXd::Zero(Nrows, Ncols);
        N.resize(Ncols);
        // Fill each column
        Eigen::Index i = 0;
        for (auto &term : terms) {
            const Eigen::VectorXd &coef = (F_SPECIFIED) ? term.F.coef() : term.G.coef();
            B.col(i).head(coef.size()) = coef;
            N(i) = term.n_i;
            i++;
        }
        matrix_dep_built = true;
    };
    Eigen::VectorXd ChebyshevSummation::get_nFcoefficients_parallel(double input) {
        build_independent_matrix();
        build_dependent_matrix();
        Eigen::Index Ncols = 0;
        // Determine how many rows are needed 
        for (auto &term : terms) { Ncols = (!F_SPECIFIED) ? std::max(Ncols, term.G.coef().size()) : std::max(Ncols, term.F.coef().size()); }

        Eigen::RowVectorXd A(Ncols);
        double inmin = (F_SPECIFIED) ? terms[0].F.xmin() : terms[0].G.xmin();
        double inmax = (F_SPECIFIED) ? terms[0].F.xmax() : terms[0].G.xmax();
        double xscaled = (2 * input - (inmax + inmin)) / (inmax - inmin);

        // Use the recurrence relationships to evaluate the Chebyshev expansion
        // In this case we do column-wise evaluations of the recurrence rule
        A(0) = 1;
        A(1) = xscaled;
        for (Eigen::Index n = 1; n < Ncols-1; ++n) {
            A(n + 1) = 2*xscaled*A(n) - A(n - 1);
        }
        Eigen::VectorXd AB = A*B;
        Eigen::VectorXd o = (AB.array())*(N.array());
        return o;
    }
    Eigen::VectorXd ChebyshevSummation::get_nFcoefficients_serial(double input) {
        build_independent_matrix();
        build_dependent_matrix();
        // For the specified one, evaluate its Chebyshev expansion
        givenvec.resize(terms.size());
        std::size_t i = 0;
        for (const auto &term : terms) {
            if (F_SPECIFIED) {
                givenvec(i) = N(i)*term.F.y_Clenshaw(input);
            }
            else {
                throw - 1;
            }
            i++;
        }
        return givenvec;
    }
    Eigen::VectorXd ChebyshevSummation::get_coefficients(double input) {
        build_independent_matrix();
        build_dependent_matrix();
        return C*get_nFcoefficients_parallel(input);
    };

    ChebyshevExpansion ChebyshevMixture::get_expansion_of_interval(std::vector<ChebyshevSummation> &interval, double tau, const Eigen::VectorXd &z, double xmin, double xmax) {
        if (all_same_order){
            // For this interval, calculate the contributions for each fluid to the expansion

            //if (std::abs(tau - previous_tau) > 1e-14) 
            {
                // If they are all the same order, can skip the copy+pad, and just write the column directly
                for (std::size_t icomp = 0; icomp < interval.size(); ++icomp) {
                    Eigen::VectorXd c = interval[icomp].get_coefficients(tau);
                    A.col(icomp).head(c.size()) = c;
                }
                previous_tau = tau;
            }
            
            // Return an expansion for this interval in terms of delta of the mixture
            return ChebyshevExpansion(A*z, xmin, xmax);
        }
        else {
            // TODO: not implemented for now
            throw -1;
        }
    }
    std::vector<Eigen::MatrixXd> ChebyshevMixture::calc_companion_matrices(double rhorRT, double p_target, double tau, const Eigen::VectorXd &z) {
        std::vector<Eigen::MatrixXd> mats;
        // Iterate over the intervals forming the domain; in each, find any roots that you can
        for (auto &interval : interval_expansions) {
            auto p = rhorRT*(get_expansion_of_interval(interval, tau, z, interval[0].xmin(), interval[0].xmax()).times_x() + 1).times_x();
            auto expansion = p - p_target;
            mats.push_back(expansion.companion_matrix());
        }
        return mats;
    }
    bool ChebyshevMixture::unlikely_root(ChebyshevExpansion &pdiff, double ptolerance)
    {
        Eigen::VectorXd nodevals = pdiff.get_node_function_values();
        double nodemin = nodevals.minCoeff(), nodemax = nodevals.maxCoeff(), nodeminabs = nodevals.cwiseAbs().minCoeff();
        double xminval = pdiff.y_Clenshaw(pdiff.xmin()), xmaxval = pdiff.y_Clenshaw(pdiff.xmax());
        if (nodemin*nodemax > 0.0 && xminval*xmaxval > 0.0 && nodeminabs > ptolerance) {
            return true;
        }
        else {
            return false;
        }
    }
    void ChebyshevMixture::calc_real_roots(double rhorRT, double p_target, double tau, const Eigen::VectorXd &z, double ptolerance) {
        m_roots.clear();
        // Iterate over the intervals forming the domain; in each, find any roots that you can
        for (auto &interval : interval_expansions) {
            auto expansion = rhorRT*(get_expansion_of_interval(interval, tau, z, interval[0].xmin(), interval[0].xmax()).times_x() + 1).times_x() - p_target;
            if (unlikely_root(expansion, ptolerance)) {
                continue;
            }
            const bool only_in_domain = true;
            std::vector<double> roots = expansion.real_roots(only_in_domain);
            m_roots.insert(m_roots.end(), roots.begin(), roots.end());
        }
    }
    double ChebyshevMixture::time_calc_real_roots(double rhorRT, double p_target, double tau, const Eigen::VectorXd &z, double ptolerance) {
        auto startTime = std::chrono::system_clock::now();
        double N = 10;
        double summer = 0;
        for (int i = 0; i < N; ++i) {
            calc_real_roots(rhorRT, p_target, tau, z, ptolerance);
        }
        auto endTime = std::chrono::system_clock::now();
        auto elap_us = std::chrono::duration<double>(endTime - startTime).count() / N*1e6;
        return elap_us;
    }
    std::vector<double> ChebyshevMixture::get_real_roots(){ return m_roots; }
    ChebyshevExpansion ChebyshevMixture::get_p(std::vector<ChebyshevSummation> &interval,  double rhorRT, double tau, const Eigen::VectorXd &z) {
        return rhorRT*(get_expansion_of_interval(interval, tau, z, interval[0].xmin(), interval[0].xmax()).times_x() + 1).times_x();
    }
    ChebyshevExpansion ChebyshevMixture::get_dalphar_ddelta(std::size_t i, double rhorRT, double tau, const Eigen::VectorXd &z) {
        if (i > 1000){
            Eigen::VectorXd c(2,1);
            c << 0,1;
            return ChebyshevExpansion(c, -1, 1);
        }
        else{
            std::vector<ChebyshevSummation> &interval = interval_expansions[i];
            return get_expansion_of_interval(interval, tau, z, interval[0].xmin(), interval[0].xmax());
        }
    }

    double ChebyshevMixture::time_get(std::string &thing, double rhorRT, double tau, double p, const Eigen::VectorXd &z) {
        auto startTime = std::chrono::system_clock::now();
        double N = 10000;
        double summer = 0;
        if (thing == "p"){
            for (int i = 0; i < N; ++i) {
                for (std::size_t j = 0; j < interval_expansions.size(); ++j) {
                    std::vector<ChebyshevSummation> &interval = interval_expansions[j];
                    auto pp = rhorRT*(get_expansion_of_interval(interval, tau, z, interval[0].xmin(), interval[0].xmax()).times_x() + 1).times_x() - p;
                    summer += pp.coef()[0];
                }
            }
        }
        else if (thing == "nF-serial"){
            for (int i = 0; i < N; ++i) {
                for (std::size_t j = 0; j < interval_expansions.size(); ++j) {
                    std::vector<ChebyshevSummation> &interval = interval_expansions[j];
                    for (auto &fluid : interval){
                        summer += fluid.get_nFcoefficients_serial(tau)(5);
                    }
                }
            }
        }
        else if (thing == "nF-parallel") {
            for (int i = 0; i < N; ++i) {
                for (std::size_t j = 0; j < interval_expansions.size(); ++j) {
                    std::vector<ChebyshevSummation> &interval = interval_expansions[j];
                    for (auto &fluid : interval) {
                        summer += fluid.get_nFcoefficients_parallel(tau)(5);
                    }
                }
            }
        }
        else {
            throw std::range_error("Invalid thing:"+thing);
        }
        auto endTime = std::chrono::system_clock::now();
        auto elap_us = std::chrono::duration<double>(endTime - startTime).count() / N*1e6 + sin(summer)/1e15;
        return elap_us;
    }
    
    Eigen::VectorXcd ChebyshevMixture::eigenvalues(Eigen::MatrixXd &A, bool balance) {
        if (balance) {
            Eigen::MatrixXd Abalanced, D;
            balance_matrix(A, Abalanced, D);
            return Abalanced.eigenvalues();
        }
        else {
            return A.eigenvalues();
        }
    }

}; /* namespace Chebtools */
