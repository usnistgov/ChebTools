#ifndef CHEBTOOLS_H
#define CHEBTOOLS_H

#include "Eigen/Dense"
#include <vector>

namespace ChebTools{

    typedef Eigen::VectorXd vectype;

    const Eigen::VectorXd &get_extrema(std::size_t N);

    class ChebyshevExpansion {
    private:
        vectype m_c;
        double m_xmin, m_xmax;

        vectype m_recurrence_buffer;
        Eigen::MatrixXd m_recurrence_buffer_matrix;
        void resize() {
            m_recurrence_buffer.resize(m_c.size());
        }

    public:

        ChebyshevExpansion(const vectype &c, double xmin = -1, double xmax = 1) : m_c(c), m_xmin(xmin), m_xmax(xmax) { resize(); };
        ChebyshevExpansion(const std::vector<double> &c, double xmin = -1, double xmax = 1) : m_xmin(xmin), m_xmax(xmax) {
            m_c = Eigen::Map<const Eigen::VectorXd>(&(c[0]), c.size());
            resize();
        };

        // Move constructor (C++11 only)
        ChebyshevExpansion(const vectype &&c, double xmin = -1, double xmax = 1) : m_c(c), m_xmin(xmin), m_xmax(xmax) { resize(); };

        ChebyshevExpansion operator+(const ChebyshevExpansion &ce2) const ;
        ChebyshevExpansion& operator+=(const ChebyshevExpansion &donor);
        ChebyshevExpansion operator*(double value) const ;
        ChebyshevExpansion& operator*=(double value);
        // Multiply two Chebyshev expansions together; thanks to Julia code from Bradley Alpert, NIST
        ChebyshevExpansion operator*(const ChebyshevExpansion &ce2) const;

        /// Friend function that allows for pre-multiplication by a constant value
        friend ChebyshevExpansion operator*(double value, const ChebyshevExpansion &ce){
            return ChebyshevExpansion(std::move(ce.coef()*value));
        };

        const vectype &coef() const ;
        /**
        * @brief Do a single input/single output evaluation of the Chebyshev expansion with the inputs scaled in [xmin, xmax]
        * @param x A value scaled in the domain [xmin,xmax]
        */
        double y_recurrence(const double x);
        double y_Clenshaw(const double x) const;
        /**
        * @brief Do a vectorized evaluation of the Chebyshev expansion with the inputs scaled in [xmin, xmax]
        * @param x A vectype of values in the domain [xmin,xmax]
        */
        vectype y(const vectype &x) const ;
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
        vectype y_recurrence_xscaled(const vectype &xscaled) const ;
        vectype y_Clenshaw_xscaled(const vectype &xscaled) const ;

        /**
        * @brief Construct and return the companion matrix of the Chebyshev expansion
        * @returns A The companion matrix of the expansion
        *
        * See Boyd, SIAM review, 2013, http://dx.doi.org/10.1137/110838297, Appendix A.2
        */
        Eigen::MatrixXd companion_matrix() const ;
        /**
        * @brief Return the real roots of the Chebyshev expansion
        * @param only_in_domain If true, only real roots that are within the domain
        *                       of the expansion will be returned, otherwise all real roots
        *
        * The roots are obtained based on the fact that the eigenvalues of the
        * companion matrix are the roots of the Chebyshev expansion.  Thus
        * this function is relatively slow, because an eigenvalue solve is required,
        * which takes O(n^3) FLOPs.  But it is numerically rather reliable.
        *
        * As the order of the expansion increases, the eigenvalue solver in Eigen becomes
        * progressively less and less able to obtain the roots properly. The eigenvalue
        * solver in numpy tends to be more reliable.
        */
        std::vector<double> real_roots(bool only_in_domain = true) const ;
        std::vector<ChebyshevExpansion> subdivide(std::size_t Nintervals, std::size_t Norder) const ;
        std::vector<double> real_roots_intervals(const std::vector<ChebyshevExpansion> &segments, bool only_in_domain = true) const ;

        double real_roots_time(long N);
        std::vector<double> real_roots_approx(long Npoints);

        //std::string toString() const {
        //    return "[" + std::to_string(x) + ", " + std::to_string(y) + "]";
        //}

        static ChebyshevExpansion factoryf(const int N, const Eigen::VectorXd &f, const double xmin, const double xmax) ;

        /**
        * @brief Given a callable function, construct the N-th order Chebyshev expansion in [xmin, xmax]
        * @param N The order of the expansion; there will be N+1 coefficients
        * @param func A callable object, taking the x value (in [xmin,xmax]) and returning the y value
        * @param xmin The minimum x value for the fit
        * @param xmax The maximum x value for the fit
        *
        * See Boyd, SIAM review, 2013, http://dx.doi.org/10.1137/110838297, Appendix A.
        */
        template<class double_function>
        static ChebyshevExpansion factory(const int N, double_function func, const double xmin, const double xmax)
        {
            // Get the precalculated extrema values
            const Eigen::VectorXd & x_extrema_n11 = get_extrema(N); 

            // Step 1&2: Grid points functional values (function evaluated at the
            // extrema of the Chebyshev polynomial of order N - there are N+1 of them)
            Eigen::VectorXd f(N + 1);
            for (int k = 0; k <= N; ++k) {
                // The extrema in [-1,1] scaled to real-world coordinates
                double x_k = ((xmax - xmin)*x_extrema_n11(k) + (xmax + xmin)) / 2.0;
                f(k) = func(x_k);
            }
            return factoryf(N, f, xmin, xmax);
        };

        /// Convert a monomial term in the form \f$x^n\f$ to a Chebyshev expansion
        static ChebyshevExpansion from_powxn(const std::size_t n, const double xmin, const double xmax);

        template<class vector_type>
        static ChebyshevExpansion from_polynomial(vector_type c, const double xmin, const double xmax) {
            vectype c0(1); c0 << 0;
            ChebyshevExpansion s(c0, xmin, xmax);
            for (std::size_t i = 0; i < c.size(); ++i) {
                s += c(i)*from_powxn(i, xmin, xmax);
            }
            return s;
        }
        /// Return the N-th derivative of this expansion, where N must be >= 1
        ChebyshevExpansion deriv(std::size_t Nderiv) const ;
    };

    class SumElement {
    public:
        double n_i; ///< The leading coefficient
        ChebyshevExpansion F, ///< Expansion in terms of variable #1
            G; ///< Expansion in terms of variable #2
        SumElement(double n_i, ChebyshevExpansion &F, ChebyshevExpansion &G) :n_i(n_i), F(F), G(G) {};
    };

    class ChebyshevSummation {
    private:
        Eigen::MatrixXd C; ///< Coefficient matrix for the coefficients associated with each ChebyshevExpansion in the non-provided variable
        std::vector<SumElement> terms;
        bool F_SPECIFIED = true;
        bool matrix_built = false;
    public:
        ChebyshevSummation(const std::vector<SumElement> &terms) : terms(terms) {}; 
        ChebyshevSummation(const std::vector<SumElement> &&terms) : terms(terms) {};
        /// Once you specify which variable will be given, you can build the independent variable matrix
        void build_independent_matrix();
        Eigen::MatrixXd get_matrix(){ return C; }
        Eigen::VectorXd get_coefficients(double input);
    };

    class ChebyshevMixture {
    private:
        std::vector<ChebyshevSummation> contributions;
        Eigen::MatrixXd A;
        bool all_same_order = true;
        void allocate(short Norder){
            A.resize(Norder + 1, contributions.size());
        }
    public:
        ChebyshevMixture(const std::vector<ChebyshevSummation> &contributions, short Norder) : contributions(contributions) {allocate(Norder);};
        ChebyshevMixture(const std::vector<ChebyshevSummation> &&contributions, short Norder) : contributions(contributions) {allocate(Norder);};

        Eigen::MatrixXd get_A() { return A; }
        ChebyshevExpansion get_expansion(double tau, const Eigen::VectorXd &z);
    };

}; /* namespace ChebTools */
#endif