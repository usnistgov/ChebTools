#ifndef CHEBTOOLS_H
#define CHEBTOOLS_H

#include "Eigen/Dense"
#include <vector>

namespace ChebTools{

    typedef Eigen::VectorXd vectype;
    
    /// Get the Chebyshev-Lobatto nodes for an expansion of degree \f$N\f$
    const Eigen::VectorXd &get_CLnodes(std::size_t N);

    Eigen::VectorXcd eigenvalues(const Eigen::MatrixXd &A, bool balance);
    Eigen::VectorXd eigenvalues_upperHessenberg(const Eigen::MatrixXd &A, bool balance);

    /**
    * @brief This is the main underlying object that makes all of the code of ChebTools work.
    *
    * This class has accessor methods for getting things from the object, and static factory
    * functions for generating new expansions.  It also has methods for calculating derivatives,
    * roots, etc.
    */
    class ChebyshevExpansion {
    private:
        vectype m_c;
        double m_xmin, m_xmax;

        vectype m_recurrence_buffer;
        Eigen::MatrixXd m_recurrence_buffer_matrix;
        void resize() {
            m_recurrence_buffer.resize(m_c.size());
        }

        //reduce_zeros changes the m_c field so that our companion matrix doesnt have nan values in it
        //all this does is truncate m_c such that there are no trailing zero values
        static Eigen::VectorXd reduce_zeros(const Eigen:: VectorXd &chebCoeffs){
          //these give us a threshold for what coefficients are large enough
          double largeTerm = 1e-15;
          if (chebCoeffs.size()>=1 && std::abs(chebCoeffs(0))>largeTerm){
            largeTerm = chebCoeffs(0);
          }
          //if the second coefficient is larger than the first, then make our tolerance
          //based on the second coefficient, this is useful for functions whose mean value
          //is zero on the interval
          if (chebCoeffs.size()>=2 && std::abs(chebCoeffs(1))>largeTerm){
            largeTerm = chebCoeffs(1);
          }
          double tol = largeTerm*(1e-15);
          int neededSize = static_cast<int>(chebCoeffs.size());
          //loop over m_c backwards, if we run into large enough coefficient, then record the size and break
          for (int i=static_cast<int>(chebCoeffs.size())-1; i>=0; i--){
            if (std::abs(chebCoeffs(i))>tol){
              neededSize = i+1;
              break;
            }
            neededSize--;
          }
          //neededSize gives us the number of coefficients that are nonzero
          //we will resize m_c such that there are essentially no trailing zeros
          return chebCoeffs.head(neededSize);
        }

    public:
        /// Initializer with coefficients, and optionally a range provided
        ChebyshevExpansion(const vectype &c, double xmin = -1, double xmax = 1) : m_c(c), m_xmin(xmin), m_xmax(xmax) { resize(); };
        /// Initializer with coefficients, and optionally a range provided
        ChebyshevExpansion(const std::vector<double> &c, double xmin = -1, double xmax = 1) : m_xmin(xmin), m_xmax(xmax) {
            m_c = Eigen::Map<const Eigen::VectorXd>(&(c[0]), c.size());
            resize();
        };
        /// Move constructor (C++11 only)
        ChebyshevExpansion(const vectype &&c, double xmin = -1, double xmax = 1) : m_c(c), m_xmin(xmin), m_xmax(xmax) { resize(); };
        /// Get the minimum value of \f$x\f$ for the expansion
        double xmin(){ return m_xmin; }
        /// Get the maximum value of \f$x\f$ for the expansion
        double xmax(){ return m_xmax; }

        /// Get the vector of coefficients in increasing order
        const vectype &coef() const;

        /// Return the N-th derivative of this expansion, where N must be >= 1
        ChebyshevExpansion deriv(std::size_t Nderiv) const;
        /// Get the Chebyshev-Lobatto nodes in the domain [-1,1]
        Eigen::VectorXd get_nodes_n11();
        /// Get the Chebyshev-Lobatto nodes in the domain [xmin, xmax]
        Eigen::VectorXd get_nodes_realworld();
        /// Values of the function at the Chebyshev-Lobatto nodes
        Eigen::VectorXd get_node_function_values();

        // ******************************************************************
        // ***********************      OPERATORS     ***********************
        // ******************************************************************

        /// A ChebyshevExpansion plus another ChebyshevExpansion yields a new ChebyheveExpansion
        ChebyshevExpansion operator+(const ChebyshevExpansion &ce2) const ;
        /** 
        * @brief An inplace addition of two expansions
        * @note The lower degree one is right-padded with zeros to have the same degree as the higher degree one
        * @param donor The other expansion in the summation
        */
        ChebyshevExpansion& operator+=(const ChebyshevExpansion &donor);
        /// Multiplication of an expansion by a constant
        ChebyshevExpansion operator*(double value) const;
        /// Addition of a constant to an expansion
        ChebyshevExpansion operator+(double value) const;
        /// Subtraction of a constant from an expansion
        ChebyshevExpansion operator-(double value) const;
        /// An inplace multiplication of an expansion by a constant
        ChebyshevExpansion& operator*=(double value);
        /// An inplace addition of a constant to an expansion
        ChebyshevExpansion& operator+=(double value);
        /// An inplace subtraction of a constant from an expansion
        ChebyshevExpansion& operator-=(double value);
        /**
         * @brief Multiply two Chebyshev expansions together; thanks to Julia code from Bradley Alpert, NIST
         *
         * Converts padded expansions to nodal functional values, functional values are multiplied together,
         * and then inverse transformation is used to return to coefficients of the product
         * @param ce2 The other expansion
         */
        ChebyshevExpansion operator*(const ChebyshevExpansion &ce2) const;
        /**
         * @brief Multiply a Chebyshev expansion by its independent variable \f$x\f$
         */
        ChebyshevExpansion times_x() const;

        /** 
         * @brief Multiply a Chebyshev expansion by its independent variable \f$x\f$ in-place
         *
         * This operation is carried out in-place to minimize the amount of memory re-allocation
         * which proved during profiling to be a major source of inefficiency
         */
        ChebyshevExpansion& times_x_inplace();

        /// Friend function that allows for pre-multiplication by a constant value
        friend ChebyshevExpansion operator*(double value, const ChebyshevExpansion &ce){
            return ChebyshevExpansion(std::move(ce.coef()*value),ce.m_xmin, ce.m_xmax);
        };
        
        /**
         * @brief Apply a function to the expansion
         *
         * This function first converts the expansion to functional values at the 
         * Chebyshev-Lobatto nodes, applies the function to the nodal values, and then
         * does the inverse transformation to arrive at the coefficients of the expansion
         * after applying the transformation
         */
        ChebyshevExpansion apply(std::function<Eigen::ArrayXd(const Eigen::ArrayXd &)> &f);

        // ******************************************************************
        // **********************      EVALUATORS     ***********************
        // ******************************************************************

        /**
        * @brief Do a single input/single output evaluation of the Chebyshev expansion with the inputs scaled in [xmin, xmax]
        * @param x A value scaled in the domain [xmin,xmax]
        */
        double y_recurrence(const double x);
        /**
        * @brief Do a single input/single output evaluation of the Chebyshev expansion with the inputs scaled in [xmin, xmax]
        * @param x A value scaled in the domain [xmin,xmax]
        */
        double y_Clenshaw(const double x) const;
        /**
        * @brief Do a single input/single output evaluation of the Chebyshev expansion with the inputs scaled in [-1,1]
        * @param x A value scaled in the domain [-1,1]
        */
        double y_Clenshaw_xscaled(const double x) const;
        /**
        * @brief Do a vectorized evaluation of the Chebyshev expansion with the inputs scaled in [xmin, xmax]
        * @param x A vectype of values in the domain [xmin,xmax]
        */
        vectype y(const vectype &x) const;
        /**
        * @brief Do a vectorized evaluation of the Chebyshev expansion with the inputs scaled in [xmin, xmax]
        * @param x A value scaled in the domain [xmin,xmax]
        */
        double y(const double x) const{ return y_Clenshaw(x); }
        /**
        * @brief Do a vectorized evaluation of the Chebyshev expansion with the input scaled in the domain [-1,1]
        * @param xscaled A vectype of values scaled to the domain [-1,1] (the domain of the Chebyshev basis functions)
        * @returns y A vectype of values evaluated from the expansion
        *
        * By using vectorizable types like Eigen::MatrixXd, without
        * any additional work, "magical" vectorization is happening
        * under the hood, giving a significant speed improvement. From naive
        * testing, the increase was a factor of about 10x.
        */
        vectype y_recurrence_xscaled(const vectype &xscaled) const ;
        /**
        * @brief Do a vectorized evaluation of the Chebyshev expansion with the input scaled in the domain [-1,1] with Clenshaw's method
        * @param xscaled A vectype of values scaled to the domain [-1,1] (the domain of the Chebyshev basis functions)
        * @returns y A vectype of values evaluated from the expansion
        */
        vectype y_Clenshaw_xscaled(const vectype &xscaled) const ;

        /**
        * @brief Construct and return the companion matrix of the Chebyshev expansion
        * @returns A The companion matrix of the expansion
        *
        * See Boyd, SIAM review, 2013, http://dx.doi.org/10.1137/110838297, Appendix A.2
        */
        Eigen::MatrixXd companion_matrix(const Eigen::VectorXd &coeffs) const ;
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
        /**
        * @brief The second-generation rootfinder of ChebyshevExpansions
        * @param only_in_domain True: only keep roots that are in the domain of the expansion. False: all real roots
        */
        std::vector<double> real_roots2(bool only_in_domain = true) const;
        /**
        * @brief Subdivide the original interval into a set of subintervals that are linearly spaced
        * @note A vector of ChebyshevExpansions are returned
        * @param Nintervals The number of intervals
        * @param Ndegree The degree of the Chebyshev expansion in each interval
        */
        std::vector<ChebyshevExpansion> subdivide(std::size_t Nintervals, std::size_t Ndegree) const ;

        /**
        * @brief For a vector of ChebyshevExpansions, find all roots in each interval
        * @param segments The vector of ChebyshevExpansions
        * @param only_in_domain True: only keep roots that are in the domain of the expansion. False: all real roots
        */
        static std::vector<double> real_roots_intervals(const std::vector<ChebyshevExpansion> &segments, bool only_in_domain = true);

        /**
        * @brief Time how long (in seconds) it takes to evaluate the roots
        * @param N How many repeats to do (maybe a million?  It's pretty fast for small degrees)
        */
        double real_roots_time(long N);

        /// A DEPRECATED function for approximating the roots (do not use)
        std::vector<double> real_roots_approx(long Npoints);

        // ******************************************************************
        // ***********************      BUILDERS      ***********************
        // ******************************************************************

        /**
        * @brief Given a set of values at the Chebyshev-Lobatto nodes, perhaps obtained from the ChebyshevExpansion::factory function, 
        * get the expansion
        *
        * @param N The degree of the expansion
        * @param f The set of values at the Chebyshev-Lobatto nodes
        * @param xmin The minimum value of x for the expansion
        * @param xmax The maximum value of x for the expansion
        */
        static ChebyshevExpansion factoryf(const std::size_t N, const Eigen::VectorXd &f, const double xmin, const double xmax) ;

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
        static ChebyshevExpansion factory(const std::size_t N, double_function func, const double xmin, const double xmax)
        {
            // Get the precalculated Chebyshev-Lobatto nodes
            const Eigen::VectorXd & x_nodes_n11 = get_CLnodes(N);

            // Step 1&2: Grid points functional values (function evaluated at the
            // extrema of the Chebyshev polynomial of order N - there are N+1 of them)
            Eigen::VectorXd f(N + 1);
            for (int k = 0; k <= N; ++k) {
                // The extrema in [-1,1] scaled to real-world coordinates
                double x_k = ((xmax - xmin)*x_nodes_n11(k) + (xmax + xmin)) / 2.0;
                f(k) = func(x_k);
            }
            return factoryf(N, f, xmin, xmax);
        };

        /// Convert a monomial term in the form \f$x^n\f$ to a Chebyshev expansion
        static ChebyshevExpansion from_powxn(const std::size_t n, const double xmin, const double xmax);

        /** 
        * @brief Convert a polynomial expansion in monomial form to a Chebyshev expansion
        *
        * The monomial expansion is of the form \f$ y = \displaystyle\sum_{i=0}^N c_ix_i\f$
        *
        * This transformation can be carried out analytically.  For convenience we repetitively use
        * calls to ChebyshevExpansion::from_powxn to build up the expansion.  This is probably not
        * the most efficient option, but it works.
        *
        * @param c The vector of coefficients of the monomial expansion in *increasing* degree: 
        * @param xmin The minimum value of \f$x\f$ for the expansion
        * @param xmax The maximum value of \f$x\f$ for the expansion
        */
        template<class vector_type>
        static ChebyshevExpansion from_polynomial(vector_type c, const double xmin, const double xmax) {
            vectype c0(1); c0 << 0;
            ChebyshevExpansion s(c0, xmin, xmax);
            for (std::size_t i = 0; i < static_cast<std::size_t>(c.size()); ++i) {
                s += c(i)*from_powxn(i, xmin, xmax);
            }
            return s;
        }
    };

}; /* namespace ChebTools */
#endif
