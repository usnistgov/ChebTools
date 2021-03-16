#pragma once

#include <Eigen/Dense>
#include <optional>
#include <iostream>

namespace ChebTools{

    /* Classic Clenshaw method, templated */
    template <typename VecType>
    double Clenshaw(double xscaled, const VecType& m_c) {
        // See https://en.wikipedia.org/wiki/Clenshaw_algorithm#Special_case_for_Chebyshev_series
        auto Norder = m_c.size() - 1;
        auto u_k = 0.0, u_kp1 = m_c[Norder], u_kp2 = 0.0;
        auto k = 0.0;
        for (int k = Norder - 1; k > 0; --k) {
            // Do the recurrent calculation
            u_k = 2.0 * xscaled * u_kp1 - u_kp2 + m_c[k];
            // Update the values
            u_kp2 = u_kp1; u_kp1 = u_k;
        }
        return m_c[0] + xscaled * u_kp1 - u_kp2;
    }

    /** Clenshaw method of Basu, templated
    * Needed because of the treatment of the leading coefficients, which is non-standard
    */
    template <typename NumType, typename VecType>
    double ClenshawBasu(NumType xscaled, const VecType& m_c) {
        // See https://en.wikipedia.org/wiki/Clenshaw_algorithm#Special_case_for_Chebyshev_series
        int Norder = static_cast<int>(m_c.size()) - 1;
        NumType u_k = 0.0, u_kp1 = 0.0, u_kp2 = 0.0;
        for (int k = Norder; k >= 0; --k) {
            // Do the recurrent calculation
            u_k = 2.0 * xscaled * u_kp1 - u_kp2 + m_c[k];
            if (k > 0){
                // Update the values
                u_kp2 = u_kp1; u_kp1 = u_k;
            }
        }
        return (u_k - u_kp2)/2.0;
    }

    /// Evaluation of a Chebyshev expansion (you probably shouldn't use this function except for testing)
    template<typename Integer, typename ValType>
    auto TCheb(const Integer &n, const ValType &x){
        if (n == 0) {
            return 1.0;
        }
        else if (n == 1) {
            return x;
        }
        else{
            return 2.0*x*TCheb(n-1, x) - TCheb(n-2, x);
        }
    }

    namespace TwoD{
       
        template<typename NumType>
        struct ChebyshevExpansion2DBounds{
            NumType xmin = -1, xmax = 1.0, ymin = -1.0, ymax = 1.0;
        };

        /// The 2D Chebyshev expansion over a rectangular domain
        template <typename CoeffMatType, int Rows = CoeffMatType::RowsAtCompileTime, int Cols = CoeffMatType::ColsAtCompileTime>
        class ChebyshevExpansion2D{
        private:
            
            using NumType = typename CoeffMatType::value_type;
            using ArrayType = Eigen::Array<NumType, Eigen::Dynamic, 1>; 

            const CoeffMatType mat;
            const ChebyshevExpansion2DBounds<NumType> bounds;
            const std::size_t Nx, Ny;
            bool _resize_required = false;
            
        public:
            auto get_mat(){ return mat; };
            auto get_bounds(){ return bounds; };

            ChebyshevExpansion2D(const CoeffMatType&mat, const ChebyshevExpansion2DBounds<NumType> &bounds)
                : mat(mat), bounds(bounds), Nx(mat.cols()-1), Ny(mat.rows()-1)
            {
                int rr =0 ;
            }
            /// Scale input x value from real-world into [-1,1]
            template <typename T>
            auto scalex(const T &x) const {
                return (2 * x - (bounds.xmax + bounds.xmin)) / (bounds.xmax - bounds.xmin);
            }
            /// Scale input x value in [-1,1] into real-world values
            template <typename T>
            auto unscalex(const T &xn11) const {    
                return (xn11 * (bounds.xmax - bounds.xmin) + (bounds.xmax + bounds.xmin)) / 2;
            }
            /// Scale input y value from real-world into [-1,1]
            template <typename T>
            auto scaley(const T &y) const {    
                return (2 * y - (bounds.ymax + bounds.ymin)) / (bounds.ymax - bounds.ymin);
            }
            /// Scale input y value in [-1,1] into real-world values
            template <typename T>
            auto unscaley(const T &yn11) const {
                return (yn11 * (bounds.ymax - bounds.ymin) + (bounds.ymax + bounds.ymin)) / 2;
            }

            /// Get the x locations in [-1,1] where the m+1 expansion crosses zero
            template <typename T>
            static auto get_xroots(T m) {
                return ((2 * ArrayType::LinSpaced(m + 1, 0, static_cast<NumType>(m)) + 1) * EIGEN_PI / (2 * (m + 1))).cos();
            }
            /// Get the y locations in [-1,1] where the n+1 expansion crosses zero
            template <typename T>
            static auto get_yroots(T n) {
                return ((2 * ArrayType::LinSpaced(n + 1, 0, static_cast<NumType>(n)) + 1) * EIGEN_PI / (2 * (n + 1))).cos();
            }

            /// Evaluate the output from nested evaluations of Clenshaw's method
            auto eval_Clenshaw(NumType x, NumType y) {
                auto xscaled = scalex(x), yscaled = scaley(y);
                auto m = Nx, n = Ny;
                ArrayType b; b.resize(m + 1);
                for (auto i = 0; i < b.size(); ++i) {
                    b[i] = ClenshawBasu(yscaled, mat.row(i));
                }
                return ClenshawBasu(xscaled, b);
            }

            // This naive implementation is a one-to-one translation of Eq. 11 of Basu, SIAM, 1973
            // It can be accelerated by moving the function evaluations at the tensor products of the roots 
            // into an outer matrix
            template<typename Integer, typename Function>
            static auto build_matrix_naive(Integer m, Integer n, Function f){

                ArrayType xroots = get_xroots(m), yroots = get_yroots(n);

                CoeffMatType a; a.resize(m+1, n+1); a.setZero();
                for (auto i = 0; i < m+1; ++i){
                    for (auto j = 0; j < n+1; ++j){
                        auto dsum = 0.0;
                        for (auto r = 0; r < m+1; ++r){
                            for (auto s = 0; s < n+1; ++s){
                                auto xr = xroots[r];
                                auto ys = yroots[s];
                                dsum += f(xr, ys)*TCheb(i, xr)*TCheb(j, ys);
                            }
                        }
                        a(i,j) = dsum;
                    }
                }
                CoeffMatType c = 4.0/((m+1.0)*(n+1.0))*a;
                return c;
            }

            // This implementation is an improved implementation of Eq. 11 of Basu, SIAM, 1973
            template<typename Integer, typename Function>
            static auto build_matrix(Integer m, Integer n, Function f, const ChebyshevExpansion2DBounds<NumType> &bounds){

                auto unscalex_ = [&bounds](const auto &xn11) {    
                    return (xn11 * (bounds.xmax - bounds.xmin) + (bounds.xmax + bounds.xmin)) / 2;
                };
                auto unscaley_ = [&bounds](const auto &yn11) {    
                    return (yn11 * (bounds.ymax - bounds.ymin) + (bounds.ymax + bounds.ymin)) / 2;
                };
                ArrayType xroots_n11 = get_xroots(m), yroots_n11 = get_yroots(n);
                ArrayType xroots = unscalex_(xroots_n11), yroots = unscaley_(yroots_n11);

                // Evaluate the function at the tensor product of roots
                CoeffMatType F; F.resize(m + 1, n + 1); F.setZero();
                for (auto i = 0; i < m+1; ++i) {
                    for (auto j = 0; j < n+1; ++j) {
                        F(i, j) = f(xroots[i], yroots[j]);
                    }
                }

                // Now build the matrix
                CoeffMatType a; a.resize(m + 1, n + 1); a.setZero();
                for (auto i = 0; i < m+1; ++i){
                    for (auto j = 0; j < n+1; ++j){
                        auto dsum = 0.0;
                        for (auto r = 0; r < m+1; ++r){
                            for (auto s = 0; s < n+1; ++s){
                                auto xr = xroots_n11[r];
                                auto ys = yroots_n11[s];
                                dsum += F(r, s)*TCheb(i, xr)*TCheb(j, ys);
                            }
                        }
                        a(i, j) = dsum;
                    }
                }
                CoeffMatType c = 4.0/((m+1)*(n+1))*a;
                return c;
            }

            template<typename T>
            static auto factory(
                    const std::function<T(T, T)> &f, 
                    const std::tuple<std::size_t, std::size_t> &orders, 
                    const std::optional<ChebyshevExpansion2DBounds<T>> bounds_ = std::nullopt
            )
            {
                // Use provided bounds, or [-1,1]x[-1,1] if not specified
                auto bounds = (bounds_.has_value()) ? bounds_.value() : ChebyshevExpansion2DBounds<T>{};
                // Get the orders in each direction
                auto [Nx, Ny] = orders;
                // Build matrix
                CoeffMatType mat = build_matrix(Nx, Ny, f, bounds);
                // Return expansion
                return ChebyshevExpansion2D(mat, bounds);
            }
        };
    }
}