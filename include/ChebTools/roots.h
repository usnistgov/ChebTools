# pragma once

namespace ChebTools{
namespace rootfinding{

/// Balance the matrix to make it prepared for eigenvalue solving
/// Following: https://arxiv.org/pdf/1401.5766.pdf (Algorithm #3)
template<typename mat=Eigen::MatrixXd>
inline void balance_matrix(const mat &A, mat &Aprime, mat &D) {
    // https://arxiv.org/pdf/1401.5766.pdf (Algorithm #3)
    const int p = 2;
    double beta = 2; // Radix base (2?)
    int iter = 0;
    Aprime = A;
    D = mat::Identity(A.rows(), A.cols());
    bool converged = false;
    do {
        converged = true;
        for (Eigen::Index i = 0; i < A.rows(); ++i) {
            double c = Aprime.col(i).template lpNorm<p>();
            double r = Aprime.row(i).template lpNorm<p>();
            double s = pow(c, p) + pow(r, p);
            double f = 1;
            if (!std::isfinite(c)){
                std::cout << A << std::endl;
                throw std::range_error("c is not a valid number in balance_matrixx");
            }
            if (!std::isfinite(r)){
                std::cout << A << std::endl;
                throw std::range_error("r is not a valid number in balance_matrixx");
            }
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

// Given the T matrix in modified block triangular form, extract the real eigenvalues
template<typename mat, int N = mat::RowsAtCompileTime>
inline std::vector<double> Schur_realeigenvalues(const mat &T){
    std::vector<double> roots;
    if constexpr (N > 0){
        for (int i = 0; i < N; ++i) {
            if (i == T.cols()-1 || T(i+1,i) == 0){
                // This is a real 1x1 block, if it were the second row in a 2x2 block it
                // would have been skipped in the next conditional
                roots.push_back(T(i, i));
            }
            else{
                // this is the upper left element of a 2x2 block,
                // keep moving, skip the next row too
                i += 1;
            }
        }
    }
    else{
        for (int i = 0; i < T.cols(); ++i) {
            if (i == T.cols()-1 || T(i+1,i) == 0){
                // This is a real 1x1 block, if it were the second row in a 2x2 block it
                // would have been skipped in the next conditional
                roots.push_back(T(i, i));
            }
            else{
                // this is the upper left element of a 2x2 block,
                // keep moving, skip the next row too
                i += 1;
            }
        }
    }
    return roots;
}

template<int N>
inline Eigen::Matrix<double, N, N> companion_matrixN(const Eigen::Array<double,N+1,1> &coeffs) {
    Eigen::Matrix<double, N, N> A = Eigen::Matrix<double, N, N>::Zero(N, N);
    // First row
    A(0, 1) = 1;
    // Last row
    A.row(N-1) = -coeffs.head(N)/(2.0*coeffs(N));
    A(N - 1, N - 2) += 0.5;
    // All the other rows
    for (int j = 1; j < N - 1; ++j) {
        A(j, j - 1) = 0.5;
        A(j, j + 1) = 0.5;
    }
    return A;
}

inline Eigen::MatrixXd companion_matrix(const Eigen::ArrayXd &coeffs) {
    auto N = coeffs.size() - 1; // degree
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(N, N);
    // First row
    A(0, 1) = 1;
    // Last row
    A.row(N-1) = -coeffs.head(N)/(2.0*coeffs(N));
    A(N - 1, N - 2) += 0.5;
    // All the other rows
    for (int j = 1; j < N - 1; ++j) {
        A(j, j - 1) = 0.5;
        A(j, j + 1) = 0.5;
    }
    return A;
}

template<int N>
inline Eigen::Matrix<double, N, N> companion_matrixN_transposed(const Eigen::Array<double,N+1,1> &coeffs) {
    Eigen::Matrix<double, N, N> A = Eigen::Matrix<double, N, N>::Zero(N, N);
    // First col
    A(1, 0) = 1;
    // Last col
    A.col(N-1) = -coeffs.head(N)/(2.0*coeffs(N));
    A(N - 2, N - 1) += 0.5;
    // All the other cols
    for (int j = 1; j < N - 1; ++j) {
        A(j - 1, j) = 0.5;
        A(j + 1, j) = 0.5;
    }
    return A;
}

inline Eigen::MatrixXd companion_matrix_transposed(const Eigen::ArrayXd &coeffs) {
    auto N = coeffs.size() - 1; // degree
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(N, N);
    // First col
    A(1, 0) = 1;
    // Last col
    A.col(N-1) = -coeffs.head(N)/(2.0*coeffs(N));
    A(N - 2, N - 1) += 0.5;
    // All the other cols
    for (int j = 1; j < N - 1; ++j) {
        A(j - 1, j) = 0.5;
        A(j + 1, j) = 0.5;
    }
    return A;
}


/// Some helper functions to take real values, upcast them to complex before doing further calculations
inline auto pow1(const std::complex<double>& x, int e){ return std::pow(x, e); };
inline auto pow1(double x, int e){ return std::pow(static_cast<std::complex<double>>(x), e); };
inline auto cbrt1(double x){ return cbrt(x); };
inline auto cbrt1(const std::complex<double>& x){ return std::pow(x, 1.0/3.0); };
inline auto sqrt1(double x){ return std::sqrt(std::complex<double>{x,0.0}); };
inline auto sqrt1(const std::complex<double>& x){ return std::sqrt(x); };

/**
 \brief Get the eigenvalues of a 4x4 matrix in lower hessenberg form (as in the case of companion matrices)
 
 \note The loss in precision of the roots can be sometimes quite significant, so much so that the obtained roots are useless, so use with caution, and primarily for testing
 */
inline auto explicit_Chebyshev_4x4(const Eigen::Matrix<double, 4,4>& A4) -> std::tuple<std::complex<double>,std::complex<double>,std::complex<double>,std::complex<double>>{
    double a41 = A4(3,0), a42 = A4(3,1), a43 = A4(3,2), a44 = A4(3,3);
    if (a41 == 0){
        throw std::invalid_argument("not lower Hessenberg");
    }
    //# see: https://en.wikipedia.org/wiki/Quartic_equation#Summary_of_Ferrari's_method
    double A = 1, B = -a44, C = (-2*a43 - 3)/4, D = (-a42+3*a44)/4, E = (-a41+a43)/4.0;
    double a = -3*B*B/(8*A*A) + C/A,
    b = B*B*B/(8*A*A*A)-B*C/(2*A*A)+D/A,
    c = -3*B*B*B*B/(256*A*A*A*A) + C*B*B/(16*A*A*A) - B*D/(4*A*A) + E/A;
    if (b == 0){
        auto solns = std::make_tuple(
           -B/(4*A)+sqrt1((-a+sqrt1(a*a-4*c))/2.0),
           -B/(4*A)+sqrt1((-a-sqrt1(a*a-4*c))/2.0),
           -B/(4*A)-sqrt1((-a+sqrt1(a*a-4*c))/2.0),
           -B/(4*A)-sqrt1((-a-sqrt1(a*a-4*c))/2.0)
       );
        return solns;
    }
    else{
        double P = -a*a/12 - c;
        double Q = -a*a*a/108 + a*c/3 - b*b/8;
        auto R = -Q/2+sqrt1(Q*Q/4 + P*P*P/27);
        auto U = cbrt1(R);
//        auto y = -5.0/6.0*a + ((U == 0) ? -cbrt1(Q) : U-P/(3.0*U));
        auto y = -5.0/6.0*a + (U-P/(3.0*U));
        auto W = sqrt1(a+2.0*y);
        auto solns = std::make_tuple(
           -B/(4*A)+(+W+sqrt1(-(3*a+2.0*y+2*b/W)))/2.0,
           -B/(4*A)+(+W-sqrt1(-(3*a+2.0*y+2*b/W)))/2.0,
           -B/(4*A)+(-W+sqrt1(-(3*a+2.0*y-2*b/W)))/2.0,
           -B/(4*A)+(-W-sqrt1(-(3*a+2.0*y-2*b/W)))/2.0
        );
        
        auto check_one = [&](auto& x){
//                auto chkval = A*x*x*x*x + B*x*x*x + C*x*x + D*x + E;
//                if ((std::imag(chkval) != 0) || std::abs(std::real(chkval)) < 1e-10){
//                    std::cout << x << "," << chkval << std::endl;
//                }
        };
        // See example at https://stackoverflow.com/a/54053084
        auto checker = [&check_one](auto&&... root) {((  check_one(root)  ), ...);};
        std::apply(checker, solns);
        
        return solns;
    }
     
}

/** 
 Get the eigenvalues of a 3x3 matrix in lower hessenberg form (as in the case of companion matrices)
 \note The loss in precision of the roots can be sometimes quite significant, so much so that the obtained roots are useless, so use with caution, and primarily for testing
 */
inline auto explicit_Chebyshev_3x3(const Eigen::Matrix<double, 3,3>& A3){
    double a31 = A3(2,0), a32 = A3(2,1), a33 = A3(2,2);
    if (a31 == 0){
        throw std::invalid_argument("not lower Hessenberg");
    }
    std::complex<double> I{0,1.0};
    auto sqrt3 = sqrt(3.0);
    auto F = cbrt1(-27*a31 - 4*pow(a33, 3) - 9*a33*(a32 + 1) + 27*a33 + sqrt1(-2*pow(3*a32 + 2*pow(a33, 2) + 3, 3) + pow(-27*a31 - 4.0*pow(a33, 3) - 9*a33*(a32 + 1) + 27*a33, 2)));
    std::complex<double> soln1 = (1.0/6.0)*(F*(-cbrt(2)*F + 2*a33) - pow(2, 2.0/3.0)*(3*a32 + 2*pow(a33, 2) + 3))/F;
    std::complex<double> soln2 = (1.0/12.0)*(F*(1.0 + sqrt3*I)*(cbrt(2)*F*(1.0 + sqrt3*I) + 4*a33) + 4*pow(2, 2.0/3.0)*(3*a32 + 2*pow(a33, 2) + 3))/(F*(1.0 + sqrt3*I));
    std::complex<double> soln3 = (1.0/12.0)*(F*(1.0 - sqrt3*I)*(cbrt(2)*F*(1.0 - sqrt3*I) + 4*a33) + 4*pow(2, 2.0/3.0)*(3*a32 + 2*pow(a33, 2) + 3))/(F*(1.0 - sqrt3*I));

    return std::make_tuple(soln1, soln2, soln3);
}

}
}

