//#if defined(_MSC_VER)
//#   if _MSC_VER < 1900 
//    // Do nothing, not C++11 compliant
//#   else
//#       define CHEBTOOLS_CPP11
//#   endif
//#elif __cplusplus <= 199711L
//  // Do nothing, not C++11 compliant
//#else
//# define CHEBTOOLS_CPP11
//#endif

#define CHEBTOOLS_CPP11

#include <algorithm>
#include <functional>
#include <vector>
#include <chrono>
#include <iostream>
#include <valarray>

#include "Eigen/Dense"

typedef Eigen::VectorXd vectype;

class ChebyshevExpansion {
private:
     vectype m_c;
     vectype m_recurrence_buffer;
     Eigen::MatrixXd m_recurrence_buffer_matrix;
     void resize(){
         m_recurrence_buffer.resize(m_c.size());
     }

public:
    ChebyshevExpansion(const vectype &c) : m_c(c) { resize(); };
    ChebyshevExpansion(const std::vector<double> &c) { 
        m_c = Eigen::Map<const Eigen::VectorXd>(&(c[0]), c.size());
        resize();
    };

#if defined(CHEBTOOLS_CPP11)
    // Move constructor (C++11 only)
    ChebyshevExpansion(const vectype &&c) : m_c(c) { resize(); };
#endif
    
public:
    ChebyshevExpansion operator+(const ChebyshevExpansion &ce2) const {
        // TODO: when m_c and ce2.coef() not the same size, resize shorter one and pad the longer one
        if (m_c.size() != ce2.coef().size()) { throw std::exception("lengths not the same"); }
#if defined(CHEBTOOLS_CPP11) 
        return ChebyshevExpansion(std::move(ce2.coef()+m_c));
#else
        return ChebyshevExpansion(c);
#endif
    };
    ChebyshevExpansion& operator+=(const ChebyshevExpansion &donor) {
        std::size_t Ndonor = donor.coef().size(), N1 = m_c.size();
        std::size_t Nmin = std::min(N1, Ndonor), Nmax = std::max(N1, Ndonor);
        // The first Nmin terms overlap between the two vectors
        m_c.head(Nmin) += donor.coef().head(Nmin);
        // If the donor vector is longer than the current vector, resizing is needed
        if (Ndonor > N1) {
            m_c.resize(Ndonor);
            // Copy the last Nmax-Nmin values from the donor
            m_c.tail(Nmax-Nmin) = donor.coef().tail(Nmax-Nmin);
        }
        return *this;
    }
    ChebyshevExpansion operator*(double value) const { 
#if defined(CHEBTOOLS_CPP11) 
        return ChebyshevExpansion(std::move(m_c*value));
#else
        return ChebyshevExpansion(m_c*value);
#endif
    }
    ChebyshevExpansion& operator*=(double value) {
        m_c *= value;
        return *this; 
    }
    /// Friend function that allows for pre-multiplication by a constant value
    friend ChebyshevExpansion operator*(double value, const ChebyshevExpansion &ce) {
#if defined(CHEBTOOLS_CPP11) 
        return ChebyshevExpansion(std::move(ce.coef()*value));
#else
        return ChebyshevExpansion(mult_coeffs_by_double(ce.coef(), value));
#endif
    }

    const vectype &coef() const {
        return m_c; 
    };
    double y(const double x) {
        // Use the recurrence relationships to evaluate the Chebyshev expansion
        std::size_t Norder = m_c.size()-1;
        vectype &o = m_recurrence_buffer;
        o(0) = 1;
        o(1) = x;
        for (int n = 1; n < Norder; ++n){
            o(n+1) = 2*x*o(n) - o(n-1);
        }
        return m_c.dot(o);
    }
    vectype y(const vectype &x) {
        // Use the recurrence relationships to evaluate the Chebyshev expansion
        std::size_t Norder = m_c.size() - 1;
        Eigen::MatrixXd &A = m_recurrence_buffer_matrix;
        A.resize(x.size(), Norder+1);
        A.col(0).fill(1);
        A.col(1) = x;
        for (int n = 1; n < Norder; ++n) {
            A.col(n + 1).array() = 2 * x.array()*A.col(n).array() - A.col(n - 1).array();
        }
        return A*m_c;
    }

    //std::string toString() const {
    //    return "[" + std::to_string(x) + ", " + std::to_string(y) + "]";
    //}
};


int p_i(int i, int N){
    if (i == 0 || i == N)
        return 2;
    else
        return 1;
}

ChebyshevExpansion generate_Chebyshev_expansion(int N, double (*func)(double), double xmin, double xmax)
{
    Eigen::VectorXd f(N + 1);

    // See Boyd, 2013

    // Step 1&2: Grid points functional values (function evaluated at the
    // roots of the Chebyshev polynomial of order N)
    for (int k = 0; k <= N; ++k){
        double x_k = (xmax - xmin)/2.0*cos((EIGEN_PI*k)/N) + (xmax + xmin) / 2.0;
        f(k) = (*func)(x_k);
    }

    // Step 3: Constrct the matrix of coefficients used to obtain a
    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(N + 1, N + 1); ///< Matrix of coefficients
    for (int j = 0; j <= N; ++j){
        for (int k = 0; k <= N; ++k){
            L(j, k) = 2.0/(p_i(j, N)*p_i(k, N)*N)*cos((j*EIGEN_PI*k)/N);
        }
    }

    // Step 4: Obtain coefficients from vector - matrix product
    Eigen::VectorXd c = (L*f).rowwise().sum();
    return ChebyshevExpansion(c);
}


double plus_by_inplace(ChebyshevExpansion &ce, const ChebyshevExpansion &ce2, int N) {
    for (std::size_t i = 0; i < N; ++i) {
        ce += ce2;
    }
    return ce.coef()(0);
}

double mult_by_inplace(ChebyshevExpansion &ce, double val, int N) {
    for (std::size_t i = 0; i < N; ++i) {
        ce *= val;
    }
    return ce.coef()(0);
}

void mult_by(ChebyshevExpansion &ce, double val, int N) {
    Eigen::VectorXd c(2); c << 1, 0;
    ChebyshevExpansion ce2(c);
    for (std::size_t i = 0; i < N; ++i) {
        ce2 = ce*val;
    }
    //return ce2;
}

double f(double x){
    return exp(-pow(x,1));
}

#if defined(PYBIND11)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

namespace py = pybind11;

ChebyshevExpansion generate_Chebyshev_expansion2(int N, const std::function<double(double)> &func, double xmin, double xmax)
{
    Eigen::VectorXd f(N + 1);

    // See Boyd, 2013

    // Step 1&2: Grid points functional values (function evaluated at the
    // roots of the Chebyshev polynomial of order N)
    for (int k = 0; k <= N; ++k) {
        double x_k = (xmax - xmin) / 2.0*cos((EIGEN_PI*k) / N) + (xmax + xmin) / 2.0;
        f(k) = func(x_k);
    }

    // Step 3: Constrct the matrix of coefficients used to obtain a
    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(N + 1, N + 1); ///< Matrix of coefficients
    for (int j = 0; j <= N; ++j) {
        for (int k = 0; k <= N; ++k) {
            L(j, k) = 2.0 / (p_i(j, N)*p_i(k, N)*N)*cos((j*EIGEN_PI*k) / N);
        }
    }

    // Step 4: Obtain coefficients from vector - matrix product
    Eigen::VectorXd c = (L*f).rowwise().sum();
    return ChebyshevExpansion(c);
}

PYBIND11_PLUGIN(ChebTools) {
    py::module m("ChebTools", "C++ tools for working with Chebyshev expansions");

    m.def("mult_by", &mult_by);
    m.def("mult_by_inplace", &mult_by_inplace);
    m.def("generate_Chebyshev_expansion2", &generate_Chebyshev_expansion2);

    
    py::class_<ChebyshevExpansion>(m, "ChebyshevExpansion")
        .def(py::init<const std::vector<double> &>())
        .def(py::self + py::self)
        .def(py::self += py::self)
        .def(py::self * double())
        .def(double() * py::self)
        .def(py::self *= double())
        //.def("__repr__", &Vector2::toString);
        .def("coef", &ChebyshevExpansion::coef)
        ;
    return m.ptr();
}
#else

// Monolithic build
int main(){

    ChebyshevExpansion ce0 = generate_Chebyshev_expansion(10, *f, 0, 6);
    std::cout << ce0.coef() << std::endl;

    long N = 1000000;
    Eigen::VectorXd c(50);
    c.fill(1);
    ChebyshevExpansion ce(c);

    auto startTime = std::chrono::system_clock::now();
        mult_by_inplace(ce, 1.001, N);
    auto endTime = std::chrono::system_clock::now();
    auto elap_us = std::chrono::duration<double>(endTime - startTime).count()/N*1e6;
    std::cout << elap_us << " us/call (mult inplace)\n";
    
    startTime = std::chrono::system_clock::now();
    plus_by_inplace(ce, ce, N);
    endTime = std::chrono::system_clock::now();
    elap_us = std::chrono::duration<double>(endTime - startTime).count()/N*1e6;
    std::cout << elap_us << " us/call (plus inplace)\n";

    startTime = std::chrono::system_clock::now();
        mult_by(ce, 1.001, N);
    endTime = std::chrono::system_clock::now();
    elap_us = std::chrono::duration<double>(endTime - startTime).count()/N*1e6;
    std::cout << elap_us << " us/call (mult)\n";

    int Norder = 50, Npoints = 200;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(Npoints, Norder+1);
    Eigen::VectorXd a = Eigen::VectorXd::Random(Norder + 1);
    Eigen::VectorXd xpts = (Eigen::VectorXd::LinSpaced(Npoints, 0, Npoints-1)*EIGEN_PI / Npoints).array().cos();
    Eigen::VectorXd ypts;
    std::cout << a << std::endl;

    startTime = std::chrono::system_clock::now();
        for (int i = 0; i < N; ++i){
            A.col(1) = xpts;
            for (int n=1; n < Norder; ++n){
                A.col(n + 1).array() = 2*xpts.array()*A.col(n).array() - A.col(n - 1).array();
            }
            ypts = A*a;
        }
        std::cout << "y[0]:" << ypts[0] << std::endl;
        Eigen::VectorXi signchange = (ypts.segment(0, Npoints).cwiseSign().cast<int>().array()*ypts.segment(1, Npoints).cwiseSign().cast<int>().array()) - 1;
        std::cout << "this many roots:" << signchange.count() << std::endl;
    endTime = std::chrono::system_clock::now();
    elap_us = std::chrono::duration<double>(endTime - startTime).count()/N*1e6;
    std::cout << elap_us << " us/call (yvals)\n";

    Eigen::MatrixXd B = Eigen::MatrixXd::Random(50, 50);
    N = 100;
    startTime = std::chrono::system_clock::now();
        const bool computeEigenvectors = false; 
        for (int i = 0; i < N; ++i){
            Eigen::EigenSolver<Eigen::MatrixXd> es(B, computeEigenvectors);
        }
    endTime = std::chrono::system_clock::now();
    elap_us = std::chrono::duration<double>(endTime - startTime).count()/N*1e6;
    std::cout << elap_us << " us/call (eigs 50x50)\n";

    return EXIT_SUCCESS;
}
#endif