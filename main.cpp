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

typedef std::valarray<double> vectype;

class ChebyshevExpansion {
private:
     vectype m_c;
public:
    ChebyshevExpansion(const vectype &c) : m_c(c) { };

    ChebyshevExpansion(const std::vector<double> &c) : m_c(&(c[0]), c.size()) { };

#if defined(CHEBTOOLS_CPP11)
    // Move constructor (C++11 only)
    ChebyshevExpansion(const vectype &&c) : m_c(c) { };
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
    ChebyshevExpansion& operator+=(const ChebyshevExpansion &ce2) {
        // TODO: when m_c and ce2.coef() not the same size, resize shorter one and pad the longer one
        if (m_c.size() != ce2.coef().size()){ throw std::exception("lengths not the same"); }
        m_c += ce2.coef();
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

    //std::string toString() const {
    //    return "[" + std::to_string(x) + ", " + std::to_string(y) + "]";
    //}
};

double plus_by_inplace(ChebyshevExpansion &ce, const ChebyshevExpansion &ce2, int N) {
    for (std::size_t i = 0; i < N; ++i) {
        ce += ce2;
    }
    return ce.coef()[0];
}

double mult_by_inplace(ChebyshevExpansion &ce, double val, int N) {
    for (std::size_t i = 0; i < N; ++i) {
        ce *= val;
    }
    return ce.coef()[0];
}

void mult_by(ChebyshevExpansion &ce, double val, int N) {
    ChebyshevExpansion ce2({ 1,0 });
    for (std::size_t i = 0; i < N; ++i) {
        ce2 = ce*val;
    }
    //return ce2;
}


#if defined(PYBIND11)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
namespace py = pybind11;

PYBIND11_PLUGIN(ChebTools) {
    py::module m("ChebTools", "C++ tools for working with Chebyshev expansions");

    m.def("mult_by", &mult_by);
    m.def("mult_by_inplace", &mult_by_inplace);

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

    long N = 1000000;
    std::valarray<double> c(1, 50);
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

    Eigen::MatrixXd A = Eigen::MatrixXd::Random(50, 50);
    N = 100;
    startTime = std::chrono::system_clock::now();
        const bool computeEigenvectors = false; 
        for (int i = 0; i < N; ++i){
            Eigen::EigenSolver<Eigen::MatrixXd> es(A, computeEigenvectors);
        }
    endTime = std::chrono::system_clock::now();
    elap_us = std::chrono::duration<double>(endTime - startTime).count()/N*1e6;
    std::cout << elap_us << " us/call (eigs 50x50)\n";

    return EXIT_SUCCESS;
}
#endif