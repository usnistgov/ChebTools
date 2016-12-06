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

class ChebyshevExpansion {
private:
    std::vector<double> m_c;

    void add_coeffs(const std::vector<double>&c1, const std::vector<double>&c2, std::vector<double> &out) const{
        const std::size_t N1 = c1.size(), N2 = c2.size();
        const std::size_t N = std::max(N1, N2), Nend = std::min(N1, N2);
        out.resize(N);
        
        // Add together all the indices that overlap
        std::transform(c1.begin(), c1.begin() + Nend, c2.begin(), out.begin(), std::plus<double>());

        // Copy the indices that DO NOT overlap
        if (N1 > N2) {
            std::copy(c1.begin() + Nend, c1.begin() + N, out.begin() + Nend);
        }
        else if (N2 > N1) {
            std::copy(c2.begin() + Nend, c2.begin() + N, out.begin() + Nend);
        }
        // Nothing more to be done if N1 == N2
    }
    const std::vector<double> mult_coeffs_by_double(std::vector<double>c, double val) const {
        // See http://stackoverflow.com/a/3885136
        std::transform(c.begin(), c.end(), c.begin(), std::bind1st(std::multiplies<double>(), val));
        return c;
    }
    void mult_coeffs_by_double_inplace(std::vector<double> &c, double val) {
        // See http://stackoverflow.com/a/3885136
        std::transform(c.begin(), c.end(), c.begin(), std::bind1st(std::multiplies<double>(), val));
    }

public:
    ChebyshevExpansion(const std::vector<double> &c) : m_c(c) { };

#if defined(CHEBTOOLS_CPP11)
    // Move constructor (C++11 only)
    ChebyshevExpansion(const std::vector<double> &&c) : m_c(c) { };
#endif
    
public:
    ChebyshevExpansion operator+(const ChebyshevExpansion &ce2) const {
        const std::vector<double> &c2 = ce2.coef(); 
        std::vector<double> c;
        add_coeffs(m_c, c2, c);
#if defined(CHEBTOOLS_CPP11) 
        return ChebyshevExpansion(std::move(c));
#else
        return ChebyshevExpansion(c);
#endif
    };
    ChebyshevExpansion& operator+=(const ChebyshevExpansion &ce2) {
        add_coeffs(m_c, ce2.coef(), m_c);
        return *this;
    }
    ChebyshevExpansion operator*(double value) const { 
#if defined(CHEBTOOLS_CPP11) 
        return ChebyshevExpansion(std::move(mult_coeffs_by_double(m_c, value)));
#else
        return ChebyshevExpansion(mult_coeffs_by_double(c, value));
#endif
    }
    ChebyshevExpansion& operator*=(double value) {
        mult_coeffs_by_double_inplace(m_c, value);
        return *this; 
    }
    /// Friend function that allows for pre-multiplication by a constant value
    friend ChebyshevExpansion operator*(double value, const ChebyshevExpansion &ce) {
#if defined(CHEBTOOLS_CPP11) 
        return ChebyshevExpansion(std::move(ce.mult_coeffs_by_double(ce.coef(), value)));
#else
        return ChebyshevExpansion(mult_coeffs_by_double(ce.coef(), value));
#endif
    }

    const std::vector<double> &coef() const { 
        return m_c; 
    };

    //std::string toString() const {
    //    return "[" + std::to_string(x) + ", " + std::to_string(y) + "]";
    //}
};

#if defined(PYBIND11)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
namespace py = pybind11;

PYBIND11_PLUGIN(ChebTools) {
    py::module m("ChebTools", "C++ tools for working with Chebyshev expansions");

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

#endif