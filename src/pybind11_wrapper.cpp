#include "ChebTools/ChebTools.h"
#include "ChebTools/speed_tests.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

namespace py = pybind11;
using namespace ChebTools;

PYBIND11_PLUGIN(ChebTools) {
    py::module m("ChebTools", "C++ tools for working with Chebyshev expansions");

    m.def("mult_by", &mult_by);
    m.def("mult_by_inplace", &mult_by_inplace);

    m.def("evaluation_speed_test", &evaluation_speed_test);
    m.def("eigs_speed_test", &eigs_speed_test);
    m.def("generate_Chebyshev_expansion", &ChebyshevExpansion::factory<std::function<double(double)> >);

    py::class_<SumElement>(m, "SumElement")
        .def(py::init<double, ChebyshevExpansion &, ChebyshevExpansion &>())
        .def_readonly("n_i", &SumElement::n_i)
        .def_readonly("F", &SumElement::F, py::return_value_policy::take_ownership)
        .def_readonly("G", &SumElement::G, py::return_value_policy::take_ownership);

    py::class_<ChebyshevSummation>(m, "ChebyshevSummation")
        .def(py::init<const std::vector<SumElement> &>())
        .def("build_independent_matrix", &ChebyshevSummation::build_independent_matrix)
        .def("get_coefficients", &ChebyshevSummation::get_coefficients)
        ;

    py::class_<ChebyshevExpansion>(m, "ChebyshevExpansion")
        .def(py::init<const std::vector<double> &, double, double>())
        .def(py::self + py::self)
        .def(py::self += py::self)
        .def(py::self * double())
        .def(double() * py::self)
        .def(py::self *= double())
        //.def("__repr__", &Vector2::toString);
        .def("coef", &ChebyshevExpansion::coef)
        .def("companion_matrix", &ChebyshevExpansion::companion_matrix)
        .def("y", (vectype(ChebyshevExpansion::*)(const vectype &)) &ChebyshevExpansion::y)
        .def("y", (double (ChebyshevExpansion::*)(const double)) &ChebyshevExpansion::y)
        .def("y_Clenshaw", &ChebyshevExpansion::y_Clenshaw)
        .def("real_roots", &ChebyshevExpansion::real_roots)
        .def("real_roots_time", &ChebyshevExpansion::real_roots_time)
        .def("real_roots_approx", &ChebyshevExpansion::real_roots_approx)
        .def("subdivide", &ChebyshevExpansion::subdivide)
        .def("real_roots_intervals", &ChebyshevExpansion::real_roots_intervals)
        ;
    return m.ptr();
}