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
    m.def("to_p", [](ChebyshevExpansion &dalphar_dDelta, double rhoRT){ return rhoRT*(dalphar_dDelta.times_x() + 1.0).times_x(); });

    m.def("evaluation_speed_test", &evaluation_speed_test);
    m.def("eigs_speed_test", &eigs_speed_test);
    m.def("generate_Chebyshev_expansion", &ChebyshevExpansion::factory<std::function<double(double)> >);
    m.def("Eigen_nbThreads", []() { return Eigen::nbThreads(); });
    m.def("Eigen_setNbThreads", [](int Nthreads) { return Eigen::setNbThreads(Nthreads); });

    py::class_<SumElement>(m, "SumElement")
        .def(py::init<double, ChebyshevExpansion &, ChebyshevExpansion &>())
        .def_readonly("n_i", &SumElement::n_i)
        .def_readonly("F", &SumElement::F, py::return_value_policy::take_ownership)
        .def_readonly("G", &SumElement::G, py::return_value_policy::take_ownership);

    py::class_<ChebyshevSummation>(m, "ChebyshevSummation")
        .def(py::init<const std::vector<SumElement> &, double, double>())
        .def("build_independent_matrix", &ChebyshevSummation::build_independent_matrix)
        .def("build_dependent_matrix", &ChebyshevSummation::build_dependent_matrix)
        .def("get_coefficients", &ChebyshevSummation::get_coefficients)
        .def("get_independent_matrix", &ChebyshevSummation::get_independent_matrix)
        .def("get_dependent_matrix", &ChebyshevSummation::get_dependent_matrix)
        .def("xmin", &ChebyshevSummation::xmin)
        .def("xmax", &ChebyshevSummation::xmax)
        ;

    py::class_<ChebyshevMixture>(m, "ChebyshevMixture")
        .def(py::init<const std::vector<std::vector<ChebyshevSummation> > &>())
        .def("get_A", &ChebyshevMixture::get_A)
        .def("get_p", &ChebyshevMixture::get_p)
        .def("time_get", &ChebyshevMixture::time_get)
        .def("get_dalphar_ddelta", &ChebyshevMixture::get_dalphar_ddelta)
        .def("Nintervals", &ChebyshevMixture::Nintervals)
        .def("get_intervals", &ChebyshevMixture::get_intervals)
        .def("calc_real_roots", &ChebyshevMixture::calc_real_roots)
        .def("time_calc_real_roots", &ChebyshevMixture::time_calc_real_roots)
        .def("get_real_roots", &ChebyshevMixture::get_real_roots)
        .def("calc_companion_matrices", &ChebyshevMixture::calc_companion_matrices)
        .def("unlikely_root", &ChebyshevMixture::unlikely_root)
        .def("eigenvalues", &ChebyshevMixture::eigenvalues)
        ;

    py::class_<ChebyshevExpansion>(m, "ChebyshevExpansion")
        .def(py::init<const std::vector<double> &, double, double>())
        .def(py::self + py::self)
        .def(py::self += py::self)
        .def(py::self + double())
        .def(py::self - double())
        .def(py::self * double())
        .def(double() * py::self)
        .def(py::self *= double())
        .def(py::self * py::self)
        .def("times_x", &ChebyshevExpansion::times_x)
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
        .def("deriv", &ChebyshevExpansion::deriv)
        .def("xmin", &ChebyshevExpansion::xmin)
        .def("xmax", &ChebyshevExpansion::xmax)
        .def("get_nodes_n11", &ChebyshevExpansion::get_nodes_n11)
        .def("get_node_function_values", &ChebyshevExpansion::get_node_function_values)
        ;
    return m.ptr();
}