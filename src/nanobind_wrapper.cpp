#include "ChebTools/ChebTools.h"
#include "ChebTools/speed_tests.h"

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/eigen/dense.h>

namespace nb = nanobind;

using namespace nb::literals;

using namespace ChebTools;

template<typename vectype>
auto Clenshaw1D(const vectype &c, double ind){
    int N = static_cast<int>(c.size()) - 1;
    typename vectype::Scalar u_k = 0, u_kp1 = 0, u_kp2 = 0;
    for (int k = N; k >= 0; --k){
        // Do the recurrent calculation
        u_k = 2.0*ind*u_kp1 - u_kp2 + c[k];
        if (k > 0){
            // Update the values
            u_kp2 = u_kp1; u_kp1 = u_k;
        }
    }
    return (u_k - u_kp2)/2.0;
}

template<typename MatType, int Cols = MatType::ColsAtCompileTime>
auto Clenshaw1DByRow(const MatType& c, double ind) {
    int N = static_cast<int>(c.rows()) - 1;
    static Eigen::Array<typename MatType::Scalar, 1, Cols> u_k, u_kp1, u_kp2;
    // Not statically sized    
    if constexpr (Cols < 0) {
        int M = static_cast<int>(c.rows());
        u_k.resize(M); 
        u_kp1.resize(M);
        u_kp2.resize(M);
    }
    u_k.setZero(); u_kp1.setZero(); u_kp2.setZero();
    
    for (int k = N; k >= 0; --k) {
        // Do the recurrent calculation
        u_k = 2.0 * ind * u_kp1 - u_kp2 + c.row(k);
        if (k > 0) {
            // Update the values
            u_kp2 = u_kp1; u_kp1 = u_k;
        }
    }
    return (u_k - u_kp2) / 2.0;
}

/// With Eigen datatypes
template<typename MatType>
auto Clenshaw2DEigen(const MatType& a, double x, double y) {
    auto b = Clenshaw1DByRow(a, y);
    return Clenshaw1D(b.matrix(), x);
}

void init_ChebTools(nb::module_ &m){

    nb::class_<ChebyshevExpansion>(m, "ChebyshevExpansion")
        .def(nb::init<const std::vector<double> &, double, double>())
        .def(nb::self + nb::self)
        .def(nb::self += nb::self)
        .def(nb::self + double())
        .def(nb::self - double())
        .def(nb::self * double())
        .def(double() * nb::self)
        .def(nb::self *= double())
        .def(nb::self * nb::self)
        // Unary operators
        .def(-nb::self)

        .def("times_x", &ChebyshevExpansion::times_x)
        .def("times_x_inplace", &ChebyshevExpansion::times_x_inplace)
        .def("apply", &ChebyshevExpansion::apply)
        //.def("__repr__", &Vector2::toString);
        .def("coef", &ChebyshevExpansion::coef)
        .def("companion_matrix", &ChebyshevExpansion::companion_matrix)
        .def("y", (vectype(ChebyshevExpansion::*)(const vectype &) const) &ChebyshevExpansion::y)
        .def("y", (double (ChebyshevExpansion::*)(const double) const) &ChebyshevExpansion::y)
        .def("y_Clenshaw", &ChebyshevExpansion::y_Clenshaw)
        .def("real_roots", &ChebyshevExpansion::real_roots)
        .def("real_roots2", &ChebyshevExpansion::real_roots2)
        .def("real_roots_UH", &ChebyshevExpansion::real_roots_UH)
        .def("real_roots_time", &ChebyshevExpansion::real_roots_time)
        .def("real_roots_approx", &ChebyshevExpansion::real_roots_approx)
        .def("is_monotonic", &ChebyshevExpansion::is_monotonic)
        .def("has_real_roots_Descartes", &ChebyshevExpansion::has_real_roots_Descartes)
        .def("to_monomial_increasing", &ChebyshevExpansion::to_monomial_increasing)
        .def("subdivide", &ChebyshevExpansion::subdivide)
        .def("real_roots_intervals", &ChebyshevExpansion::real_roots_intervals)
        .def("deriv", &ChebyshevExpansion::deriv)
        .def("integrate", &ChebyshevExpansion::integrate)
        .def("xmin", &ChebyshevExpansion::xmin)
        .def("xmax", &ChebyshevExpansion::xmax)
        .def("xmid", &ChebyshevExpansion::xmid)
        .def("scale_x", &ChebyshevExpansion::scale_x)
        .def("unscale_x", &ChebyshevExpansion::unscale_x)
        .def("split_apart", &ChebyshevExpansion::split_apart)
        .def("get_nodes_n11", nb::overload_cast<>(&ChebyshevExpansion::get_nodes_n11, nb::const_), "Get the Chebyshev-Lobatto nodes in [-1,1]")
        .def("get_nodes_realworld", nb::overload_cast<>(&ChebyshevExpansion::get_nodes_realworld, nb::const_), "Get the Chebyshev-Lobatto nodes in [xmin, xmax]")
        .def("get_node_function_values", &ChebyshevExpansion::get_node_function_values)
        .def("cache_nodal_function_values", &ChebyshevExpansion::cache_nodal_function_values)
        .def("monotonic_solvex", &ChebyshevExpansion::monotonic_solvex)
        ;

    using Container = ChebyshevCollection::Container;
    nb::class_<ChebyshevCollection>(m, "ChebyshevCollection")
        .def(nb::init<const Container&>())
        .def("__call__", [](const ChebyshevCollection& c, const double x) { return c(x); }, nb::is_operator())
        .def("y_unsafe", &ChebyshevCollection::y_unsafe)
        .def("integrate", &ChebyshevCollection::integrate)
        .def("get_xmin", &ChebyshevCollection::get_xmin)
        .def("get_xmax", &ChebyshevCollection::get_xmax)
    
        .def("get_exps", &ChebyshevCollection::get_exps)
        .def("get_extrema", &ChebyshevCollection::get_extrema)
        .def("solve_for_x", &ChebyshevCollection::solve_for_x)
        .def("make_inverse", &ChebyshevCollection::make_inverse, "N"_a, "xmin"_a, "xmax"_a, "M"_a, "tol"_a, "max_refine_passes"_a, "assume_monotonic"_a = false, "unsafe_evaluation"_a = false)
        .def("get_index", &ChebyshevCollection::get_index)
        .def("get_hinted_index", &ChebyshevCollection::get_hinted_index)
        ;

    using TE = TaylorExtrapolator<Eigen::ArrayXd>;
    nb::class_<TE>(m, "TaylorExtrapolator")
        .def("__call__", [](const TE& c, const Eigen::ArrayXd &x) { return c(x); }, nb::is_operator())
        .def("__call__", [](const TE& c, const double& x) { return c(x); }, nb::is_operator())
        .def("get_coef", &TaylorExtrapolator<Eigen::ArrayXd>::get_coef)
        ;

    m.def("mult_by", &mult_by);
    m.def("mult_by_inplace", &mult_by_inplace);
    m.def("evaluation_speed_test", &evaluation_speed_test);
    m.def("eigs_speed_test", &eigs_speed_test);
    m.def("eigenvalues", &eigenvalues);
    m.def("eigenvalues_upperHessenberg", &eigenvalues_upperHessenberg);
    m.def("Schur_matrixT", &Schur_matrixT);
    m.def("Schur_realeigenvalues", &Schur_realeigenvalues);
    m.def("factoryfDCT", &ChebyshevExpansion::factoryf);
    m.def("factoryfFFT", &ChebyshevExpansion::factoryfFFT);
    m.def("generate_Chebyshev_expansion", &ChebyshevExpansion::factory<std::function<double(double)> >, "N"_a, "func"_a, "xmin"_a, "xmax"_a);
    m.def("dyadic_splitting", &ChebyshevExpansion::dyadic_splitting<std::vector<ChebyshevExpansion>>, "N"_a, "func"_a, "xmin"_a, "xmax"_a, "M"_a, "tol"_a, "max_refine_passes"_a = 8, "callback"_a=nb::none());
    m.def("Eigen_nbThreads", []() { return Eigen::nbThreads(); });
    m.def("Eigen_setNbThreads", [](int Nthreads) { return Eigen::setNbThreads(Nthreads); });
    m.def("get_monomial_from_Cheb_basis", &get_monomial_from_Cheb_basis);
    m.def("count_sign_changes", &count_sign_changes);

    m.def("Clenshaw2DEigen", &Clenshaw2DEigen<Eigen::Ref<const Eigen::ArrayXXd>>, "cmat"_a.noconvert(), "x"_a, "y"_a);
    m.def("Clenshaw2DEigencomplex", &Clenshaw2DEigen<Eigen::Ref<const Eigen::ArrayXXcd>>, "cmat"_a.noconvert(), "x"_a, "y"_a);
}

NB_MODULE(ChebTools, m) {
    m.doc() = "C++ tools for working with Chebyshev expansions";
    m.attr("__version__") = CHEBTOOLSVERSION;
    init_ChebTools(m);
}
