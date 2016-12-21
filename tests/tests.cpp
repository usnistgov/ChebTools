#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "ChebTools/ChebTools.h"

/*
From numpy:
----------
from numpy.polynomial.chebyshev import Chebyshev
c = Chebyshev([1,2,3,4])
print c.coef
print c.deriv(1).coef
print c.deriv(2).coef
print c.deriv(3).coef
*/
TEST_CASE("Expansion derivatives (3rd order)", "")
{
    Eigen::Vector4d c;
    c << 1, 2, 3, 4;

    auto ce = ChebTools::ChebyshevExpansion(c, -1, 1);
    SECTION("first derivative") {
        Eigen::Vector3d c_expected; c_expected << 14,12,24;
        auto d1 = ce.deriv(1);
        auto d1c = d1.coef();
        auto err = std::abs((c_expected - d1c).sum());
        CAPTURE(err);
        CHECK(err < 1e-100);
    }
    SECTION("second derivative") {
        Eigen::Vector2d c_expected; c_expected << 12, 96;
        auto d2 = ce.deriv(2);
        auto d2c = d2.coef();
        auto err = std::abs((c_expected - d2c).sum());
        CAPTURE(err);
        CHECK(err < 1e-100);
    }
    SECTION("third derivative") {
        Eigen::VectorXd c_expected(1); c_expected << 96;
        auto d3 = ce.deriv(3);
        auto d3c = d3.coef();
        auto err = std::abs((c_expected - d3c).sum());
        CAPTURE(err);
        CHECK(err < 1e-100);
    }
}

/*
From numpy:
----------
from numpy.polynomial.chebyshev import Chebyshev
c = Chebyshev([1,2,3,4,5])
print c.coef
print c.deriv(1).coef
print c.deriv(2).coef
print c.deriv(3).coef
print c.deriv(4).coef
*/
TEST_CASE("Expansion derivatives (4th order)", "")
{
    Eigen::VectorXd c(5);
    c << 1, 2, 3, 4, 5;

    auto ce = ChebTools::ChebyshevExpansion(c, -1, 1);
    SECTION("first derivative") {
        Eigen::Vector4d c_expected; c_expected << 14, 52, 24, 40;
        auto d1c = ce.deriv(1).coef();
        auto err = std::abs((c_expected - d1c).sum());
        CAPTURE(err);
        CHECK(err < 1e-100);
    }
    SECTION("second derivative") {
        Eigen::Vector3d c_expected; c_expected << 172, 96, 240;
        auto d2c = ce.deriv(2).coef();
        auto err = std::abs((c_expected - d2c).sum());
        CAPTURE(err);
        CHECK(err < 1e-100);
    }
    SECTION("third derivative") {
        Eigen::Vector2d c_expected; c_expected << 96, 960;
        auto d3c = ce.deriv(3).coef();
        auto err = std::abs((c_expected - d3c).sum());
        CAPTURE(err);
        CHECK(err < 1e-100);
    }
    SECTION("fourth derivative") {
        Eigen::VectorXd c_expected(1); c_expected << 960;
        auto d4c = ce.deriv(4).coef();
        auto err = std::abs((c_expected - d4c).sum());
        CAPTURE(err);
        CHECK(err < 1e-100);
    }
}

TEST_CASE("Expansion from single monomial term", "")
{
    // From Mason and Handscomb, Chebyshev Polynomials, p. 23
    auto ce = ChebTools::ChebyshevExpansion::from_powxn(4, -1, 1);
    Eigen::VectorXd c_expected(5); c_expected << 3.0/8.0, 0, 0.5, 0, 1.0/8.0;
    auto err = std::abs((c_expected - ce.coef()).sum());
    CAPTURE(err);
    CHECK(err < 1e-100);
}

TEST_CASE("Expansion from polynomial", "")
{
    Eigen::VectorXd c_poly(4); c_poly << 0, 1, 2, 3; 
    Eigen::VectorXd c_expected(4); c_expected << 1.0, 3.25, 1.0, 0.75;

    // From https ://docs.scipy.org/doc/numpy/reference/generated/numpy.polynomial.chebyshev.poly2cheb.html
    auto ce = ChebTools::ChebyshevExpansion::from_polynomial(c_poly, -1, 1);
    
    auto err = std::abs((c_expected - ce.coef()).sum());
    CAPTURE(err);
    CHECK(err < 1e-100);
}
