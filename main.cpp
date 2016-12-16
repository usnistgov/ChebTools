
#include "ChebTools/ChebTools.h"
#include "ChebTools/speed_tests.h"

#include <iostream>
#include <chrono>

double f(double x){
    return pow(x,3);
    //return exp(-5*pow(x,2)) - 0.5;
}

// Monolithic build
int main(){

    using namespace ChebTools;

    ChebyshevExpansion ee = ChebyshevExpansion::from_powxn(3, -1, 1);
    std::cout << ee.coef() << std::endl;
    ee = ChebyshevExpansion::factory(40, f, -1, 1);
    std::cout << ee.coef() << std::endl;

    {
        Eigen::MatrixXd mat = Eigen::MatrixXd::Random(20, 50);
        Eigen::VectorXd Tpart = Eigen::VectorXd::Random(20);

        long N = 1000000;
        Eigen::VectorXd c;
        auto startTime = std::chrono::system_clock::now();
        for (int i = 0; i < N; ++i) {
            // see http://stackoverflow.com/a/36849915/1360263
            c = (mat.array().colwise() * Tpart.array()).colwise().sum();
        }
        auto endTime = std::chrono::system_clock::now();
        auto elap_us = std::chrono::duration<double>(endTime - startTime).count() / N*1e6;
        std::cout << elap_us << " us/call (matrix coeff eval)\n";
    }

    {
        long N = 10000;
        auto startTime = std::chrono::system_clock::now(); 
        for (int i = 0; i < N; ++i) {
            ChebyshevExpansion cee = ChebyshevExpansion::factory(40, f, -2, 2);
        }
        auto endTime = std::chrono::system_clock::now();
        auto elap_us = std::chrono::duration<double>(endTime - startTime).count()/N*1e6;
        std::cout << elap_us << " us/call (generation)\n";

        double s = 0; 
        double x = 0.3; 
        N = 10000000;
        startTime = std::chrono::system_clock::now();
        for (int i = 0; i < N; ++i) {
            s += f(x+i*1e-10);
        }
        endTime = std::chrono::system_clock::now();
        elap_us = std::chrono::duration<double>(endTime - startTime).count() / N*1e6;

        std::cout << elap_us << " us/call (f(x)); s: " << s << "\n";
    }

    ChebyshevExpansion cee = ChebyshevExpansion::factory(40, f, -6, 6);
    auto intervals = cee.subdivide(10,10);
    auto roots = cee.real_roots_intervals(intervals);

    long N = 10000;
    Eigen::VectorXd c(50);
    c.fill(1);
    ChebyshevExpansion ce(c);

    auto startTime = std::chrono::system_clock::now();
        mult_by_inplace(ce, 1.001, N);
    auto endTime = std::chrono::system_clock::now();
    auto elap_us = std::chrono::duration<double>(endTime - startTime).count()/N*1e6;
    std::cout << elap_us << " us/call (mult inplace)\n";

    {
        auto intervals = cee.subdivide(20, 4); 
        startTime = std::chrono::system_clock::now();
        for (int i = 0; i < N; ++i) {
            auto roots = cee.real_roots_intervals(intervals);
        }
        endTime = std::chrono::system_clock::now();
        elap_us = std::chrono::duration<double>(endTime - startTime).count()/N*1e6;
        std::cout << elap_us << " us/call (roots inplace)\n";
    }
    
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