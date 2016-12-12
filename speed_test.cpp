#include <algorithm>
#include <functional>
#include <vector>
#include <chrono>
#include <iostream>
#include <map>

#include "Eigen/Dense"

Eigen::MatrixXd eigs_speed_test(std::vector<std::size_t> &Nvec, std::size_t Nrepeats) {
    Eigen::MatrixXd results(Nvec.size(), 3);
    for (std::size_t i = 0; i < Nvec.size(); ++i)
    {
        Eigen::MatrixXd A = Eigen::MatrixXd::Random(Nvec[i], Nvec[i]);
        double maxeig;
        auto startTime = std::chrono::system_clock::now();
            for (int i = 0; i < Nrepeats; ++i) {
                maxeig = A.eigenvalues().real().minCoeff();
            }
        auto endTime = std::chrono::system_clock::now();
        auto elap_us = std::chrono::duration<double>(endTime - startTime).count()/Nrepeats*1e6;
        results.row(i) << Nvec[i], elap_us, maxeig;
    }
    return results;
}

int main(){

    std::vector<std::size_t> NxN;
    for (std::size_t i=2; i < 64; i *= 2){
        NxN.push_back(i);
    }
    Eigen::MatrixXd res = eigs_speed_test(NxN, 10000);
    for (std::size_t i =0; i < res.rows(); ++i){
        std::cout << res(i,0) << ": " << res(i,1) << " us\n";
    }
    return EXIT_SUCCESS;
}