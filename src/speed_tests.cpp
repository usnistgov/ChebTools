#include "ChebTools/ChebTools.h"
#include <chrono>
#include <map>

using namespace ChebTools;

double ChebyshevExpansion::real_roots_time(long N) {
    auto startTime = std::chrono::system_clock::now();
    for (int i = 0; i < N; ++i) {
        auto ypts = real_roots();
    }
    auto endTime = std::chrono::system_clock::now();
    auto elap_us = std::chrono::duration<double>(endTime - startTime).count() / N*1e6;
    return elap_us;
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
std::map<std::string, double> evaluation_speed_test(ChebyshevExpansion &cee, const vectype &xpts, long N) {
    std::map<std::string, double> output;
    vectype ypts;

    auto startTime = std::chrono::system_clock::now();
    for (int i = 0; i < N; ++i) {
        ypts = cee.y_recurrence_xscaled(xpts);
    }
    auto endTime = std::chrono::system_clock::now();
    auto elap_us = std::chrono::duration<double>(endTime - startTime).count() / N*1e6;
    output["recurrence[vector]"] = elap_us;

    startTime = std::chrono::system_clock::now();
    for (int i = 0; i < N; ++i) {
        ypts = cee.y_Clenshaw_xscaled(xpts);
    }
    endTime = std::chrono::system_clock::now();
    elap_us = std::chrono::duration<double>(endTime - startTime).count() / N*1e6;
    output["Clenshaw[vector]"] = elap_us;

    startTime = std::chrono::system_clock::now();
    ypts.resize(xpts.size());
    for (int i = 0; i < N; ++i) {
        for (int n = static_cast<int>(xpts.size()) - 1; n >= 0; --n) {
            ypts(n) = cee.y_recurrence(xpts(n));
        }
    }
    endTime = std::chrono::system_clock::now();
    elap_us = std::chrono::duration<double>(endTime - startTime).count() / N*1e6;
    output["recurrence[1x1]"] = elap_us;

    startTime = std::chrono::system_clock::now();
    ypts.resize(xpts.size());
    for (int i = 0; i < N; ++i) {
        for (int n = static_cast<int>(xpts.size()) - 1; n >= 0; --n) {
            ypts(n) = cee.y_Clenshaw(xpts(n));
        }
    }
    endTime = std::chrono::system_clock::now();
    elap_us = std::chrono::duration<double>(endTime - startTime).count() / N*1e6;
    output["Clenshaw[1x1]"] = elap_us;
    return output;
}

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
        auto elap_us = std::chrono::duration<double>(endTime - startTime).count() / Nrepeats*1e6;
        results.row(i) << static_cast<double>(Nvec[i]), elap_us, maxeig;
    }
    return results;
}