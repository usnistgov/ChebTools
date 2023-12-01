#include <vector>
#include <chrono>
#include <iostream>
#include <Eigen/Dense>

template<typename vectype>
double Clenshaw1D(const vectype &c, double ind){
    int N = static_cast<int>(c.size()) - 1;
    double u_k = 0, u_kp1 = 0, u_kp2 = 0;
    for (int k = N; k >= 0; --k){
        // Do the recurrent calculation
        u_k = 2.0*ind*u_kp1 - u_kp2 + c[k];
        if (k > 0){
            // Update the values
            u_kp2 = u_kp1; u_kp1 = u_k;
        }
    }
    return (u_k - u_kp2)/2;
}

/// With STL datatypes
template<typename Mat, typename vectype>
double Clenshaw2D(const Mat& a, double x, double y, vectype& b) {
    std::size_t m = a.size() - 1;
    std::size_t n = a[0].size() - 1;
    for (auto i = 0; i < b.size(); ++i) {
        b[i] = Clenshaw1D(a[i], y);
    }
    return Clenshaw1D(b, x);
}

template<typename MatType, int Cols = MatType::ColsAtCompileTime>
auto Clenshaw1DByRow(const MatType& c, double ind) {
    int N = static_cast<int>(c.rows()) - 1;
    static Eigen::Array<double, 1, Cols> u_k, u_kp1, u_kp2;
    // Not statically sized    
    if constexpr (Cols < 0) {
        int M = c.rows();
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
    return (u_k - u_kp2) / 2;
}

/// With Eigen datatypes
template<typename MatType>
double Clenshaw2DEigen(const MatType& a, double x, double y) {
    auto b = Clenshaw1DByRow(a, y);
    return Clenshaw1D(b.matrix(), x);
}

template<int Rows, int Cols>
void test_Eigen(int M){
    using MatType = Eigen::Array<double, Rows, Cols>;
    MatType aa; 
    if constexpr ((Rows < 0) || (Cols < 0)) {
        aa.resize(M + 1, M + 1);
    }
    else{
        aa.resize(M + 1, M + 1);
    }
    aa.fill(0.0);
    for (auto i = 0; i < M + 1; ++i) {
        for (auto j = 0; j < M + 1; ++j) {
            aa(i, j) = i + j;
        }
    }
    int N = 1000 * 1000;
    volatile auto r = 0.0, x = 0.1, y = 0.7;
    auto startTime = std::chrono::system_clock::now();
    for (int i = 0; i < N; ++i) {
        auto v = Clenshaw2DEigen(aa, x, y);
        r += v;
    }
    auto endTime = std::chrono::system_clock::now();
    auto elap_us = std::chrono::duration<double>(endTime - startTime).count() / N * 1e6;
    std::cout << elap_us << " us/call. (Eigen-powered) value:" << (r / N) << std::endl;
}

int main(){
    std::vector<std::vector<double>> a;
    for (auto i = 0; i <= M; ++i){
        a.push_back(std::vector<double>(M+1, 0.0)); // One would normally use Eigen here, but the challenge is to use only standard library elements...
    }
    {
        for (auto i = 0; i < M + 1; ++i) {
            for (auto j = 0; j < M + 1; ++j) {
                a[i][j] = i + j;
            }
        }
        std::vector<double> b(M+1, 0.0);
        int N = 1000 * 1000;
        volatile auto r = 0.0, x = 0.1, y = 0.7;
        auto startTime = std::chrono::system_clock::now();
        for (int i = 0; i < N; ++i) {
            auto v = Clenshaw2D(a, x, y, b);
            r += v;
        }
        auto endTime = std::chrono::system_clock::now();
        auto elap_us = std::chrono::duration<double>(endTime - startTime).count() / N * 1e6;
        std::cout << elap_us << " us/call. value:" << (r / N) << std::endl;
    }
    std::cout  << "Dynamic:" << std::endl;
    test_Eigen<Eigen::Dynamic, Eigen::Dynamic>(M);

    std::cout  << "Static:" << std::endl;
    test_Eigen<M+1, M+1>(M);
}
int main() {
    do_one<11>();
    do_one<13>();
    do_one<17>();
    do_one<21>();
}