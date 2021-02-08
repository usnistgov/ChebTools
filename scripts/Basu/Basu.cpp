#include <vector>
#include <chrono>
#include <iostream>

using Mat = std::vector<std::vector<double>>;

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

template<typename vectype>
double Clenshaw2D(const Mat &a, double x, double y, vectype &b){
    std::size_t m = a.size()-1;
    std::size_t n = a[0].size()-1;
    for (auto i = 0; i < b.size(); ++i){
        b[i] = Clenshaw1D(a[i], y);
    }
    return Clenshaw1D(b, x);
}

int main(){
    Mat a;
    int M = 6;
    for (auto i = 0; i <= M; ++i){
        a.push_back(std::vector<double>(M+1, 0.0)); // One would normally use Eigen here, but the challenge is to use only standard library elements...
    }
    {
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
        std::cout << elap_us << " us/call. value:" << (r / N) << " val: " << std::endl;
    }
}