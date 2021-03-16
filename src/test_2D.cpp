#include "ChebTools2D/ChebTools2D.hpp"
#include <chrono>

void one(const int Nx, const int Ny){
	using namespace ChebTools::TwoD;

	// https://www.chebfun.org/examples/approx2/PrettyFunctions.html
	//auto f = [](auto x, auto y){ return cos(10.0*(x*x+y))*sin(10.0*(x+y*y)); };

	auto f = [](auto x, auto y) { return x * x * x * sin(-x * x) * cos(x) * exp(-x * x - y * y); };

	std::tuple<std::size_t, std::size_t> orders = { Nx - 1, Ny - 1 };

	using MatType = Eigen::Matrix<double, -1, -1>;
	using ExpType = ChebyshevExpansion2D<MatType>;
	auto e = ExpType::factory<double>(f, orders, ChebyshevExpansion2DBounds<double>{});

	volatile auto x = 0.7, y = 0.3, ff = 0.0;
	int Nrep = 10000;
	auto tic0 = std::chrono::steady_clock::now();
	for (auto ii = 0; ii < Nrep; ++ii){
		ff += e.eval_Clenshaw(x, y);
	}
	auto tic1 = std::chrono::steady_clock::now();
	auto fchk = f(x, y);

	std::cout << Nx << "x" << Ny << ": " << std::chrono::duration<double>(tic1 - tic0).count()/Nrep*1e6 << " us; error: " << ff/Nrep/fchk-1 << std::endl;
}

int main(){
	one(5, 5);
	one(8, 8);
	one(10, 10);
	one(15, 15);
	one(20, 20);
	return EXIT_SUCCESS;
}