#include <vector>
#include <chrono>
#include <random>
#include <string>
#include <algorithm>

namespace utilities
{

	class timer
	{

	private:
		std::chrono::high_resolution_clock::time_point t1;
		std::chrono::high_resolution_clock::time_point t2;

	public:
		void start() { t1 = std::chrono::high_resolution_clock::now(); }
		void stop() { t2 = std::chrono::high_resolution_clock::now(); }
		double get() { return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count(); }
	};

}

void print_mnist(double* grid) {
  for (size_t i = 0; i < 28; ++i) {
    for (size_t j = 0; j < 28; ++j) {
        // take up 3 digits
        std::cout << std::setw(3) << (int)grid[i*28 + j] << ' ';
    }
    std::cout << '\n';
  }
}