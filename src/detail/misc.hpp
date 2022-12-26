#include <vector>
#include <chrono>
#include <random>
#include <string>
#include <algorithm>

namespace misc
{

	class random_vector_generator
	{

	private:
		std::mt19937 mersenne_engine{42};

	public:
		std::vector<double> get_doubles(size_t size, double _min = 1.0, double _max = 1024.0)
		{

			std::uniform_real_distribution<double> dist_double{_min, _max};

			auto gen = [this, &dist_double]()
			{
				return dist_double(mersenne_engine);
			};

			std::vector<double> vec(size);
			std::generate(vec.begin(), vec.end(), gen);
			return vec;
		}

		std::vector<int> get_ints(size_t size, int _min = 0, int _max = 1024)
		{
			// returns random ints between (and including) min and max.
			std::uniform_int_distribution<int> dist_int{_min, _max};

			auto gen = [this, &dist_int]()
			{
				return dist_int(mersenne_engine);
			};

			std::vector<int> vec(size);
			std::generate(vec.begin(), vec.end(), gen);
			return vec;
		}
	};

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