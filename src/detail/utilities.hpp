#pragma once

#include <vector>
#include <chrono>
#include <random>
#include <string>
#include <algorithm>
#include <iomanip>
#include <unistd.h>

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

	class tracer
	{
		struct trace_entry : std::vector<std::pair<std::string, std::string>>{};

	
	private:
		utilities::timer timer;
		std::string pid;
		std::vector<trace_entry> trace;
	public:
		tracer(){
			timer.start();
			pid = std::to_string(getpid());
		}
		void write_event_begin(std::string event_name){
			timer.stop();
			std::string timestamp = std::to_string(timer.get()/1000);
			trace_entry entry;
			entry.push_back({"name", event_name});
			entry.push_back({"ph", "B"});
			entry.push_back({"ts", timestamp});
			entry.push_back({"pid", pid});
			trace.push_back(entry);
		}
		void write_event_end(std::string event_name){
			timer.stop();
			std::string timestamp = std::to_string(timer.get()/1000);
			trace_entry entry;
			entry.push_back({"name", event_name});
			entry.push_back({"ph", "E"});
			entry.push_back({"ts", timestamp});
			entry.push_back({"pid", pid});
			trace.push_back(entry);
		}
		void output_to_console(){
			std::cout << "[ ";
			for(auto& entry : trace){
				std::cout << "{";
				for(auto& str_pair : entry){
					std::cout << '"' << str_pair.first << "\":\"" << str_pair.second << '"';
					if(str_pair != entry.back()) std::cout << ", ";
				}
				std::cout << "}";
				if(entry != trace.back()) std::cout << ",\n";
			}
			std::cout << "\n]\n";
		}
	};

}

inline void print_mnist(double* grid) {
  for (size_t i = 0; i < 28; ++i) {
    for (size_t j = 0; j < 28; ++j) {
        // take up 3 digits
        std::cout << std::setw(3) << (int)grid[i*28 + j] << ' ';
    }
    std::cout << '\n';
  }
}