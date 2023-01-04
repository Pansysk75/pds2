#include <iostream>
#include "detail/knn_structs.hpp"
#include "detail/knn_utils.hpp"
#include "detail/fileio.hpp"

#include <unistd.h>

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << "Usage:\t" << argv[0] << " -f=[filename] -g=random -n=[number of lines] -d=[number of dimensions] [optional: -p to pad with a line on top and a label=0]" << std::endl;
        std::cout << "or:\t" << argv[0] << " -f=[filename] -g=regular -s=[side length] -d=[number of dimensions] [optional: -p to pad with a line on top and a label=0]" << std::endl;
        return 1;
    }

    size_t n=0, s=0, d=0;
    enum { RANDOM, REGULAR } mode;
    std::string filename;
    bool pad = false;

    int opt, opt_got = 0;
    while((opt = getopt(argc, argv, "f:g:n:s:d:p")) != -1)
    {
        // skip first char if it is =
        if(optarg) {
            if(optarg[0] == '=') optarg++;
            std::cout << "Mode given: " << (char)opt << " = " << optarg << std::endl;
        } else {
            std::cout << "Mode given: " << (char)opt << std::endl;
        }

        switch(opt)
        {
            case 'f':
                filename = std::string(optarg);
                break;
            case 'g':
                if (std::string(optarg) == "random"){
                    mode = RANDOM;
                    s=-1;
                }
                else if (std::string(optarg) == "regular") {
                    mode = REGULAR;
                    n = -1;
                }
                else
                {
                    std::cout << "Invalid mode" << std::endl;
                    return 1;
                }
                break;
            case 'n':
                n = std::stoi(optarg);
                break;
            case 's':
                s = std::stoi(optarg);
                break;
            case 'd':
                d = std::stoi(optarg);
                break;
            case 'p':
                pad = true;
                break;
            default:
                std::cout << "Invalid option" << std::endl;
                return 1;
        }
    }

    if(n == 0 || d == 0 || (mode == RANDOM && s != -1) || (mode == REGULAR && n != -1))
    {
        std::cout << "Invalid options" << std::endl;
        return 1;
    }

    if(mode == RANDOM)
    {
        auto [__, corpus] = random_grid(0, n, d);
        vectorToCSV(filename, corpus.Y, n, d, pad);

        std::cout << "Generated random dataset" << std::endl;
    }
    else
    {
        auto [__, corpus, ___] = regular_grid(s, d, 0);
        size_t lines = 1;
        for (size_t i = 0; i < d; i++)
            lines *= s;
        vectorToCSV(filename, corpus.Y, lines, d, pad);

        std::cout << "Generated regular dataset" << std::endl;
    }
}