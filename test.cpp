#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <time.h>

using std::cout;
using std::endl;

auto time2String = [](struct timespec t_start, struct timespec t_end) -> const std::string {
    std::string result;
    std::stringstream ss;
    long double temp = 0.0;
    temp += (t_end.tv_sec - t_start.tv_sec);
    temp += static_cast<long double>((t_end.tv_nsec - t_start.tv_nsec) / 1000u) / static_cast<long double>(1000000.0);
    ss.precision(6);
    ss.setf(std::ios::fixed);
    ss << temp  ;
    ss >> result;
    return result;
};

int main(int argc, char **argv)
{  
    if (argc != 2){
        std::cerr << "Please run as: cpp_test 1000" << std::endl \
                  << "with 1000 means loop count" << std::endl;
        return -1;
    }

    std::size_t loop_count = stoul(std::string(argv[1]));
    volatile std::size_t i = 0; 

    std::vector<float> vfloat(loop_count, 5.5f);
    std::vector<double> vdouble(loop_count, 5.5);

    struct timespec time_start;
    if (0 != clock_gettime(CLOCK_THREAD_CPUTIME_ID, &time_start)){
        std::cerr << "get time wrong" << std::endl;
    }

    for (; i < loop_count; ++i){
        vfloat[i] += 2.3f;
        vfloat[i] -= 3.4f;
        vfloat[i] *= 4.5f;
        vfloat[i] /= 5.6f;
    }

    struct timespec time_1;
    if (0 == clock_gettime(CLOCK_THREAD_CPUTIME_ID, &time_1)){
        std::cout << "float  +-*/ time: " << time2String(time_start, time_1) << "s" << std::endl;
    }

    i = 0; 
    for (; i < loop_count; ++i){
        vdouble[i] += 2.3;
        vdouble[i] -= 3.4;
        vdouble[i] *= 4.5;
        vdouble[i] /= 5.6;
    }
    struct timespec time_2;
    if (0 == clock_gettime(CLOCK_THREAD_CPUTIME_ID, &time_2)){
        std::cout << "double +-*/ time: " << time2String(time_1, time_2) << "s" << std::endl;
    }
    
}
