#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <string.h>
#include <climits>
#include <boost/timer/timer.hpp>
#include <boost/rose.hpp>
#include "loop_timer.hpp"

#define PRINT_BENCHMARK(label)  \
  BOOST_LOOP_TIMER(300, boost::timer::print_mean_time(label))

int main()
{
  using namespace boost::rose;
  char buff[1000000];
  char* char_ptr_output = buff;

  // std::string std_string_hello("hello");
  // const char* hello = std_string_hello.c_str();


  std::cout << std::endl
            << "write integers" 
            << std::endl;
  PRINT_BENCHMARK("char_ptr_output << argf(25)")
  {
    char_ptr_output << argf(25);
  }
  PRINT_BENCHMARK("char_ptr_output << listf{25}")
  {
    char_ptr_output << listf{25};
  }
  PRINT_BENCHMARK("char_ptr_output << argf(INT_MAX)")
  {
    char_ptr_output << argf(INT_MAX);
  }
  PRINT_BENCHMARK("char_ptr_output << listf{INT_MAX}")
  {
    char_ptr_output << listf{INT_MAX};
  }
  PRINT_BENCHMARK("char_ptr_output << argf(LLONG_MAX)")
  {
    char_ptr_output << argf(LLONG_MAX);
  }
  PRINT_BENCHMARK("char_ptr_output << listf{LLONG_MAX}")
  {
    char_ptr_output << listf{LLONG_MAX};
  }



  return 0;
}
