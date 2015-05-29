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
  long long x = LLONG_MIN + 10;
  long long n = 0;

  // std::string std_string_hello("hello");
  // const char* hello = std_string_hello.c_str();


  std::cout << std::endl
            << "write integers" 
            << std::endl;


/*
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
  x = LLONG_MIN + 10;
  n = 0;
  PRINT_BENCHMARK("char_ptr_output << argf(LLONG_MIN)")
  {
    x += (++n % 2) ? +1 : -1;
    char_ptr_output << argf(x);
  }
  x = LLONG_MIN + 10;
  n = 0;
  PRINT_BENCHMARK("char_ptr_output << listf{LLONG_MIN}")
  {
    x += (++n % 2) ? +1 : -1;
    char_ptr_output << listf{x};
  }
*/
  x = 100;
  n = 0;
  PRINT_BENCHMARK("sprintf()")
  {
    x += (++n % 2) ? +1 : -1;
    sprintf(char_ptr_output,
            "%lld%lld%lld%lld%lld%lld%lld%lld%lld%lld",
            x+1, x+2, x+3, x+4, x+5, x+6, x+7, x+8, x+9, x+10);
  }
  x = 100;
  n = 0;
  PRINT_BENCHMARK("listf{}")
  {
    x += (++n % 2) ? +1 : -1; // avoid some "unfair" compiler optimizations ( clang )
    char_ptr_output << listf{x+1 , x+2, x+3, x+4, x+5, x+6, x+7, x+8, x+9, x+10};
  }


//  char_ptr_output << listf{123};
//  std::cout << char_ptr_output  << std::endl;


  return 0;
}














