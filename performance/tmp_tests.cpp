#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <string.h>
#include <climits>
#include <boost/timer/timer.hpp>
#include <boost/listf.hpp>
#include "loop_timer.hpp"

#define PRINT_BENCHMARK(label)  \
  BOOST_LOOP_TIMER(300, boost::timer::print_mean_time(label))

int main()
{
  using boost::listf;
  char buff[1000000];
  char* char_ptr_output = buff;

  // std::string std_string_hello("hello");
  // const char* hello = std_string_hello.c_str();

  PRINT_BENCHMARK("sprintf(char_ptr_output, \"%s%s\", \"hello\", \"hello\")")
  {
    sprintf(char_ptr_output, "%s%s", "hello", "hello");
  }

  PRINT_BENCHMARK("delete new int(0)")
  {
    delete new int(0);
  }

  return 0;
}
