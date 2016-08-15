#include <iostream>
#include <stdio.h>
#include <climits>
#include <boost/timer/timer.hpp>
#include <boost/stringify.hpp>
#include "loop_timer.hpp"

#define PRINT_BENCHMARK(label)  \
  BOOST_LOOP_TIMER(300, boost::timer::print_mean_time(label))

int main()
{
  using boost::stringify::listf;

  PRINT_BENCHMARK("listf{\"hello\"}.minimal_length()")
  {
    listf{"hello"}.minimal_length();
  }
  PRINT_BENCHMARK("listf{\"hello\", \"hello\"}.minimal_length()")
  {
    listf{"hello", "hello"}.minimal_length();
  }
  std::string std_string_with_1000_chars(1000, 'x');
  const char* string_with_1000_chars = std_string_with_1000_chars.c_str();

  PRINT_BENCHMARK("listf{string_with_1000_chars}.minimal_length()")
  {
    listf{string_with_1000_chars}.minimal_length();
  }
  PRINT_BENCHMARK("listf{std_string_with_1000_chars}.minimal_length()")
  {
    listf{std_string_with_1000_chars}.minimal_length();
  }
  PRINT_BENCHMARK("listf{25}.minimal_length()")
  {
    listf{25}.minimal_length();
  }
  PRINT_BENCHMARK("listf{25, 25, 25}.minimal_length()")
  {
    listf{25, 25, 25}.minimal_length();
  }
  PRINT_BENCHMARK("listf{INT_MAX}.minimal_length()")
  {
    listf{INT_MAX}.minimal_length();
  }
  PRINT_BENCHMARK("listf{INT_MAX, INT_MAX, INT_MAX}.minimal_length()")
  {
    listf{INT_MAX, INT_MAX, INT_MAX}.minimal_length();
  }

  return 0;
}
