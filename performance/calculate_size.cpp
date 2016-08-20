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
/*    
  using boost::stringify::listf;

  PRINT_BENCHMARK("listf<char>{\"hello\"}.minimal_length()")
  {
    listf<char>{"hello"}.minimal_length();
  }
  PRINT_BENCHMARK("listf<char>{\"hello\", \"hello\"}.minimal_length()")
  {
    listf<char>{"hello", "hello"}.minimal_length();
  }
  std::string std_string_with_1000_chars(1000, 'x');
  const char* string_with_1000_chars = std_string_with_1000_chars.c_str();

  PRINT_BENCHMARK("listf<char>{string_with_1000_chars}.minimal_length()")
  {
    listf<char>{string_with_1000_chars}.minimal_length();
  }
  PRINT_BENCHMARK("listf<char>{std_string_with_1000_chars}.minimal_length()")
  {
    listf<char>{std_string_with_1000_chars}.minimal_length();
  }
  PRINT_BENCHMARK("listf<char>{25}.minimal_length()")
  {
    listf<char>{25}.minimal_length();
  }
  PRINT_BENCHMARK("listf<char>{25, 25, 25}.minimal_length()")
  {
    listf<char>{25, 25, 25}.minimal_length();
  }
  PRINT_BENCHMARK("listf<char>{INT_MAX}.minimal_length()")
  {
    listf<char>{INT_MAX}.minimal_length();
  }
  PRINT_BENCHMARK("listf<char>{INT_MAX, INT_MAX, INT_MAX}.minimal_length()")
  {
    listf<char>{INT_MAX, INT_MAX, INT_MAX}.minimal_length();
  }
*/
  return 0;
}
