//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <string.h>
#include <climits>
#include <boost/timer/timer.hpp>
#include <boost/stringify.hpp>
#include "loop_timer.hpp"

#define PRINT_BENCHMARK(label)  \
  BOOST_LOOP_TIMER(900, boost::timer::print_mean_time(label))


int main()
{
  using namespace boost::stringify;
  char buff[1000000];
  char* char_ptr_output = buff;

  writef(char_ptr_output) () (LLONG_MAX);
  std::cout << char_ptr_output << std::endl;
  writef(char_ptr_output) () (ULLONG_MAX);
  std::cout << char_ptr_output << std::endl;

  
  PRINT_BENCHMARK("writef(char_ptr_output)()(LLONG_MAX)")
  {
      writef(char_ptr_output) () (std::numeric_limits<long long>::max());
          
  }

  PRINT_BENCHMARK("writef(char_ptr_output)()(LLONG_MAX, LLONG_MAX, LLONG_MAX)")
  {
      writef(char_ptr_output) ()
          ( std::numeric_limits<long long>::max()
          , std::numeric_limits<long long>::max()
          , std::numeric_limits<long long>::max()
          );          
  }

  PRINT_BENCHMARK("writef(char_ptr_output)()(LLONG_MAX, LLONG_MAX, LLONG_MAX, LLONG_MAX, LLONG_MAX, LLONG_MAX)")
  {
      writef(char_ptr_output) ()
          ( std::numeric_limits<long long>::max()
          , std::numeric_limits<long long>::max()
          , std::numeric_limits<long long>::max()
          , std::numeric_limits<long long>::max()
          , std::numeric_limits<long long>::max()
          , std::numeric_limits<long long>::max()
          );          
  }

  PRINT_BENCHMARK("sprintf(\"%lld%lld%lld%lld%lld%lld\", LLONG_MAX, LLONG_MAX, LLONG_MAX, LLONG_MAX, LLONG_MAX, LLONG_MAX)")
  {
      sprintf( char_ptr_output
             , "%lld%lld%lld%lld%lld%lld"
             , LLONG_MAX, LLONG_MAX, LLONG_MAX, LLONG_MAX, LLONG_MAX, LLONG_MAX);
  }

  
  PRINT_BENCHMARK("writef(char_ptr_output)()(INT_MAX)")
  {
      writef(char_ptr_output) () (std::numeric_limits<int>::max());
  }
  
  PRINT_BENCHMARK("writef(char_ptr_output)()(INT_MAX, INT_MAX, INT_MAX)")
  {
      writef(writef(writef(char_ptr_output) () (INT_MAX)) () (INT_MAX)) () (INT_MAX);
          
  }

  PRINT_BENCHMARK("writef(char_ptr_output)()(INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX)")
  {
      writef(char_ptr_output) () (
          std::numeric_limits<int>::max(),
          std::numeric_limits<int>::max(),
          std::numeric_limits<int>::max(),
          std::numeric_limits<int>::max(),
          std::numeric_limits<int>::max(),
          std::numeric_limits<int>::max());
  }
  

  PRINT_BENCHMARK("writef(char_ptr_output)()(25, 25, 25)")
  {
      writef(char_ptr_output) () (25, 25, 25);
  }

  PRINT_BENCHMARK("writef(char_ptr_output)()(25)")
  {
      writef(char_ptr_output) () (25);
  }

  
 


  return 0;
}














