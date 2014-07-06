#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <boost/timer/timer.hpp>
#include <boost/listf.hpp>
#include "measure_time.hpp"


void write_hello(char* out)
{
  *out   = 'h';
  *++out = 'e';
  *++out = 'l';
  *++out = 'l';
  *++out = 'o';
  *++out = '\0';
}

int main()
{
  using boost::listf;
  char buff[1000000];

  {
    MEASURE_TIME1  buff << listf{"hello"};
    MEASURE_TIME2  write_hello(buff);
    MEASURE_TIME3  sprintf(buff, "hello");
    MEASURE_TIME4  sprintf(buff, "%s", "hello");
    MEASURE_TIME5  strcpy(buff, "hello");
    print_times(5);
  }
  {
    std::string std_string_hello("hello");
    const char* hello = std_string_hello.c_str();

    std::string std_string_fmt("%s");
    const char* fmt = std_string_fmt.c_str();

    MEASURE_TIME1  buff << listf{hello};
    MEASURE_TIME2  sprintf(buff, hello);
    MEASURE_TIME3  sprintf(buff, "%s", hello);
    MEASURE_TIME4  sprintf(buff, fmt, hello);
    MEASURE_TIME5  strcpy(buff, hello);
    print_times(5);
  }
  {
    int x = 12345;
    std::string std_string_fmt("%d");
    const char* fmt = std_string_fmt.c_str();
    MEASURE_TIME1  buff << listf{x};
    MEASURE_TIME2  sprintf(buff, "%d", 12345);
    MEASURE_TIME3  sprintf(buff, "%d", x);
    MEASURE_TIME4  sprintf(buff, fmt, x);
    print_times(4);
  }
  {
    MEASURE_TIME1  buff << listf{12345, "  ", 12345};
    MEASURE_TIME2  sprintf(buff, "%d  %d", 12345, 12345);
    print_times(2);
  }

  {
    MEASURE_TIME1  buff << listf{"ten =  ", 10, ", twenty = ", 20};
    MEASURE_TIME2  sprintf(buff, "ten = %d, twenty= %d", 10, 20);
    print_times(2);
  }



  return 1;
}
