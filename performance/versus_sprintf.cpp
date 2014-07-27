#include <iostream>
//#include <iomanip>
#include <stdio.h>
//#include <string.h>
#include <climits>
#include <boost/timer/timer.hpp>
#include <boost/listf.hpp>
#include "loop_timer.hpp"

#define PRINT_BENCHMARK(label)  \
  BOOST_LOOP_TIMER(300, boost::timer::print_mean_time(label))

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
  char* char_ptr_output = buff;

  std::cout << std::endl 
            << "Copy from a string literal:" 
            << std::endl;

  PRINT_BENCHMARK("char_ptr_output << listf{\"hello\"}")
  {
    char_ptr_output << listf{"hello"};
  }
  PRINT_BENCHMARK("write_hello(char_ptr_output)")
  {
    write_hello(char_ptr_output);
  }
  PRINT_BENCHMARK("sprintf(char_ptr_output, \"hello\")")
  {
    sprintf(char_ptr_output, "hello");
  }
  PRINT_BENCHMARK("sprintf(char_ptr_output, \"%s\", \"hello\")")
  {
    sprintf(char_ptr_output, "%s", "hello");
  }
  PRINT_BENCHMARK("strcpy(char_ptr_output, \"hello\")")
  {
    strcpy(char_ptr_output, "hello");
  }

  std::cout << std::endl
            << "Copy from a heap allocated short string:"
            << std::endl;
  {
    std::string std_string_hello("hello");
    const char* hello = std_string_hello.c_str();

    std::string std_string_fmt("%s");
    const char* fmt = std_string_fmt.c_str();

    PRINT_BENCHMARK("char_ptr_output << listf{hello}")
    {
      char_ptr_output << listf{hello};
    }
    PRINT_BENCHMARK("strcpy(char_ptr_output, hello)")
    {
      strcpy(char_ptr_output, hello);
    }
    PRINT_BENCHMARK("sprintf(char_ptr_output, hello)")
    {
      sprintf(char_ptr_output, hello);
    }
    PRINT_BENCHMARK("sprintf(char_ptr_output, \"%s\", hello)")
    {
      sprintf(char_ptr_output, "%s", hello);
    }
    PRINT_BENCHMARK("sprintf(char_ptr_output, fmt, hello)")
    {
      sprintf(char_ptr_output, fmt, hello);
    }
  }

  std::cout << std::endl 
            << "Copy two strings" 
            << std::endl;
  PRINT_BENCHMARK("char_ptr_output << listf{\"hello\", \"hello\"}")
  {
    char_ptr_output << listf{"hello", "hello"};
  }
  PRINT_BENCHMARK("sprintf(char_ptr_output, \"%s%s\", \"hello\", \"hello\")")
  {
    sprintf(char_ptr_output, "%s%s", "hello", "hello");
  }

  std::cout << std::endl
            << "Copy a long string ( 1000 characters ):"
            << std::endl;
  {
    std::string std_string_long_string(1000, 'x');
    const char* long_string = std_string_long_string.c_str();

    std::string std_string_fmt("%s");
    const char* fmt = std_string_fmt.c_str();

    PRINT_BENCHMARK("char_ptr_output << listf{long_string}")
    {
      char_ptr_output << listf{long_string};
    }
    PRINT_BENCHMARK("strcpy(char_ptr_output, long_string)")
    {
      strcpy(char_ptr_output, long_string);
    }
    PRINT_BENCHMARK("sprintf(char_ptr_output, long_string)")
    {
      sprintf(char_ptr_output, long_string);
    }
    PRINT_BENCHMARK("sprintf(char_ptr_output, \"%s\", long_string)")
    {
      sprintf(char_ptr_output, "%s", long_string);
    }
    PRINT_BENCHMARK("sprintf(char_ptr_output, fmt, long_string)")
    {
      sprintf(char_ptr_output, fmt, long_string);
    }
  }
  std::cout << std::endl
            << "write integers" 
            << std::endl;
  PRINT_BENCHMARK("char_ptr_output << listf{25}")
  {
    char_ptr_output << listf{25};
  }
  PRINT_BENCHMARK("char_ptr_output << listf{25, 25, 25}")
  {
    char_ptr_output << listf{25, 25, 25};
  }
  PRINT_BENCHMARK("char_ptr_output << listf{INT_MAX}")
  {
    char_ptr_output << listf{INT_MAX};
  }
  PRINT_BENCHMARK("char_ptr_output << listf{INT_MAX, INT_MAX, INT_MAX}")
  {
    char_ptr_output << listf{INT_MAX, INT_MAX, INT_MAX};
  }
  PRINT_BENCHMARK("char_ptr_output << listf{\"ten =  \", 10, \", twenty = \", 20}")
  {
    char_ptr_output << listf{"ten =  ", 10, ", twenty = ", 20};
  }
  PRINT_BENCHMARK("sprintf(char_ptr_output, \"%d\", 25)")
  {
    sprintf(char_ptr_output, "%d", 12345);
  }
  PRINT_BENCHMARK("sprintf(char_ptr_output, \"%d%d%d\", 25, 25)")
  {
    sprintf(char_ptr_output, "%d%d%d", 25, 25, 25);
  }
  PRINT_BENCHMARK("sprintf(char_ptr_output, \"%d\", INT_MAX)")
  {
    sprintf(char_ptr_output, "%d", INT_MAX);
  }
  PRINT_BENCHMARK("sprintf(char_ptr_output, \"%d%d%d\", INT_MAX, INT_MAX, INT_MAX)")
  {
    sprintf(char_ptr_output, "%d%d%d", INT_MAX, INT_MAX, INT_MAX);
  }
  PRINT_BENCHMARK("sprintf(char_ptr_output, \"ten = %d, twenty= %d\", 10, 20)")
  {
    sprintf(char_ptr_output, "ten = %d, twenty= %d", 10, 20);
  }

  return 1;
}
