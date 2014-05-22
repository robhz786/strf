#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <boost/timer/timer.hpp>
#include <boost/string_ini_list.hpp>



static char buff[1000000];

static long common_multiplier = 500000;
static long number_of_repetitions = 1000 * common_multiplier;

using boost::timer::cpu_times;
using boost::timer::cpu_timer;
using boost::timer::nanosecond_type;

double time1;
double time2;
double time3;
double time4;
double time5;


class loop_manager
{
public:
  loop_manager(double & _elapsed_time_per_iteration, long _number_of_iterations):
    elapsed_time_per_iteration(_elapsed_time_per_iteration),
    number_of_iterations(_number_of_iterations),
    counter(0)
  {
  }

  bool loop_condition() const
  {
    return (counter < number_of_iterations);
  }

  void advance_iteration()
  {
    ++counter;
  }

  ~loop_manager()
  {
    cpu_times total_elapse_time =  timer.elapsed();
    elapsed_time_per_iteration 
      = static_cast<double>(total_elapse_time.system + total_elapse_time.user)
      / static_cast<double>(number_of_iterations);
  }
  
private:

  double& elapsed_time_per_iteration;
  long number_of_iterations;
  long counter;
  cpu_timer timer;
 };




#define MEASURE_TIME_OF_LOOP(ELAPSED_TIME_RECEIVER, NUMBER_OF_REPETITIONS) \
  for(loop_manager lman(ELAPSED_TIME_RECEIVER, NUMBER_OF_REPETITIONS);     \
      lman.loop_condition();                                               \
      lman.advance_iteration())

#define MEASURE_TIME1 MEASURE_TIME_OF_LOOP(time1, number_of_repetitions)
#define MEASURE_TIME2 MEASURE_TIME_OF_LOOP(time2, number_of_repetitions)
#define MEASURE_TIME3 MEASURE_TIME_OF_LOOP(time3, number_of_repetitions)
#define MEASURE_TIME4 MEASURE_TIME_OF_LOOP(time4, number_of_repetitions)
#define MEASURE_TIME5 MEASURE_TIME_OF_LOOP(time5, number_of_repetitions)


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
  number_of_repetitions = 1000 * common_multiplier;

  {
    MEASURE_TIME1
    {
      sprintf(buff, "hello"); 
    }
    MEASURE_TIME2
    {
      sprintf(buff, "%s", "hello"); 
    }
    MEASURE_TIME3
    {
      strcpy(buff, "hello"); 
    }
    MEASURE_TIME4
    {
      write_hello(buff);
    }
    number_of_repetitions = 100 * common_multiplier;
    MEASURE_TIME5
    {
      write(buff, boost::string_il{"hello"});
    }
    std::cout << time1 << "  " 
              << time2 << "  " 
              << time3 << "  " 
              << time4 << "  " 
              << time5 << std::endl;
  }



  number_of_repetitions = 100 * common_multiplier;
  {
    std::string hello("hello");
    std::string fmt("%s");
    const char* hello_cstr = hello.c_str();
    const char* fmt_cstr = fmt.c_str();


    MEASURE_TIME1
    {
      sprintf(buff, hello_cstr); 
    }
    MEASURE_TIME2
    {
      sprintf(buff, fmt_cstr, hello_cstr); 
    }
    MEASURE_TIME3
    {
      write(buff, boost::string_il{hello_cstr});
    }
    std::cout << time1 << "  " << time2 << "  " << time3 << std::endl;
  }






  return 1;
}
