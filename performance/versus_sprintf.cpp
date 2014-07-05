#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <boost/timer/timer.hpp>
#include <boost/listf.hpp>



static char buff[1000000];

static long common_multiplier = 500000;

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

#define MEASURE_TIME(clock_resolution, output_variable) \
MEASURE_TIME_OF_LOOP(output_variable, clock_resolution* common_multiplier)

#define MEASURE_TIME1(clock_resolution) MEASURE_TIME(clock_resolution, time1)
#define MEASURE_TIME2(clock_resolution) MEASURE_TIME(clock_resolution, time2)
#define MEASURE_TIME3(clock_resolution) MEASURE_TIME(clock_resolution, time3)
#define MEASURE_TIME4(clock_resolution) MEASURE_TIME(clock_resolution, time4)
#define MEASURE_TIME5(clock_resolution) MEASURE_TIME(clock_resolution, time5)




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

  {
    MEASURE_TIME1(100)
    {
      write_hello(buff);
    }
    MEASURE_TIME2(1000)
    {
      write(buff, boost::listf{"hello"});
    }
    MEASURE_TIME3(1000)
    {
      sprintf(buff, "hello"); 
    }
    MEASURE_TIME4(1000)
    {
      sprintf(buff, "%s", "hello"); 
    }
    MEASURE_TIME5(500)
    {
      strcpy(buff, "hello"); 
    }
    std::cout << time1 << " / "  << std::flush
              << time2 << " / "  << std::flush
              << time3 << " / "  << std::flush
              << time4 << " / "  << std::flush
              << time5 << " / "  << std::flush 
              << std::endl;
  }

  {

    std::string std_string_hello("hello");
    const char* hello = std_string_hello.c_str();

    std::string std_string_fmt("%s");
    const char* fmt = std_string_fmt.c_str();


    MEASURE_TIME1(100)
    {
      write(buff, boost::listf{hello});
    }
    MEASURE_TIME2(25)
    {
      sprintf(buff, hello); 
    }
    MEASURE_TIME3(200)
    {
      sprintf(buff, "%s", hello); 
    }
    MEASURE_TIME4(10)
    {
      sprintf(buff, fmt, hello); 
    }
    MEASURE_TIME5(200)
    {
      strcpy(buff, hello); 
    }
    std::cout << time1 << " / "  << std::flush
              << time2 << " / "  << std::flush
              << time3 << " / "  << std::flush
              << time4 << " / "  << std::flush
              << time5 << " / "  << std::flush
              << std::endl;
  }
  {
    int x = 12345;
    std::string std_string_fmt("%d");
    const char* fmt = std_string_fmt.c_str();

    MEASURE_TIME1(40)
    {
      write(buff, boost::listf{x});
    }
    MEASURE_TIME2(100)
    {
      sprintf(buff, "%d", 12345);
    }
    MEASURE_TIME3(10)
    {
      sprintf(buff, "%d", x); 
    }
    MEASURE_TIME4(10)
    {
      sprintf(buff, fmt, x); 
    }

    std::cout << time1 << " / "  << std::flush
              << time2 << " / "  << std::flush
              << time3 << " / "  << std::flush
              << time4 << " / "  << std::flush
              << std::endl;
  }

  return 1;
}
