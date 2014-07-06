#include <boost/timer/timer.hpp>

static boost::timer::nanosecond_type calcule_user_time_resolution()
{
  boost::timer::cpu_timer cpu;
  boost::timer::cpu_times elapsed;
  do
  {
    elapsed = cpu.elapsed();
  }
  while(elapsed.user == 0);
  return elapsed.user;
}

static boost::timer::nanosecond_type calcule_system_time_resolution()
{
  boost::timer::cpu_timer cpu;
  boost::timer::cpu_times elapsed;
  do
  {
    elapsed = cpu.elapsed();
  }
  while(elapsed.system == 0);
  return elapsed.system;
}

static boost::timer::nanosecond_type get_user_time_resolution()
{
  static boost::timer::nanosecond_type t = calcule_user_time_resolution();
  return t;
}

static boost::timer::nanosecond_type get_system_time_resolution()
{
  static boost::timer::nanosecond_type t = calcule_system_time_resolution();
  return t;
}

class loop_manager
{
public:

  loop_manager(double & elapsed_time_per_iteration):
    elapsed_time_per_iteration_ref(elapsed_time_per_iteration),
    result_good_enough(false),
    num_iteratios_sample(1),
    iterations_counter(0),
    usr_time_resolution(get_user_time_resolution()),
    sys_time_resolution(get_system_time_resolution())
  {
    timer.start();
  }

  bool loop_condition() const
  {
    return ! result_good_enough;
  }

  void advance_iteration()
  {
    if(++iterations_counter == num_iteratios_sample)
    {
      boost::timer::cpu_times sample_times = timer.elapsed();
      const long resolution_factor = 100;

      if(sample_times.user < resolution_factor * usr_time_resolution &&
         sample_times.system < resolution_factor * sys_time_resolution)
      {
        // num_iteratios_sample is not great enought
        if(sample_times.user == 0 && sample_times.system == 0)
          num_iteratios_sample *= resolution_factor;
        else
          if(sample_times.user > sample_times.system)
          {
            num_iteratios_sample *= resolution_factor * usr_time_resolution;
            num_iteratios_sample /= sample_times.user;
          }
          else
          {
            num_iteratios_sample *= resolution_factor * sys_time_resolution;
            num_iteratios_sample /= sample_times.user;
          }
      }
      else                             
      {
        double num_its = static_cast<double>(num_iteratios_sample);
        mean_usr_time = static_cast<double>(sample_times.user) / num_its;
        mean_sys_time = static_cast<double>(sample_times.system) / num_its;
        result_good_enough = true;                     
      }

      iterations_counter = 0;
      timer.start();
    }
  }

  ~loop_manager()
  {
    elapsed_time_per_iteration_ref = mean_usr_time + mean_sys_time;
  }
  
private:

  double mean_usr_time;
  double mean_sys_time;

  double& elapsed_time_per_iteration_ref;
  bool result_good_enough;
  long num_iteratios_sample;
  long iterations_counter;
  boost::timer::nanosecond_type usr_time_resolution;
  boost::timer::nanosecond_type sys_time_resolution;
  boost::timer::cpu_timer timer;

};


#define MEASURE_TIME(output_variable)     \
  for(loop_manager lman(output_variable); \
      lman.loop_condition();              \
      lman.advance_iteration())

static double times[100];

#define MEASURE_TIME1 MEASURE_TIME(times[1])
#define MEASURE_TIME2 MEASURE_TIME(times[2])
#define MEASURE_TIME3 MEASURE_TIME(times[3])
#define MEASURE_TIME4 MEASURE_TIME(times[4])
#define MEASURE_TIME5 MEASURE_TIME(times[5])

void print_times(int last)
{
  for(int i = 1; i <= last; ++i)
    std::cout << times[i] << " / ";

  std::cout << std::endl;
}

