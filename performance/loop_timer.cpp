#include "loop_timer.hpp"
#include <boost/io/ios_state.hpp>

namespace boost{
namespace timer{

static boost::timer::cpu_times calculate_time_resolutions()
{
  boost::timer::cpu_timer cpu;
  boost::timer::cpu_times elapsed;
  boost::timer::cpu_times resolutions;
  do
  {
    elapsed = cpu.elapsed();
    if(elapsed.wall != 0 && resolutions.wall == 0)
      resolutions.wall = elapsed.wall;
    if(elapsed.user != 0 && resolutions.user == 0)
      resolutions.user = elapsed.user;
    if(elapsed.system != 0 && resolutions.system == 0)
      resolutions.system = elapsed.system;
  }
  while (resolutions.user == 0 ||
         resolutions.system == 0 ||
         resolutions.wall == 0);

  return resolutions;
}


boost::timer::cpu_times get_time_resolutions()
{
  static cpu_times t = calculate_time_resolutions();
  return t;
}

loop_timer::loop_timer(result_handler_function _result_handler):
  result_handler(_result_handler),
  resolution(get_time_resolutions().user +
             get_time_resolutions().system),
  required_minimal_accumulated_duration(resolution * 200),
  sample_iterations_counter(0),
  num_iteratios_per_sample(1)
{
  start_new_sample();
}

loop_timer::loop_timer(int time_resolution_multiplier, 
                       result_handler_function _result_handler):
  result_handler(_result_handler),
  resolution(get_time_resolutions().user +
             get_time_resolutions().system),
  required_minimal_accumulated_duration(resolution *
                                        time_resolution_multiplier),
  sample_iterations_counter(0),
  num_iteratios_per_sample(1)
{
  start_new_sample();
}


loop_timer::~loop_timer()
{
  result_handler(accumulated_results);
}


void loop_timer::process_last_sample()
{
  if(last_sample_was_too_small())
    resize_next_sample(); // and discard its results
  else
    update_accumulated_results();
}


bool loop_timer::last_sample_was_too_small()
{
  nanosecond_type duration = (last_sample_duration.user + 
                              last_sample_duration.system);

  return duration * 5 < required_minimal_accumulated_duration;
}


void loop_timer::resize_next_sample()
{
  nanosecond_type duration = (last_sample_duration.user + 
                              last_sample_duration.system);
  if(duration < resolution)
    num_iteratios_per_sample *= (required_minimal_accumulated_duration /
                                 resolution);
  else
    num_iteratios_per_sample *= (required_minimal_accumulated_duration /
                                 duration);

  num_iteratios_per_sample += num_iteratios_per_sample / 10;
}


void loop_timer::update_accumulated_results()
{
  accumulated_results.total_elapsed_time.user   += last_sample_duration.user;
  accumulated_results.total_elapsed_time.system += last_sample_duration.system;
  accumulated_results.total_elapsed_time.wall   += last_sample_duration.wall;
  accumulated_results.number_of_iterations += sample_iterations_counter;

  nanosecond_type user_plus_system = (accumulated_results.total_elapsed_time.user +
                                      accumulated_results.total_elapsed_time.system);
  results_are_good_enough = (user_plus_system >
                             required_minimal_accumulated_duration);
}


void print_mean_time::operator()(const loop_timer::results_type& results)
{
  boost::io::ios_precision_saver ps(output);
  boost::io::ios_width_saver ws(output);

  output.precision(3);
  output.width(8);
  output << results.get_mean_user_plus_system_time()
         << "  "
         << label
         << std::endl;
}

} //namespace boost
} //namespace timer

