#ifndef LOOP_TIMER_HPP
#define LOOP_TIMER_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <functional>
#include <iostream>
#include <string>
#include <boost/timer/timer.hpp>
#include <boost/config.hpp>

namespace boost
{
namespace timer
{
  boost::timer::cpu_times get_time_resolutions();

  class loop_timer
  {
  public:
    struct results_type
    {
      results_type():
        number_of_iterations(0)
      {
        total_elapsed_time.user = 0;
        total_elapsed_time.system = 0;
        total_elapsed_time.wall = 0;
      }
      results_type(const results_type& other):
        total_elapsed_time(other.total_elapsed_time),
        number_of_iterations(other.number_of_iterations)
      {
      }
      
      timer::cpu_times total_elapsed_time;
      boost::int_least64_t number_of_iterations;

      double get_mean_user_time() const
      {
        return (static_cast<double>(total_elapsed_time.user)
                /
                static_cast<double>(number_of_iterations));
      } 
      double get_mean_system_time() const
      {
        return (static_cast<double>(total_elapsed_time.system)
                /
                static_cast<double>(number_of_iterations));
      } 
      double get_mean_wall_time() const
      {
        return (static_cast<double>(total_elapsed_time.wall)
                /
                static_cast<double>(number_of_iterations));
      } 
      double get_mean_user_plus_system_time() const
      {
        return (static_cast<double>(total_elapsed_time.user
                                    + 
                                    total_elapsed_time.system)
                /
                static_cast<double>(number_of_iterations));
      } 
    };


    typedef std::function<void(const results_type&)> result_handler_function;

    loop_timer(result_handler_function _result_handler);

    loop_timer(int time_resolution_multiplier, 
               result_handler_function _result_handler);

    ~loop_timer();
  
    bool finished()
    {
      return results_are_good_enough;
    }
  
    void on_end_of_each_iteration()
    {
      if(++sample_iterations_counter == num_iteratios_per_sample)
        on_end_of_each_sample();
    }

  private:

    void on_end_of_each_sample()
    {
      last_sample_duration = timer.elapsed();
      process_last_sample();
      start_new_sample();
    }

    void process_last_sample();

    void start_new_sample()
    {
      sample_iterations_counter = 0;
      timer.start();
    }

    bool last_sample_was_too_small();

    void resize_next_sample();

    void update_accumulated_results();
  
    result_handler_function result_handler;
    results_type accumulated_results;
    boost::timer::cpu_timer timer;
    boost::timer::cpu_times last_sample_duration;
    boost::timer::nanosecond_type resolution;
    boost::timer::nanosecond_type required_minimal_accumulated_duration;
    boost::int_least64_t sample_iterations_counter;
    boost::int_least64_t num_iteratios_per_sample;

    bool results_are_good_enough = false;
  };


  struct print_mean_time
  {
    print_mean_time(const std::string& _label,
                    std::ostream& _output = std::cout):
      label(_label),
      output(_output)
    {
    }

    void operator()(const loop_timer::results_type& results);

    std::string label;
    std::ostream& output;
  };
} // namespace timer
} // namepace boost


#define BOOST_LOOP_TIMER(size, result_handler)                           \
  for(boost::timer::loop_timer boost_loop_timer(size, result_handler);   \
      ! boost_loop_timer.finished();                                     \
      boost_loop_timer.on_end_of_each_iteration())


#endif

