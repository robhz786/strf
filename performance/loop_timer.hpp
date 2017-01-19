#ifndef LOOP_TIMER_HPP
#define LOOP_TIMER_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <chrono>
#include <string>
#include <functional>
#include <iostream>
#include <iomanip>

class loop_timer
{
public:
    typedef std::chrono::steady_clock clock_type;
    typedef clock_type::duration duration;
    typedef clock_type::time_point time_point;
    typedef clock_type::rep rep;

    struct result
    {
        duration total_duration;
        std::size_t iterations_count;     
    };


    struct default_reporter
    {
        default_reporter(const std::string& label)
            : m_label(label)
        {
        }
        
        void operator()(result r)
        {
            std::chrono::duration<double, std::nano> dur = r.total_duration;
            std::cout
                << std::setprecision(2)
                << std::fixed
                << std::setw(15)
                << dur.count() / (double)r.iterations_count
                << " ns  "
                << m_label
                << std::endl;
        }
        
        std::string m_label;
    };

    
    /**
       \param loop_relative_size duration of loop / resolution of the clock
       \param report_func function to be called at the end
     */
    loop_timer(rep loop_relative_size, const std::function<void(result)>& report)
            : m_report(report)
            , m_loop_relative_size(loop_relative_size)
            , m_it_count(0)
            , m_it_count_stop(8)
            , m_start_time(clock_type::now())
    {
    }

    loop_timer(rep loop_relative_size, const std::string& label)
        : loop_timer(loop_relative_size, default_reporter(label))
    {
    }

    
    ~loop_timer()
    {
        result r = {(m_stop_time - m_start_time), m_it_count};
        m_report(r);
    }

    bool after_each_iteration()
    {
        if(++ m_it_count == m_it_count_stop)
        {
            m_stop_time = clock_type::now();
            return restart_count();
        }
        return true;
    }

    bool restart_count()
    {
        duration dur = m_stop_time - m_start_time;
        if (dur.count() > m_loop_relative_size)
        {
            return false;
        }
        else
        {
            calibrate((double)dur.count());
            m_it_count = 0;
            m_start_time = clock_type::now();
            return true;
        }
    }

    void calibrate(double dur_count)
    {
        if (dur_count < 1000.0)
        {
            m_it_count_stop *= 2;
        }
        else
        {
            m_it_count_stop = std::ceil
                ( 1.4
                * (double)(m_it_count_stop)
                * (double)(m_loop_relative_size)
                / dur_count
                );
        }
    }
    
private:

    std::function<void(result)> m_report;
    rep m_loop_relative_size;
    std::size_t m_it_count;
    std::size_t m_it_count_stop;
    time_point m_stop_time;
    time_point m_start_time;
};


#define BOOST_LOOP_TIMER(size, label)          \
  for(loop_timer loop_timer_instance(size, label);   \
      loop_timer_instance.after_each_iteration() ; )



#endif

