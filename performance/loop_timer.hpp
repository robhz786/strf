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
#include <cmath>

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
            std::chrono::duration<double, std::nano>  total_ns = r.total_duration;
            std::cout
                << std::setprecision(2)
                << std::fixed
                << std::setw(15)
                // << total_ms.count()
                // << " ms (total) | "
                // << std::setw(10)
                // << r.iterations_count
                // << " iterations | "
                // << std::setw(10)
                << total_ns.count() / (double)r.iterations_count
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
    loop_timer
        ( const std::function<void(result)>& report
        , duration expected_duration = std::chrono::seconds{1}
        )
        : m_report(report)
        , m_expected_duration(expected_duration)
        , m_it_count(0)
        , m_it_count_stop(1000)
    {
        start();
    }

    loop_timer
        ( const std::string& label
        , duration expected_duration = std::chrono::seconds{1}
        )
        : loop_timer(default_reporter(label), expected_duration)
    {
    }


    ~loop_timer()
    {
        result r = {(m_stop_time - m_start_time), m_it_count};
        m_report(r);
    }

    bool shall_continue()
    {
        if(++ m_it_count == m_it_count_stop)
        {
            m_stop_time = clock_type::now();
            if (duration_is_long_enough())
            {
                return false;
            }
            calibrate();
            start();
        }
        return true;
    }

private:

    bool duration_is_long_enough()
    {
        duration dur = m_stop_time - m_start_time;
        return dur > m_expected_duration;
    }

    void calibrate()
    {
        decltype(m_expected_duration) dur = m_stop_time - m_start_time;
        if (dur < std::chrono::milliseconds{1})
        {
            // duration too small even for calibration.
            // So calibrate next (or further) time
            m_it_count_stop *= 2;
        }
        else
        {
            m_it_count_stop = (std::size_t) std::ceil
                ( 1.2
                * static_cast<double>(m_it_count_stop)
                * (m_expected_duration / dur)
                );
        }
    }

    void start()
    {
        m_it_count = 0;
        std::cout << std::flush;
        std::cout << std::flush;
        fflush(stdout);
        fflush(stdout);
        (void)clock_type::now();
        (void)clock_type::now();
        (void)clock_type::now();
        (void)clock_type::now();
        (void)clock_type::now();

        m_start_time = clock_type::now();
    }

    std::function<void(result)> m_report;
    std::chrono::duration<double, std::nano> m_expected_duration;
    std::size_t m_it_count = 0;
    std::size_t m_it_count_stop = 1000;
    time_point m_stop_time;
    time_point m_start_time;
};

#define LOOP_TIMER(LABEL, TIME)      \
    for(loop_timer loop_timer_obj{LABEL, TIME};  loop_timer_obj.shall_continue(); )


#define PRINT_BENCHMARK(LABEL) LOOP_TIMER(LABEL, std::chrono::seconds{10})


#endif

