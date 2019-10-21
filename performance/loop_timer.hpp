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
    typedef unsigned long long count_type;

    struct result
    {
        duration total_duration;
        count_type iterations_count;
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
                << std::setw(9)
                << total_ns.count() / (double)r.iterations_count
                << " ns | "
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
        , unsigned quantity = 1
        )
        : m_report(report)
        , m_expected_duration(expected_duration)
        , m_it_count(0)
        , m_it_count_stop(1000)
        , m_quantity(quantity)
    {
        start();
    }

    loop_timer
        ( const std::string& label
        , duration expected_duration = std::chrono::seconds{1}
        , unsigned quantity = 1
        )
        : loop_timer(default_reporter(label), expected_duration, quantity)
    {
    }


    ~loop_timer()
    {
        result r = {(m_stop_time - m_start_time) / m_quantity, m_it_count};
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
        if (dur < std::chrono::milliseconds{10})
        {
            // duration too small even for calibration.
            // So calibrate next (or further) time
            m_it_count_stop *= 2;
        }
        else
        {
            auto new_loop_size_f = std::ceil
                ( 1.2
                * static_cast<double>(m_it_count_stop)
                * (m_expected_duration / dur)
                );
            auto new_loop_size = static_cast<count_type>(new_loop_size_f);
            if (new_loop_size < 2 * m_it_count_stop)
            {
                new_loop_size = 2 * m_it_count_stop;
            }
            m_it_count_stop = new_loop_size;
        }
    }

    void start()
    {
        m_it_count = 0;
        m_start_time = clock_type::now();
    }

    std::function<void(result)> m_report;
    std::chrono::duration<double, std::nano> m_expected_duration;
    count_type m_it_count = 0;
    count_type m_it_count_stop = 1000;
    time_point m_stop_time;
    time_point m_start_time;
    unsigned m_quantity = 1;
};

#define LOOP_TIMER(LABEL, TIME)      \
    for(loop_timer loop_timer_obj((LABEL), (TIME));  loop_timer_obj.shall_continue(); )

#define LOOP_TIMER_N(N, LABEL, TIME)                                     \
    for(loop_timer loop_timer_obj((LABEL), (TIME), (N));  loop_timer_obj.shall_continue(); )


#define PRINT_BENCHMARK(LABEL) LOOP_TIMER((LABEL), std::chrono::seconds{10})

#define PRINT_BENCHMARK_N(N, LABEL) LOOP_TIMER_N((N), (LABEL), std::chrono::seconds{10})


#if (defined(__GNUC__) || defined(__clang__))    \

inline void escape(const void* p)
{
    asm volatile("" : : "r,m"(p) : "memory");
    asm volatile("" : : : "memory");
}

inline void escape(void* p)
{
#if defined(__clang__)
    asm volatile("" : "+r,m"(p) : : "memory");
#else
    asm volatile("" : "+m,r"(p) : : "memory");
#endif
    asm volatile("" : : : "memory");
}

inline void clobber()
{
  asm volatile("" : : : "memory");
}

#elif defined(_MSC_VER)

#pragma optimize("", off)

inline void escape_hlp(const volatile char*) {}

#pragma optimize("", on)

inline void escape(const void* p)
{
    escape_hlp(reinterpret_cast<const volatile char*>(p));
    _ReadWriteBarrier();
}

inline  void clobber()
{
    _ReadWriteBarrier();
}

#endif

template <class T>
inline void escape(T& r)
{
    escape(&r);
}

#endif

