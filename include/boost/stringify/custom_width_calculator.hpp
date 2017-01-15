#ifndef BOOST_STRINGIFY_FMT_WIDTH_CALCULATOR_HPP
#define BOOST_STRINGIFY_FMT_WIDTH_CALCULATOR_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/type_traits.hpp>
#include <boost/stringify/ftuple.hpp>

namespace boost {
namespace stringify {
namespace detail
{


template <typename CharT>
class length_accumulator
{
public:
    
    length_accumulator() : m_sum(0)
    {
    }

    length_accumulator(const length_accumulator&) = delete;
        
    typedef boost::stringify::width_t width_type;
    
    void add(const CharT*, std::size_t len)
    {
        m_sum += len;
    }
    void add(CharT)
    {
        ++ m_sum;
    }

    /**
       Ensures that result() will stay lower than limit

       retval true successfully added
       retval false nothing added because reach limit 
     */
    bool add(const CharT* str, std::size_t len, width_type limit)
    {
        //if(len + (std::size_t)m_sum.units() >= (std::size_t)limit.units())
        if(len + (std::size_t)m_sum >= (std::size_t)limit)
        {
            return false;
        }
        m_sum += len;
        return true;
    }
    
    bool add(CharT, width_type limit)
    {
        if(m_sum + 1 >= limit)
        {
            return false;
        }
        ++ m_sum;
        return true;
    }
    
    void reset()
    {
        m_sum = 0;
    }

    width_type result() const
    {
        return m_sum;
    }
        
private:
    
    width_type m_sum = 0;
};

} //namespace detail

template <typename CharT> struct width_calculator_tag;

template
    < typename CharT
    , template <class> class Filter = boost::stringify::accept_any_type
    >
struct fimpl_width_as_length
{
    typedef boost::stringify::width_calculator_tag<CharT> category;
    template <typename T> using accept_input_type = Filter<T>;
    typedef boost::stringify::detail::length_accumulator<CharT> accumulator_type;
};

template <typename CharT>
struct width_calculator_tag
{
    typedef fimpl_width_as_length<CharT> default_impl;
};

template <typename Formatting, typename InputType, typename CharT>
using width_accumulator = typename boost::stringify::ftuple_get_return_type
    < Formatting
    , boost::stringify::width_calculator_tag<CharT>
    , InputType
    > :: accumulator_type;

} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_FMT_WIDTH_CALCULATOR_HPP

