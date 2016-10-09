#ifndef BOOST_STRINGIFY_FMT_WIDTH_CALCULATOR_HPP
#define BOOST_STRINGIFY_FMT_WIDTH_CALCULATOR_HPP

#include <boost/stringify/type_traits.hpp>

namespace boost {
namespace stringify {
namespace detail
{

template <typename charT>
class length_accumulator
{
public:
    
    length_accumulator() : m_sum(0)
    {
    }

    length_accumulator(const length_accumulator&) = delete;
        
    typedef boost::stringify::width_t width_type;
    
    void add(const charT*, std::size_t len)
    {
        m_sum += len;
    }
    void add(charT)
    {
        ++ m_sum;
    }

    /**
       Ensures that result() will stay lower than limit

       retval true successfully added
       retval false nothing added because reach limit 
     */
    bool add(const charT* str, std::size_t len, width_type limit)
    {
        if(len + (std::size_t)m_sum.units() >= (std::size_t)limit.units())
        {
            return false;
        }
        m_sum += len;
        return true;
    }
    
    bool add(charT, width_type limit)
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

template <typename charT> struct ftype_width_calculator;

template
    < typename charT
    , template <class> class Filter = boost::stringify::accept_any_type
    >
struct fimpl_width_as_length
{
    typedef boost::stringify::ftype_width_calculator<charT> fmt_type;
    template <typename T> using accept_input_type = Filter<T>;
    typedef boost::stringify::detail::length_accumulator<charT> accumulator_type;
};

template <typename charT>
struct ftype_width_calculator
{
    typedef fimpl_width_as_length<charT> default_impl;
};

} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_FMT_WIDTH_CALCULATOR_HPP

