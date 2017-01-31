#ifndef BOOST_STRINGIFY_ARG_FORMAT_COMMON_HPP
#define BOOST_STRINGIFY_ARG_FORMAT_COMMON_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/custom_showpos.hpp>
#include <boost/stringify/custom_width.hpp>
#include <boost/stringify/custom_alignment.hpp>

namespace boost {
namespace stringify {

// curiously recurring template pattern 
template <typename T>
class arg_format_common
{
    using showpos_tag = boost::stringify::showpos_tag;
    using width_tag = boost::stringify::width_tag;
    using width_t = boost::stringify::width_t;
    using alignment = boost::stringify::alignment;

public:


    template <typename FTuple>
    width_t get_width(const FTuple& fmt) const noexcept
    {
        width_t w = arg_width();
        return w >= 0 ? w : get_facet<width_tag>(fmt).width();
    }


    template <typename FTuple>
    alignment get_alignment(const FTuple& fmt) const noexcept
    {
        if (flag('>')) 
        {
            return alignment::right;
        }
        else if (flag('<'))
        {                        
            return alignment::left;
        }
        else if (flag('='))
        {
            return alignment::internal;
        }
        else
        {
            return get_facet<alignment_tag>(fmt).value();
        }
    }


    template <typename FTuple>
    bool get_showpos(const FTuple& fmt) const noexcept
    {
        if (flag('-'))
        {
            return false;
        }
        else if (flag('+'))
        {
            return true;
        }
        else
        {
            return get_facet<showpos_tag>(fmt).value();
        }
    }

protected:

    template <typename Tag, typename FTuple>
    decltype(auto) get_facet(const FTuple& fmt) const noexcept
    {
        return fmt.template get<Tag, typename T::input_type>();
    }
    
    constexpr bool flag(char flag) const
    {
        return static_cast<const T*>(this)->flags.has_char(flag);
    }

    constexpr width_t arg_width() const
    {
        return static_cast<const T*>(this)->width;
    }
    
};


} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_ARG_FORMAT_COMMON_HPP

