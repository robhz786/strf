#ifndef BOOST_STRINGIFY_V0_CONVENTIONAL_ARGF_READER_HPP
#define BOOST_STRINGIFY_V0_CONVENTIONAL_ARGF_READER_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/facets/alignment.hpp>

namespace boost {
namespace stringify {
inline namespace v0 {

struct width_tag;
struct showpos_tag;
struct intbase_tag;
struct case_tag;
struct showbase_tag;

template <typename InputType>
struct conventional_argf_reader
{
    template <typename FacetCategory, typename FTuple>
    constexpr static decltype(auto) getf(const FTuple& ft) noexcept
    {
        return ft.template get_facet<FacetCategory, InputType>();
    }
    
    template <typename ArgF, typename FTuple>
    constexpr static int get_width(const ArgF& argf, const FTuple& ft) noexcept
    {
        if(argf.width < 0)
        {
            return getf<boost::stringify::v0::width_tag>(ft).width();
        }
        return argf.width;
    }

    template <typename ArgF, typename FTuple>
    constexpr static boost::stringify::v0::alignment
    get_alignment(const ArgF& argf, const FTuple& ft) noexcept
    {
        if(argf.flags.has_char('>'))
        {
            return alignment::right;
        }
        else if (argf.flags.has_char('<'))
        {                        
            return alignment::left;
        }
        else if (argf.flags.has_char('='))
        {
            return alignment::internal;
        }
        else
        {
            return getf<boost::stringify::v0::alignment_tag>(ft).value();
        }
    }

    template <typename ArgF, typename FTuple>
    constexpr static bool get_showpos(const ArgF& argf, const FTuple& ft) noexcept
    {
        if (argf.flags.has_char('-'))
        {
            return false;
        }
        else if (argf.flags.has_char('+'))
        {
            return true;
        }
        else
        {
            return getf<boost::stringify::v0::showpos_tag>(ft).value();
        }
    }

    template <typename ArgF, typename FTuple>
    constexpr static int get_base(const ArgF& argf, const FTuple& ft) noexcept
    {
        if (argf.flags.has_char('d'))
        {
            return 10;
        }
        if (argf.flags.has_char('x') || argf.flags.has_char('X'))
        {
            return 16;
        }
        else if (argf.flags.has_char('o'))
        {                        
            return 8;
        }
        return getf<boost::stringify::v0::intbase_tag>(ft).value();
    }

    template <typename ArgF, typename FTuple>
    constexpr static bool get_uppercase(const ArgF& argf, const FTuple& ft) noexcept
    {
        if (argf.flags.has_char('c'))
        {
            return false;
        }
        else if (argf.flags.has_char('C') || argf.flags.has_char('X'))
        {                        
            return true;
        }
        return getf<boost::stringify::v0::case_tag>(ft).uppercase();
    }

    template <typename ArgF, typename FTuple>
    constexpr static bool get_showbase(const ArgF& argf, const FTuple& ft) noexcept
    {
        if (argf.flags.has_char('$'))
        {
            return false;
        }
        else if (argf.flags.has_char('#'))
        {                        
            return true;
        }
        return getf<boost::stringify::v0::showbase_tag>(ft).value();
    } 
};

} // inline namespace stringify
} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_V0_CONVENTIONAL_ARGF_READER_HPP

