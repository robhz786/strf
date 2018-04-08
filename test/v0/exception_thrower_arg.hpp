#ifndef BOOST_STRINGIFY_TEST_EXCEPTION_THROWER_HPP
#define BOOST_STRINGIFY_TEST_EXCEPTION_THROWER_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/basic_types.hpp>

struct exception_tag {};

constexpr exception_tag exception_thrower_arg {};

namespace detail{

template <typename CharT>
class exceptional_printer: public boost::stringify::v0::printer<CharT>
{

public:

    exceptional_printer( exception_tag) noexcept
    {
    }

    std::size_t length() const override
    {
        return 0;
    }

    void write() const override
    {
        throw std::invalid_argument("invalid printer");
    }

    int remaining_width(int w) const override
    {
        return w;
    }
};

// struct exception_tag_input_traits
// {
//     template <typename CharT, typename FTuple>
//     static inline detail::exceptional_printer<CharT>
//     make_printer(const FTuple& ft, exception_tag x)
//     {
//         return {ft, x};
//     }
// };

} // namespace detail

//detail::exception_tag_input_traits stringify_get_input_traits(exception_tag x);


BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <typename CharT, typename FTuple>
inline ::detail::exceptional_printer<CharT>
stringify_make_printer
    ( const stringify::v0::output_writer<CharT>&
    , const FTuple&
    , exception_tag x )
{
    return {x};
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif
