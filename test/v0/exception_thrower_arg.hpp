#ifndef BOOST_STRINGIFY_TEST_EXCEPTION_THROWER_HPP
#define BOOST_STRINGIFY_TEST_EXCEPTION_THROWER_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/formatter.hpp>

struct exception_tag {};

constexpr exception_tag exception_thrower_arg {};

namespace detail{

template <typename CharT>
class exceptional_formatter: public boost::stringify::v0::formatter<CharT>
{

public:

    template <typename FTuple>
    exceptional_formatter(const FTuple&, exception_tag) noexcept
    {
    }

    std::size_t length() const override
    {
        return 0;
    }

    void write(boost::stringify::v0::output_writer<CharT>&) const override
    {
        throw std::invalid_argument("invalid formatter");
    }

    int remaining_width(int w) const override
    {
        return w;
    }
};

} // namespace detail

template <typename CharT, typename FTuple>
inline detail::exceptional_formatter<CharT>
boost_stringify_make_formatter(const FTuple& ft, exception_tag x)
{
    return {ft, x};
}

#endif
