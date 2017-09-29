#ifndef BOOST_STRINGIFY_TEST_EXCEPTION_THROWER_HPP
#define BOOST_STRINGIFY_TEST_EXCEPTION_THROWER_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/stringifier.hpp>

struct exception_tag {};

constexpr exception_tag exception_thrower_arg {};

namespace detail{

template <typename CharT>
class exceptional_stringifier: public boost::stringify::v0::stringifier<CharT>
{

public:

    template <typename FTuple>
    exceptional_stringifier(const FTuple&, exception_tag) noexcept
    {
    }

    std::size_t length() const override
    {
        return 0;
    }

    void write(boost::stringify::v0::output_writer<CharT>&) const override
    {
        throw std::invalid_argument("invalid stringifier");
    }

    int remaining_width(int w) const override
    {
        return w;
    }
};

struct exceptional_stringifier_traits
{
    template <typename CharT, typename>
    using stringifier = exceptional_stringifier<CharT>;    
};


}

detail::exceptional_stringifier_traits boost_stringify_input_traits_of(exception_tag);


#endif
