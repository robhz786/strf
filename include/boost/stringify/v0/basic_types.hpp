#ifndef BOOST_STRINGIFY_V0_BASIC_TYPES_HPP
#define BOOST_STRINGIFY_V0_BASIC_TYPES_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <system_error>
#include <boost/stringify/v0/config.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <typename CharT>
class output_writer
{
public:

    using char_type = CharT;

    virtual ~output_writer()
    {
    }

    virtual void set_error(std::error_code err) = 0;

    virtual bool good() const = 0;

    virtual bool put(const CharT* str, std::size_t size) = 0;

    virtual bool put(CharT ch) = 0;

    virtual bool repeat(std::size_t count, CharT ch) = 0;

    virtual bool repeat(std::size_t count, CharT ch1, CharT ch2) = 0;

    virtual bool repeat(std::size_t count, CharT ch1, CharT ch2, CharT ch3) = 0;

    virtual bool repeat(std::size_t count, CharT ch1, CharT ch2, CharT ch3, CharT ch4) = 0;
};

template <typename CharT>
class formatter
{
public:

    virtual ~formatter()
    {
    }

    virtual std::size_t length() const = 0;

    virtual void write(stringify::v0::output_writer<CharT>& out) const = 0;

    virtual int remaining_width(int w) const = 0;
};

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_BASIC_TYPES_HPP

