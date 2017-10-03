#ifndef BOOST_STRINGIFY_V0_STRINGIFIER_HPP
#define BOOST_STRINGIFY_V0_STRINGIFIER_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/output_writer.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <typename CharT>
class stringifier
{
public:

    virtual ~stringifier()
    {
    }

    virtual std::size_t length() const = 0;
        
    virtual void write(stringify::v0::output_writer<CharT>& out) const = 0;

    virtual int remaining_width(int w) const = 0;
};

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_STRINGIFIER_HPP

