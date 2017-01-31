#ifndef BOOST_STRINGIFY_STRINGIFIER_HPP_INCLUDED
#define BOOST_STRINGIFY_STRINGIFIER_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/assert.hpp>
#include <boost/stringify/custom_fill.hpp>
#include <boost/stringify/custom_width.hpp>
#include <boost/stringify/custom_width_calculator.hpp>
#include <cstddef>

namespace boost
{
namespace stringify 
{

template <typename CharT, typename Output, typename FTuple>
class stringifier
{
public:
  virtual ~stringifier()
    {}

    /**
       return the amount of character that needs to be allocated,
       not counting the termination character. The implementer is
       not required to calculate the exact value if this is 
       difficult. But it must be greater or equal. And should not
       be much greater.
    */
    virtual std::size_t length() const = 0;

    virtual void write(Output&) const = 0;
};

} //namespace stringify
} //namespace boost


#endif






