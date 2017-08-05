#ifndef BOOST_STRINGIFY_V0_DETAIL_INT_DIGITS_HPP
#define BOOST_STRINGIFY_V0_DETAIL_INT_DIGITS_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/assert.hpp>
#include <boost/type_traits.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN
namespace detail{

template <typename intT, int Base>
class int_digits
{
public:
    int_digits(intT x)
        : m_size(0)
    {
        while(x >= Base)
        {
            push(x % 10);
            x /= Base;
        }
        push(static_cast<unsigned>(x));
    }

    int pop()
    {
        if (m_size)
        {
            return m_digits[--m_size];
        }
        return 0;
    }

    bool empty() const
    {
        return m_size == 0;
    }
    
    unsigned size() const
    {
        return m_size;
    }
    
private:
    
    void push(unsigned digit)
    {
        m_digits[m_size] = digit;
        ++m_size;
    }
    
    unsigned m_size;
    static constexpr int Capacity = (Base == 16 ? 2 : 3) * sizeof(intT);
    unsigned char m_digits[Capacity];
};

} // namespace detail
BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_INT_DIGITS_HPP

