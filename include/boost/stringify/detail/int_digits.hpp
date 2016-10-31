#ifndef BOOST_STRINGIFY_DETAIL_INT_DIGITS_HPP
#define BOOST_STRINGIFY_DETAIL_INT_DIGITS_HPP

#include <boost/assert.hpp>
#include <boost/type_traits.hpp>

namespace boost {
namespace stringify {
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
        push(x);
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
} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_DETAIL_INT_DIGITS_HPP

