#ifndef BOOST_STRINGIFY_WIDTH_T_HPP
#define BOOST_STRINGIFY_WIDTH_T_HPP

#include <limits>

namespace boost {
namespace stringify {

class width_t
{
public:

    typedef long scalar_type;

    constexpr width_t(scalar_type scalar = 0.0)
        : m_value(scalar * unit_value)
    {
    }
    
    constexpr width_t(scalar_type nominator, scalar_type denominator)
        : m_value(nominator * unit_value / denominator)
    {
    }

    constexpr width_t(const width_t&) = default;
    
    constexpr width_t& operator=(const width_t&) = default;

    constexpr bool operator==(const width_t& other) const
    {
        return m_value == other.m_value;
    }

    constexpr bool operator<(const width_t& other) const
    {
        return m_value < other.m_value;
    }

    constexpr bool operator<=(const width_t& other) const
    {
        return m_value <= other.m_value;
    }

    constexpr bool operator>(const width_t& other) const
    {
        return m_value > other.m_value;
    }

    constexpr bool operator>=(const width_t& other) const
    {
        return m_value >= other.m_value;
    }

    constexpr width_t& operator+=(const width_t& other)
    {
        m_value += other.m_value;
        return *this;
    }

    constexpr width_t& operator-=(const width_t& other)
    {
        m_value -= other.m_value;
        return *this;
    }

    constexpr width_t& operator*=(scalar_type scalar)
    {
        m_value *= scalar;
        return *this;
    }

    constexpr width_t& operator/=(scalar_type scalar)
    {
        m_value /= scalar;
        return *this;
    }

    constexpr scalar_type operator/(width_t other) const
    {
        return m_value / other.m_value;
    }
    
    constexpr width_t& operator++()
    {
        m_value += unit_value;
        return *this;
    }

    constexpr width_t operator++(int)
    {
        width_t backup(*this);
        this->operator++();
        return backup;
    }

    constexpr width_t& operator--()
    {
        m_value += unit_value;
        return *this;
    }

    constexpr width_t operator--(int)
    {
        width_t backup(*this);
        this->operator--();
        return backup;
    }
    
    constexpr scalar_type units() const
    {
        return m_value / unit_value;
    }
    
    static constexpr scalar_type unit_value = 16 * 9 * 25 * 7;
    static constexpr scalar_type max_scalar
      = (std::numeric_limits<scalar_type>::max)() / unit_value;

private:
    scalar_type m_value;
};

inline boost::stringify::width_t
operator+(boost::stringify::width_t a, boost::stringify::width_t b)
{
    return a += b;
}

inline boost::stringify::width_t
operator-(boost::stringify::width_t a, boost::stringify::width_t b)
{
    return a -= b;    
}

inline boost::stringify::width_t
operator*(boost::stringify::width_t w, boost::stringify::width_t::scalar_type s)
{
    return w *= s;    
}

inline boost::stringify::width_t
operator/(boost::stringify::width_t w, boost::stringify::width_t::scalar_type s)
{
    return w /= s;    
}

} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_WIDTH_T_HPP

