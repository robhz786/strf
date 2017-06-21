#ifndef BOOST_STRINGIFY_V1_DETAIL_CHAR_FLAGS_HPP
#define BOOST_STRINGIFY_V1_DETAIL_CHAR_FLAGS_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

namespace boost{
namespace stringify{
inline namespace v1 {

template <char ...>
class char_flags;

template <>
class char_flags<>
{
    
public:
    
    constexpr char_flags(const char*) : m_bits(0)
    {
    }

    constexpr char_flags() : m_bits(0)
    {
    }
    
    constexpr bool has_char(char c) const
    {
        return false;
    }

    constexpr char_flags(const char_flags&) = default;

    char_flags& operator=(const char_flags&) = default;
    
protected:

    constexpr static int mask(char c)
    {
        return 0;
    }
    
    int m_bits;
};


template <char Char, char ... OtherChars>
class char_flags<Char, OtherChars ...> : private char_flags<OtherChars ...>
{
    
    typedef char_flags<OtherChars ...> parent;
    
    static_assert(sizeof...(OtherChars) <= 8 * sizeof(int), "too many chars");
    
public:

    constexpr char_flags()
    {
    }

    constexpr char_flags(const char_flags& other) = default;

    char_flags& operator=(const char_flags& other) = default;
    
    constexpr char_flags(const char* str)
    {
        for (std::size_t i = 0; str[i] != '\0'; ++i)
        {
            this->m_bits |= mask(str[i]);
        }

    }

    constexpr bool has_char(char ch) const
    {
        return 0 != (this->m_bits & mask(ch));
    }
    
protected:

    using parent::m_bits;
    
    constexpr static int mask(char ch)
    {
        return ch == Char ? this_mask() : parent::mask(ch);
    }
    
    constexpr static int this_mask()
    {
        return 1 << sizeof...(OtherChars);
    }
    
    constexpr static bool has_char(const char* str, char ch)
    {
        return *str != 0 && (*str == ch || has_char(str + 1, ch));
    }
};
    

} // inline namespace v1
} // namespace stringify
} // namespace boost


#endif  /* BOOST_STRINGIFY_V1_DETAIL_CHAR_FLAGS_HPP */

