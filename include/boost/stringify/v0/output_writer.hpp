#ifndef BOOST_STRINGIFY_V0_OUTPUT_WRITER_HPP
#define BOOST_STRINGIFY_V0_OUTPUT_WRITER_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

namespace boost {
namespace stringify {
inline namespace v0 {
namespace detail {


} // namespace detail


template <typename CharT>
class output_writer
{
public:

    using char_type = CharT;
    
    virtual ~output_writer()
    {
    }

    virtual void put(CharT ch) = 0;
    
    virtual void put(const CharT* str, std::size_t size) = 0;
/*    
    virtual void repeat(CharT ch, std::size_t repeat) = 0;

    virtual void repeat(CharT ch1, CharT ch2, std::size_t repeat) = 0;

    virtual void repeat(CharT ch1, CharT ch2, CharT ch3, std::size_t repeat) = 0;
    
    virtual void repeat(CharT ch1, CharT ch2, CharT ch3, CharT ch4, std::size_t repeat) = 0;
*/

    virtual void repeat(CharT ch, std::size_t repeat)
    {
        for(; repeat > 0; --repeat)
        {
            put(ch);
        }
    }

    virtual void repeat(CharT ch1, CharT ch2, std::size_t repeat)
    {
        for(; repeat > 0; --repeat)
        {
            put(ch1);
            put(ch2);
        }
    }

    virtual void repeat(CharT ch1, CharT ch2, CharT ch3, std::size_t repeat)
    {
        for(; repeat > 0; --repeat)
        {
            put(ch1);
            put(ch2);
            put(ch3);
        }
    }
    
    virtual void repeat(CharT ch1, CharT ch2, CharT ch3, CharT ch4, std::size_t repeat)
    {
        for(; repeat > 0; --repeat)
        {
            put(ch1);
            put(ch2);
            put(ch3);
            put(ch4);
        }
    }

};

} // namespace v0
} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_V0_OUTPUT_WRITER_HPP

