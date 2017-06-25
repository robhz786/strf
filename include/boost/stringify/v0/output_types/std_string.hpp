#ifndef BOOST_STRINGIFY_V0_OUTPUT_TYPES_STD_STRING_HPP
#define BOOST_STRINGIFY_V0_OUTPUT_TYPES_STD_STRING_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

namespace boost {
namespace stringify {
inline namespace v0 {
namespace detail {

template <typename StringType>
class string_appender
{
public:
    typedef typename StringType::value_type char_type;
    
    string_appender(StringType& out)
        : m_out(out)
    {
    }

    string_appender(const string_appender&) = default;
    
    void put(char_type character)
    {
        m_out.push_back(character);
    }

    void put(char_type character, std::size_t repetitions)
    {
        m_out.append(repetitions, character);
    }

    void put(const char_type* str, std::size_t count)
    {
        m_out.append(str, count);
    }
    
    StringType& finish()
    {
        return m_out;
    }

    void reserve(std::size_t size)
    {
        m_out.reserve(m_out.capacity() + size);
    }

private:

    StringType& m_out;               
};
/*
template <class T>
struct std_string_destination_concept
{

private:

    template
        < typename S
        , typename CharT = typename S::value_type
        , typename Traits = typename S::traits_type
        >  
    static auto test(S* str) -> decltype
             ( str->push_back(CharT())
             , str->append(CharT(), std::size_t())
             , str->append((const CharT*)(0), std::size_t())
             , str->reserve(std::size_t())
             , std::true_type()
             );

    template <typename S>
    static std::false_type test(...);

public:

    static constexpr bool value = decltype(test<T>((T*)0))::value;

};
*/
} // namespace detail


template <typename CharT, typename Traits>
auto append_to(std::basic_string<CharT, Traits>& str)
{
    using string_type = std::basic_string<CharT, Traits>;
    using writer = boost::stringify::v0::detail::string_appender<string_type>;
    return boost::stringify::v0::make_args_handler<writer, string_type&>(str);
}


template <typename CharT, typename Traits>
auto assign_to(std::basic_string<CharT, Traits>& str)
{
    using string_type = std::basic_string<CharT, Traits>;
    str.clear();
    using writer = boost::stringify::v0::detail::string_appender<string_type>;
    return boost::stringify::v0::make_args_handler<writer, string_type&>(str);
}


} // inline namespace v0
} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_V0_OUTPUT_TYPES_STD_STRING_HPP

