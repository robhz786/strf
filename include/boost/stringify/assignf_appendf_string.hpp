#ifndef BOOST_STRINGIFY_ASSIGNF_APPENDF_STRING_HPP
#define BOOST_STRINGIFY_ASSIGNF_APPENDF_STRING_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

namespace boost {
namespace stringify {
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

template <class T>
struct std_string_output_concept
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

} // namespace detail


template <typename StringType>
auto appendf(StringType& str) -> typename std::enable_if
    < boost::stringify::detail::std_string_output_concept<StringType>::value
    , boost::stringify::writef_helper
        < boost::stringify::detail::string_appender<StringType> >
    >::type
{
    return
        boost::stringify::writef_helper
        < boost::stringify::detail::string_appender<StringType> >
        (str);
}


template <typename StringType>
auto assignf(StringType& str) -> typename std::enable_if
    < boost::stringify::detail::std_string_output_concept<StringType>::value
    , boost::stringify::writef_helper
        < boost::stringify::detail::string_appender<StringType> >
    >::type
{
    str.clear();
    return boost::stringify::writef_helper
        < boost::stringify::detail::string_appender<StringType> >
        (str);
}
                
} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_ASSIGNF_APPENDF_STRING_HPP

