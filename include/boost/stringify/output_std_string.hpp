#ifndef BOOST_STRINGIFY_OUTPUT_STD_STRING_HPP
#define BOOST_STRINGIFY_OUTPUT_STD_STRING_HPP

namespace boost {
namespace stringify {
namespace detail {


template <typename StringType>
class string_maker
{
public:

    typedef typename StringType::value_type char_type;
    
    string_maker() = default;

    string_maker(const string_maker&) = delete;

    string_maker(string_maker&&) = default;
    
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
    
    StringType finish()
    {
        return std::move(m_out);
    }

    void reserve(std::size_t size)
    {
        m_out.reserve(m_out.capacity() + size);
    }
    
private:

    StringType m_out;               
};


template <class StringType>
struct make_string_helper
{
    typedef
        boost::stringify::detail::string_maker<StringType>
        writer_type;

    template <typename ... Formaters>
    using input_args_getter_type
        = boost::stringify::detail::final_writer
            < typename StringType::value_type
            , boost::stringify::formater_tuple<Formaters...>
            , writer_type
            >;

    template <typename ... Formaters>
    static auto make_string(const Formaters& ... fmts)
    {
        return input_args_getter_type<Formaters...>(writer_type(), fmts ...);
    }
};


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


template
    < typename StringType
    , typename CharT = typename StringType::value_type
    , typename Traits = typename StringType::traits_type
    >  
auto satisfies_basic_string_output_concept(StringType& str)
    -> decltype
    ( str.push_back(CharT())
    , str.append(CharT(), std::size_t())
    , str.append((const CharT*)(0), std::size_t())
    , str.reserve(std::size_t())
    , std::true_type()
    )
{
    return std::true_type();
}

    
template <typename T>
std::false_type satisfies_basic_string_output_concept(...)
{
    return std::false_type();
}

// template <StringType>
// using satisfies_basic_string_output_concept
// = decltype
//     ( boost::stringify::detail::satisfies_basic_string_output_concept_func
//       ( std::declval<StringType>())
//     )  

// template <StringType>
// constexpr bool satisfies_basic_string_output_concept_v
//     = satisfies_basic_string_output_concept<StringType>::value;

} // namespace detail


template <typename StringType>
auto appendf(StringType& str)
-> typename std::enable_if
    < decltype
        (boost::stringify::detail::satisfies_basic_string_output_concept(str)
        )::value
    , boost::stringify::writef_helper
        < boost::stringify::detail::string_appender<StringType>
        >
    >::type
{
    return
        boost::stringify::writef_helper
        < boost::stringify::detail::string_appender<StringType>
        >
        (str);
}


template <typename StringType>
auto assignf(StringType& str)
-> typename std::enable_if
    < decltype
        (boost::stringify::detail::satisfies_basic_string_output_concept(str)
        )::value
    , boost::stringify::writef_helper
        < boost::stringify::detail::string_appender<StringType>
        >
    >::type
{
    str.clear();
    return
        boost::stringify::writef_helper
        < boost::stringify::detail::string_appender<StringType>
        >
        (str);
}

template <typename ... Formaters>
auto make_string(const Formaters& ... fmts)
{
    return boost::stringify::detail::make_string_helper<std::string>
        ::make_string(fmts ...);
}


} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_OUTPUT_STD_STRING_HPP

