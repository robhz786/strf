#ifndef BOOST_STRINGIFY_MAKE_STRING_HPP
#define BOOST_STRINGIFY_MAKE_STRING_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

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
        = boost::stringify::arg_list_stringifier
            < typename StringType::value_type
            , boost::stringify::ftuple<Formaters...>
            , writer_type
            >;

    template <typename ... Formaters>
    static auto make_string(const Formaters& ... fmts)
    {
        return input_args_getter_type<Formaters...>(writer_type(), fmts ...);
    }

    template <typename ... Formaters>
    static auto make_string(const boost::stringify::ftuple<Formaters ...>& fmts)
    {
        return input_args_getter_type<Formaters...>(writer_type(), fmts);
    }

};


} // namespace detail


template <typename ... Formaters>
auto make_string(const Formaters& ... fmts)
{
    return boost::stringify::detail::make_string_helper<std::string>
        ::make_string(fmts ...);
}

template <typename ... Formaters>
auto make_wstring(const Formaters& ... fmts)
{
    return boost::stringify::detail::make_string_helper<std::wstring>
        ::make_string(fmts ...);
}

template <typename ... Formaters>
auto make_u16string(const Formaters& ... fmts)
{
    return boost::stringify::detail::make_string_helper<std::u16string>
        ::make_string(fmts ...);
}

template <typename ... Formaters>
auto make_u32string(const Formaters& ... fmts)
{
    return boost::stringify::detail::make_string_helper<std::u32string>
        ::make_string(fmts ...);
}

} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_MAKE_STRING_HPP

