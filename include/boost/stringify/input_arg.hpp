#ifndef BOOST_STRINGIFY_INPUT_ARG_HPP
#define BOOST_STRINGIFY_INPUT_ARG_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/stringifier.hpp>

namespace boost {
namespace stringify {

template <class T>
struct input_traits;

namespace detail {

template <typename InputType>
struct failed_intput_traits
{
    template <typename C, typename O, typename F>
    struct stringifier
    {
        static_assert(sizeof(C) == 0,
                      "unable to find stringifier for this input type");
    };
};

} // namespace detail

template <typename CharT, typename Output, typename Formatting>
class input_arg
{
    template <typename T>
    static decltype(boost_stringify_input_traits_of(std::declval<const T>()))
    input_traits_of(const T&);

    template <typename T>
    static boost::stringify::detail::failed_intput_traits<T>
    input_traits_of(...);
    
    template <class T>
    using input_traits = decltype(input_traits_of<T>(std::declval<const T>()));

    template <class T>
    using stringifier_t = typename input_traits<T>        
        :: template stringifier<CharT, Output, Formatting>;
    
public:
    
    template <typename T>
    input_arg
        ( const T& value
        , stringifier_t<T> && s = stringifier_t<T>() // will be stringifier_t<T>(value)
                                                     // after P0145R2 is supported
        )
        noexcept
        : m_stringifier(s)
    {
        s.set(value); // will be removed after compilers add support to P0145R2
    }
   
    template <typename T>
    input_arg
        ( const T& arg
        , const typename stringifier_t<T>::local_formatting& arg_format
        , stringifier_t<T> && s = stringifier_t<T>() // will be stringifier_t<T>(arg, arg_format)
        ) noexcept
        : m_stringifier(s)
    {
        s.set(arg, arg_format);  // will be removed after compilers add support to P0145R2
    }
    
    std::size_t length(const Formatting& fmt) const
    {
        return m_stringifier.length(fmt);
    }

    void write(Output& out, const Formatting& fmt) const
    {
        return m_stringifier.write(out, fmt);
    }
            
private:
    
    const boost::stringify::stringifier<CharT, Output, Formatting>& m_stringifier;
};


} // namespace stringify
} // namespace boost

#endif  /* BOOST_STRINGIFY_DETAIL_STR_WRITE_REF_HPP */

