#ifndef BOOST_STRINGIFY_V1_INPUT_ARG_HPP
#define BOOST_STRINGIFY_V1_INPUT_ARG_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v1/detail/stringifier_wrapper.hpp>

namespace boost {
namespace stringify {
inline namespace v1 {

template <class T>
struct input_traits;

namespace detail {

template <typename InputType>
struct failed_input_traits
{
    template <typename C, typename O, typename F>
    struct stringifier
    {
        static_assert(sizeof(C) == 0,
                      "unable to find stringifier for this input type");
    };
};

} // namespace detail

template <typename CharT, typename Output, typename FTuple>
class input_arg
{
    template <typename T>
    static decltype(boost_stringify_input_traits_of(std::declval<const T>()))
    input_traits_of(const T&);

    template <typename T>
    static boost::stringify::v1::detail::failed_input_traits<T>
    input_traits_of(...);
    
    template <class T>
    using input_traits = decltype(input_traits_of<T>(std::declval<const T>()));

    template <class T>
    using stringifier = typename input_traits<T>        
        :: template stringifier<CharT, Output, FTuple>;

    template <typename T>
    using stringifier_wrapper_of =
        boost::stringify::v1::detail::stringifier_wrapper_impl
        <stringifier<T>, CharT, Output, FTuple>;

    using stringifier_wrapper
        = boost::stringify::v1::detail::stringifier_wrapper<CharT, Output, FTuple>;

    stringifier_wrapper& empty_wrapper()
    {
        static stringifier_wrapper ewrapper;
        return ewrapper;
    }
    
public:

    input_arg()
        : m_stringifier(empty_wrapper())
    {
    }
    
    template <typename T>
    input_arg
        ( const T* value
        , stringifier_wrapper_of<const T*> && strf = stringifier_wrapper_of<const T*>()
        ) noexcept
        : m_stringifier(strf)
    {
        strf.set_args(value);
    }

    
    template <typename T>
    input_arg
        ( const T& value
        , stringifier_wrapper_of<T> && strf = stringifier_wrapper_of<T>()
        ) noexcept
        : m_stringifier(strf)
    {
        strf.set_args(value);
    }

    template <typename T>
    input_arg
        ( const T* value
        , const typename stringifier<const T*>::arg_format_type& arg_format
        , stringifier_wrapper_of<const T*> && strf = stringifier_wrapper_of<const T*>() 
        ) noexcept
        : m_stringifier(strf)
    {
        strf.set_args(value, arg_format);
    }

    
    template <typename T>
    input_arg
        ( T&& value
        , const typename stringifier<T>::arg_format_type& arg_format
        , stringifier_wrapper_of<T> && strf = stringifier_wrapper_of<T>() 
        ) noexcept
        : m_stringifier(strf)
    {
        strf.set_args(value, arg_format);
    }

    ~input_arg()
    {
    }
    
    std::size_t length(const FTuple& fmt) const
    {
        return m_stringifier.length(fmt);
    }

    void write(Output& out, const FTuple& fmt) const
    {
        return m_stringifier.write(out, fmt);
    }

    int remaining_width(int w, const FTuple& fmt) const
    {
        return m_stringifier.remaining_width(w, fmt);
    }
    
private:

    stringifier_wrapper& m_stringifier; 
};


} // inline namespace v1
} // namespace stringify
} // namespace boost

#endif  /* BOOST_STRINGIFY_V1_DETAIL_STR_WRITE_REF_HPP */

