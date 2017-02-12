#ifndef BOOST_STRINGIFY_INPUT_ARG_HPP
#define BOOST_STRINGIFY_INPUT_ARG_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/detail/deferred_stringifier_construction.hpp>

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

template <typename CharT, typename Output, typename FTuple>
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
    using stringifier = typename input_traits<T>        
        :: template stringifier<CharT, Output, FTuple>;

    template <typename T>
    using stringifier_constructor =
        boost::stringify::detail::deferred_stringifier_construction_impl
        <stringifier<T>, CharT, Output, FTuple>;
   
public:

    template <typename T>
    input_arg
        ( const T* value
        , stringifier_constructor<const T*> && strf = stringifier_constructor<const T*>()
        ) noexcept
        : m_stringifier_constructor(strf)
    {
        strf.set_args(value);
    }

    
    template <typename T>
    input_arg
        ( const T& value
        , stringifier_constructor<T> && strf = stringifier_constructor<T>()
        ) noexcept
        : m_stringifier_constructor(strf)
    {
        strf.set_args(value);
    }

    template <typename T>
    input_arg
        ( const T* value
        , const typename stringifier<const T*>::arg_format_type& arg_format
        , stringifier_constructor<const T*> && strf = stringifier_constructor<const T*>() 
        ) noexcept
        : m_stringifier_constructor(strf)
    {
        strf.set_args(value, arg_format);
    }

    
    template <typename T>
    input_arg
        ( T&& value
        , const typename stringifier<T>::arg_format_type& arg_format
        , stringifier_constructor<T> && strf = stringifier_constructor<T>() 
        ) noexcept
        : m_stringifier_constructor(strf)
    {
        strf.set_args(value, arg_format);
    }

    ~input_arg()
    {
    }
    
    std::size_t length(const FTuple& fmt) const
    {
        return m_stringifier_constructor.length(fmt);
    }

    void write(Output& out, const FTuple& fmt) const
    {
        return m_stringifier_constructor.write(out, fmt);
    }

private:

    boost::stringify::detail::deferred_stringifier_construction
    <CharT, Output, FTuple>
    & m_stringifier_constructor; 
};


} // namespace stringify
} // namespace boost

#endif  /* BOOST_STRINGIFY_DETAIL_STR_WRITE_REF_HPP */

