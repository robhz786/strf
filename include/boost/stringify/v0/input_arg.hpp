#ifndef BOOST_STRINGIFY_V0_INPUT_ARG_HPP
#define BOOST_STRINGIFY_V0_INPUT_ARG_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/detail/stringifier_wrapper.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <typename CharT, typename FTuple>
class input_arg
{
    template <class T>
    using trait = decltype(boost_stringify_input_traits_of(std::declval<const T>()));
                                                         
    template <class T>
    using stringifier = typename trait<T>::template stringifier<CharT, FTuple>;

    template <typename S>
    using wrapper = boost::stringify::v0::detail::stringifier_wrapper_impl<S, FTuple>;
    
public:
    
    template <typename T, typename S = stringifier<T>>
    input_arg(const T& arg1, wrapper<S> && strf = wrapper<S>())
        : m_stringifier(strf)
    {
        strf.set_args(arg1);
    }
   
    template <typename T, typename S = stringifier<T>>
    input_arg
        ( const T& arg1
        , const typename S::second_arg& arg2
        , wrapper<S> && strf = wrapper<S>() 
        )
        : m_stringifier(strf)
    {
        strf.set_args(arg1, arg2);
    }

    std::size_t length(const FTuple& fmt) const
    {
        return m_stringifier.length(fmt);
    }

    void write(boost::stringify::v0::output_writer<CharT>& out, const FTuple& fmt) const
    {
        return m_stringifier.write(out, fmt);
    }

    int remaining_width(int w, const FTuple& fmt) const
    {
        return m_stringifier.remaining_width(w, fmt);
    }
    
private:

    boost::stringify::v0::detail::stringifier_wrapper<CharT, FTuple>& m_stringifier;

};

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  /* BOOST_STRINGIFY_V0_INPUT_ARG_HPP */

