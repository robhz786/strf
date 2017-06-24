#ifndef BOOST_STRINGIFY_V0_INPUT_ARG_HPP
#define BOOST_STRINGIFY_V0_INPUT_ARG_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/detail/stringifier_wrapper.hpp>

namespace boost {
namespace stringify {
inline namespace v0 {

template <typename Output, typename FTuple>
class input_arg
{
    template <class T>
    using trait = decltype(boost_stringify_input_traits_of(std::declval<const T>()));
                                                         
    template <class T>
    using stringifier = typename trait<T>::template stringifier<Output, FTuple>;

    template <typename S>
    using wrapper = boost::stringify::v0::detail::stringifier_wrapper_impl<S>;
    
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

    void write(Output& out, const FTuple& fmt) const
    {
        return m_stringifier.write(out, fmt);
    }

    int remaining_width(int w, const FTuple& fmt) const
    {
        return m_stringifier.remaining_width(w, fmt);
    }
    
private:

    boost::stringify::v0::detail::stringifier_wrapper<Output, FTuple>& m_stringifier;

};


} // inline namespace v0
} // namespace stringify
} // namespace boost

#endif  /* BOOST_STRINGIFY_V0_INPUT_ARG_HPP */

