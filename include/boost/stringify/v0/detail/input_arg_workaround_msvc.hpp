#ifndef BOOST_STRINGIFY_V0_DETAIL_INPUT_ARG_WORKAROUND_MSVC_HPP
#define BOOST_STRINGIFY_V0_DETAIL_INPUT_ARG_WORKAROUND_MSVC_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/detail/stringifier_wrapper.hpp>

#ifndef  BOOST_STRINGIFY_MAX_STRINGIFIER_SIZE
#define  BOOST_STRINGIFY_MAX_STRINGIFIER_SIZE (10*sizeof(void*))
#endif

namespace boost {
namespace stringify {
inline namespace v0 {

template <typename Output, typename FTuple>
class input_arg
{
    template <typename X>
    static decltype(auto) format_type_of(const X& x)
    {
        using traits_x = decltype(boost_stringify_input_traits_of(x));
        using stringifier_x = traits_x::template stringifier<Output, FTuple>;
        using arg_format_x = typename stringifier_x::arg_format_type;
        return std::declval<const arg_format_x>();
    }

    template <typename Value>
    auto* instantiate_wrapper(const Value& x)
    {
        using traits_x = decltype(boost_stringify_input_traits_of(x));
        using stringifier_x = traits_x::template stringifier<Output, FTuple>;
        using stringifier_wrapper_impl =
            boost::stringify::v0::detail::stringifier_wrapper_impl<stringifier_x>;
        static_assert
            ( sizeof(stringifier_wrapper_impl) <= sizeof(m_space)
            , "sizeof stringifier class too big. Define BOOST_STRINGIFY_MAX_STRINGIFIER_SIZE"
            );

        return new (get_strigifier()) stringifier_wrapper_impl();
    }

    using stringifier_wrapper_base
        = boost::stringify::v0::detail::stringifier_wrapper<Output, FTuple>;

public:
    
    template <typename T>
    input_arg(T&& arg)
    {
        instantiate_wrapper(arg)->set_args(arg);
    }

    template <typename T>
    input_arg(T&& arg1, decltype(format_type_of(arg1)) arg2)
    {
        instantiate_wrapper(arg1)->set_args(arg1, arg2);
    }

    ~input_arg()
    {
        get_strigifier()->~stringifier_wrapper_base();
    }
    
    std::size_t length(const FTuple& fmt) const
    {
        return get_strigifier()->length(fmt);
    }

    void write(Output& out, const FTuple& fmt) const
    {
        return get_strigifier()->write(out, fmt);
    }

    int remaining_width(int w, const FTuple& fmt) const
    {
        return get_strigifier()->remaining_width(w, fmt);
    }
    
private:

    mutable
        typename std::aligned_storage
           <BOOST_STRINGIFY_MAX_STRINGIFIER_SIZE + 2*sizeof(void*)>
           ::type
        m_space;

    stringifier_wrapper_base* get_strigifier() const
    {
        return reinterpret_cast<stringifier_wrapper_base*>(&m_space);
    }

};


} // inline namespace v0
} // namespace stringify
} // namespace boost

#endif  /* BOOST_STRINGIFY_V0_DETAIL_INPUT_ARG_WORKAROUND_MSVC_HPP */

