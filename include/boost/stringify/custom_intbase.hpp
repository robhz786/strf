#ifndef BOOST_STRINGIFY_CUSTOM_INTBASE_HPP
#define BOOST_STRINGIFY_CUSTOM_INTBASE_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

namespace boost {
namespace stringify {

struct intbase_tag;

template <int Base, template <class> class Filter>
struct intbase_impl_t
{
    using category = boost::stringify::intbase_tag;

    template <typename T> using accept_input_type = Filter<T>;
    
    constexpr int value() const
    {
        return Base;
    }
};

constexpr auto oct
= boost::stringify::intbase_impl_t<8, boost::stringify::true_trait>();

constexpr auto dec
= boost::stringify::intbase_impl_t<10, boost::stringify::true_trait>();

constexpr auto hex
= boost::stringify::intbase_impl_t<16, boost::stringify::true_trait>();

template <template <class> class F>
auto oct_if = boost::stringify::intbase_impl_t<8, F>();

template <template <class> class F>
auto dec_if = boost::stringify::intbase_impl_t<10, F>();

template <template <class> class F>
auto hex_if = boost::stringify::intbase_impl_t<16, F>();

struct intbase_tag
{
    using default_impl=
        boost::stringify::intbase_impl_t<10, boost::stringify::true_trait>;
};

template <typename InputType, typename Formatting>
constexpr int get_intbase(const Formatting& fmt) noexcept
{
    return fmt.template get<boost::stringify::intbase_tag, InputType>().value();
}


template <typename InputType, typename Formatting, typename Flags>
constexpr int get_intbase(const Formatting& fmt, const Flags& flags) noexcept
{
    if (flags.has_char('d'))
    {
        return 10;
    }
    if (flags.has_char('x') || flags.has_char('X'))
    {
        return 16;
    }
    else if (flags.has_char('o'))
    {                        
        return 8;
    }
    return boost::stringify::get_intbase<InputType>(fmt);
}


} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_CUSTOM_INTBASE_HPP

