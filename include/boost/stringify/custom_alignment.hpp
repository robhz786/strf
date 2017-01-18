#ifndef BOOST_STRINGIFY_CUSTOM_ALIGNMENT_HPP
#define BOOST_STRINGIFY_CUSTOM_ALIGNMENT_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)


namespace boost {
namespace stringify {


enum class alignment{left, right, internal};


struct alignment_tag;


template < template <class> class Filter>
struct align_impl
{
    typedef boost::stringify::alignment_tag category;

    template <typename T> using accept_input_type = Filter<T>;
    
    constexpr align_impl(boost::stringify::alignment align)
        : m_align(align)
    {
    }

    constexpr boost::stringify::alignment value() const
    {
        return m_align;
    }

private:

    const boost::stringify::alignment m_align;
};


template
    < boost::stringify::alignment Align
    , template <class> class Filter
    >
struct align_impl_t
{
    typedef boost::stringify::alignment_tag category;

    template <typename T> using accept_input_type = Filter<T>;
    
    constexpr boost::stringify::alignment value() const
    {
        return Align;
    }
};


constexpr auto align(boost::stringify::alignment a)
{
    return boost::stringify::align_impl<boost::stringify::true_trait>(a);
}


template < template <class> class F>
constexpr auto align_if(boost::stringify::alignment a)
{
    return boost::stringify::align_impl<F>(a);
}


constexpr boost::stringify::align_impl_t
     < boost::stringify::alignment::left
     , boost::stringify::true_trait
     >
left = boost::stringify::align_impl_t
     < boost::stringify::alignment::left
     , boost::stringify::true_trait
     > ();


constexpr boost::stringify::align_impl_t
     < boost::stringify::alignment::right
     , boost::stringify::true_trait
     >
right = boost::stringify::align_impl_t
     < boost::stringify::alignment::right
     , boost::stringify::true_trait
     > ();


constexpr boost::stringify::align_impl_t
     < boost::stringify::alignment::internal
     , boost::stringify::true_trait
     >
internal = boost::stringify::align_impl_t
     < boost::stringify::alignment::internal
     , boost::stringify::true_trait
     > ();


template <template <class> class F>
boost::stringify::align_impl_t <boost::stringify::alignment::left, F>
left_if
= boost::stringify::align_impl_t <boost::stringify::alignment::left, F>();

template <template <class> class F>
boost::stringify::align_impl_t <boost::stringify::alignment::right, F>
right_if
= boost::stringify::align_impl_t <boost::stringify::alignment::right, F>();

template <template <class> class F>
boost::stringify::align_impl_t <boost::stringify::alignment::internal, F>
internal_if
= boost::stringify::align_impl_t <boost::stringify::alignment::internal, F>();


struct alignment_tag
{
    typedef
        boost::stringify::align_impl_t
        <boost::stringify::alignment::right, boost::stringify::true_trait>
        default_impl;
};


template <typename InputType, typename Formatting>
boost::stringify::alignment
get_alignment(const Formatting& fmt) noexcept
{
    return fmt.template get<boost::stringify::alignment_tag, InputType>().value();
}


template <typename InputType, typename Formatting, typename Flags>
boost::stringify::alignment
get_alignment(const Formatting& fmt, const Flags& flags) noexcept
{
    if (flags.has_char('>'))
    {
        return boost::stringify::alignment::right;
    }
    else if (flags.has_char('<'))
    {                        
        return boost::stringify::alignment::left;
    }
    else if (flags.has_char('='))
    {
        return boost::stringify::alignment::internal;
    }
    return fmt.template get<boost::stringify::alignment_tag, InputType>().value();
}


} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_CUSTOM_ALIGNMENT_HPP

