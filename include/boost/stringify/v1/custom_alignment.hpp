#ifndef BOOST_STRINGIFY_V1_CUSTOM_ALIGNMENT_HPP
#define BOOST_STRINGIFY_V1_CUSTOM_ALIGNMENT_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)


namespace boost {
namespace stringify {
inline namespace v1 {

enum class alignment{left, right, internal};


struct alignment_tag;


template < template <class> class Filter>
struct align_impl
{
    typedef boost::stringify::v1::alignment_tag category;

    template <typename T> using accept_input_type = Filter<T>;
    
    constexpr align_impl(boost::stringify::v1::alignment align)
        : m_align(align)
    {
    }

    constexpr boost::stringify::v1::alignment value() const
    {
        return m_align;
    }

private:

    const boost::stringify::v1::alignment m_align;
};


template
    < boost::stringify::v1::alignment Align
    , template <class> class Filter
    >
struct align_impl_t
{
    typedef boost::stringify::v1::alignment_tag category;

    template <typename T> using accept_input_type = Filter<T>;
    
    constexpr boost::stringify::v1::alignment value() const
    {
        return Align;
    }
};


constexpr auto align(boost::stringify::v1::alignment a)
{
    return boost::stringify::v1::align_impl<boost::stringify::v1::true_trait>(a);
}


template < template <class> class F>
constexpr auto align_if(boost::stringify::v1::alignment a)
{
    return boost::stringify::v1::align_impl<F>(a);
}


constexpr boost::stringify::v1::align_impl_t
     < boost::stringify::v1::alignment::left
     , boost::stringify::v1::true_trait
     >
left = boost::stringify::v1::align_impl_t
     < boost::stringify::v1::alignment::left
     , boost::stringify::v1::true_trait
     > ();


constexpr boost::stringify::v1::align_impl_t
     < boost::stringify::v1::alignment::right
     , boost::stringify::v1::true_trait
     >
right = boost::stringify::v1::align_impl_t
     < boost::stringify::v1::alignment::right
     , boost::stringify::v1::true_trait
     > ();


constexpr boost::stringify::v1::align_impl_t
     < boost::stringify::v1::alignment::internal
     , boost::stringify::v1::true_trait
     >
internal = boost::stringify::v1::align_impl_t
     < boost::stringify::v1::alignment::internal
     , boost::stringify::v1::true_trait
     > ();


template <template <class> class F>
boost::stringify::v1::align_impl_t <boost::stringify::v1::alignment::left, F>
left_if
= boost::stringify::v1::align_impl_t <boost::stringify::v1::alignment::left, F>();

template <template <class> class F>
boost::stringify::v1::align_impl_t <boost::stringify::v1::alignment::right, F>
right_if
= boost::stringify::v1::align_impl_t <boost::stringify::v1::alignment::right, F>();

template <template <class> class F>
boost::stringify::v1::align_impl_t <boost::stringify::v1::alignment::internal, F>
internal_if
= boost::stringify::v1::align_impl_t <boost::stringify::v1::alignment::internal, F>();


struct alignment_tag
{
    typedef
        boost::stringify::v1::align_impl_t
        <boost::stringify::v1::alignment::right, boost::stringify::v1::true_trait>
        default_impl;
};


} // inline namespace v1
} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_V1_CUSTOM_ALIGNMENT_HPP

