#ifndef BOOST_STRINGIFY_V0_FACETS_ALIGNMENT_HPP
#define BOOST_STRINGIFY_V0_FACETS_ALIGNMENT_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)


namespace boost {
namespace stringify {
inline namespace v0 {

enum class alignment{left, right, internal};


struct alignment_tag;


template < template <class> class Filter>
struct align_impl
{
    typedef boost::stringify::v0::alignment_tag category;

    template <typename T> using accept_input_type = Filter<T>;
    
    constexpr align_impl(boost::stringify::v0::alignment align)
        : m_align(align)
    {
    }

    constexpr boost::stringify::v0::alignment value() const
    {
        return m_align;
    }

private:

    const boost::stringify::v0::alignment m_align;
};


template
    < boost::stringify::v0::alignment Align
    , template <class> class Filter
    >
struct align_impl_t
{
    typedef boost::stringify::v0::alignment_tag category;

    template <typename T> using accept_input_type = Filter<T>;
    
    constexpr boost::stringify::v0::alignment value() const
    {
        return Align;
    }
};


constexpr auto align(boost::stringify::v0::alignment a)
{
    return boost::stringify::v0::align_impl<boost::stringify::v0::true_trait>(a);
}


template < template <class> class F>
constexpr auto align_if(boost::stringify::v0::alignment a)
{
    return boost::stringify::v0::align_impl<F>(a);
}


constexpr boost::stringify::v0::align_impl_t
     < boost::stringify::v0::alignment::left
     , boost::stringify::v0::true_trait
     >
left = boost::stringify::v0::align_impl_t
     < boost::stringify::v0::alignment::left
     , boost::stringify::v0::true_trait
     > ();


constexpr boost::stringify::v0::align_impl_t
     < boost::stringify::v0::alignment::right
     , boost::stringify::v0::true_trait
     >
right = boost::stringify::v0::align_impl_t
     < boost::stringify::v0::alignment::right
     , boost::stringify::v0::true_trait
     > ();


constexpr boost::stringify::v0::align_impl_t
     < boost::stringify::v0::alignment::internal
     , boost::stringify::v0::true_trait
     >
internal = boost::stringify::v0::align_impl_t
     < boost::stringify::v0::alignment::internal
     , boost::stringify::v0::true_trait
     > ();


template <template <class> class F>
boost::stringify::v0::align_impl_t <boost::stringify::v0::alignment::left, F>
left_if
= boost::stringify::v0::align_impl_t <boost::stringify::v0::alignment::left, F>();

template <template <class> class F>
boost::stringify::v0::align_impl_t <boost::stringify::v0::alignment::right, F>
right_if
= boost::stringify::v0::align_impl_t <boost::stringify::v0::alignment::right, F>();

template <template <class> class F>
boost::stringify::v0::align_impl_t <boost::stringify::v0::alignment::internal, F>
internal_if
= boost::stringify::v0::align_impl_t <boost::stringify::v0::alignment::internal, F>();


struct alignment_tag
{
    typedef
        boost::stringify::v0::align_impl_t
        <boost::stringify::v0::alignment::right, boost::stringify::v0::true_trait>
        default_impl;
};


} // inline namespace v0
} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_V0_FACETS_ALIGNMENT_HPP

