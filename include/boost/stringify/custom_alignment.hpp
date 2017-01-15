#ifndef BOOST_STRINGIFY_CUSTOM_ALIGNMENT_HPP
#define BOOST_STRINGIFY_CUSTOM_ALIGNMENT_HPP

namespace boost {
namespace stringify {

enum class alignment{left, right, internal};

struct alignment_tag;

template
    < boost::stringify::alignment Align
    , template <class> class Filter
    >
struct fimpl_alignment
{
    typedef boost::stringify::alignment_tag category;

    template <typename T> using accept_input_type = Filter<T>;
    
    constexpr boost::stringify::alignment value() const
    {
        return Align;
    }
};

template <template <class> class Filter = boost::stringify::accept_any_type>
constexpr auto left = boost::stringify::fimpl_alignment
    < boost::stringify::alignment::left
    , Filter
    >();


template <template <class> class Filter = boost::stringify::accept_any_type>
constexpr auto right = boost::stringify::fimpl_alignment
    < boost::stringify::alignment::right
    , Filter
    >();


template <template <class> class Filter = boost::stringify::accept_any_type>
constexpr auto internal = boost::stringify::fimpl_alignment
    < boost::stringify::alignment::internal
    , Filter
    >();


struct alignment_tag
{
    typedef
        boost::stringify::fimpl_alignment
        <boost::stringify::alignment::right, boost::stringify::accept_any_type>
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

