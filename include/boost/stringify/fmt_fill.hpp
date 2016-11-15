#ifndef BOOST_STRINGIFY_FMT_FILL_HPP
#define BOOST_STRINGIFY_FMT_FILL_HPP

#include <string>
#include <boost/stringify/fmt_width.hpp>
#include <boost/stringify/type_traits.hpp>
#include <boost/stringify/detail/characters_catalog.hpp>

namespace boost {
namespace stringify {

template <typename charT> struct ftype_fill;

template
    < typename charT
    , charT fill_char = boost::stringify::detail::the_space_character<charT>()
    , template <class> class Filter = boost::stringify::accept_any_type  
    >
class fimpl_fill_static_single_char
{
public:

    typedef boost::stringify::width_t width_type;
    typedef boost::stringify::ftype_fill<charT> fmt_type;
    template <typename T> using accept_input_type = Filter<T>;
    
    template <class Output, class WidthAccumulator>
    void fill
        ( Output& out
        , width_type width
        ) const noexcept
    {
        out.put(fill_char, repetitions<WidthAccumulator>(width));
    }

    template <class WidthAccumulator>
    std::size_t length(width_type width) const noexcept
    {
        return repetitions<WidthAccumulator>(width);
    }
    
private:
            
    template <class WidthAccumulator>
    width_type::scalar_type repetitions(width_type width) const noexcept
    {
        WidthAccumulator acc;
        acc.add(fill_char);
        return width / acc.result();        
    }
};

template
    < typename charT
    , template <class> class Filter = boost::stringify::accept_any_type  
    >
class fimpl_fill_single_char
{
public:
    typedef boost::stringify::width_t width_type;
    typedef boost::stringify::ftype_fill<charT> fmt_type;
    template <typename T> using accept_input_type = Filter<T>;

    fimpl_fill_single_char(charT fill) : m_fill_char(fill)
    {
    }

    fimpl_fill_single_char(const fimpl_fill_single_char&) = default;


    template <class Output, class WidthAccumulator>
    void fill
        ( Output& out
        , width_type width
        ) const noexcept
    {
        out.put(m_fill_char, repetitions<WidthAccumulator>(width));
    }
    
    template <class WidthAccumulator>
    std::size_t length(width_type width) const noexcept
    {
        return repetitions<WidthAccumulator>(width);
    }
    
private:
            
    template <class WidthAccumulator>
    width_type::scalar_type repetitions(width_type width) const noexcept
    {
        WidthAccumulator acc;
        acc.add(m_fill_char);
        return width / acc.result() ;        
    }

    charT m_fill_char;
};


// template
//     < typename charT
//     , template <class> class Filter = boost::stringify::accept_any_type  
//     >
// class fimpl_fill_str
// {
// public:
//     typedef boost::stringify::width_t width_type;
//     typedef boost::stringify::ftype_fill<charT> fmt_type;
//     template <typename T> using accept_input_type = Filter<T>;

//     fimpl_fill_str(const charT* str) : m_fill(str)
//     {
//     }

//     fimpl_fill_str(std::basic_string<charT> str) : m_fill(std::move(str))
//     {
//     }
    
//     fimpl_fill_str(const fimpl_fill_str&) = default;
    
//     template <class charTraits,  class WidthAccumulator>
//     charT* fill(charT* out, width_type width) const noexcept
//     {
//         auto rep = repetitions<WidthAccumulator>(width);
//         std::size_t len = m_fill.lenght() * rep;
//         for(; rep > 0; --rep)
//         {
//             charTraits::copy(out, m_fill, m_fill.length());
//         }
//         return out + len;
//     }

//     template <class OutputStreamFacade, class WidthAccumulator>
//     void fill
//         ( OutputStreamFacade& out
//         , width_type width
//         ) const noexcept
//     {
//         for(auto rep = repetitions<WidthAccumulator>(width); rep > 0; --rep)
//         {
//             out.put(m_fill, m_fill.lenght());
//         }
//     }

//     template <class WidthAccumulator>
//     std::size_t length(width_type width) const noexcept
//     {
//         return repetitions<WidthAccumulator>(width) * m_fill.length();
//     }
    
// private:
            
//     template <class WidthAccumulator>
//     width_type::scalar_type repetitions(width_type width) const noexcept
//     {
//         WidthAccumulator acc;
//         acc.add(m_fill);
//         return width / acc.result();
//     }

//     std::basic_string<charT> m_fill;
// };

template <typename charT>
struct ftype_fill
{
    typedef 
        boost::stringify::fimpl_fill_static_single_char
            < charT
            , boost::stringify::detail::the_space_character<charT>()
            , boost::stringify::accept_any_type
            >
        default_impl;
};

template
    < typename charT
    , charT fillChar
    , template <class> class Filter = boost::stringify::accept_any_type
    >
auto fill()
{
    return fimpl_fill_static_single_char<charT, fillChar, Filter>();
}

template
    < typename charT
    , template <class> class Filter = boost::stringify::accept_any_type
    >
auto fill(charT fillChar)
{
    return fimpl_fill_single_char<charT, Filter>(fillChar);
}

// template
//     < typename charT
//     , template <class> class Filter = boost::stringify::accept_any_type
//     >
// auto fill(std::basic_string<charT> fill)
// {
//     return fimpl_fill_str<charT, Filter>(std::move(fill));
// }
    
} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_FMT_FILL_HPP

