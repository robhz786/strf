#ifndef BOOST_STRINGIFY_CUSTOM_FILL_HPP
#define BOOST_STRINGIFY_CUSTOM_FILL_HPP

#include <string>
#include <boost/stringify/ftuple.hpp>
#include <boost/stringify/custom_width.hpp>
#include <boost/stringify/type_traits.hpp>
#include <boost/stringify/detail/characters_catalog.hpp>

namespace boost {
namespace stringify {

template <typename CharT> struct fill_tag;

template
    < typename CharT
    , CharT fill_char = boost::stringify::detail::the_space_character<CharT>()
    , template <class> class Filter = boost::stringify::accept_any_type  
    >
class fimpl_fill_static_single_char
{
public:

    typedef boost::stringify::width_t width_type;
    typedef boost::stringify::fill_tag<CharT> category;
    template <typename T> using accept_input_type = Filter<T>;
    
    template <class WidthAccumulator, class Output>
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
    < typename CharT
    , template <class> class Filter = boost::stringify::accept_any_type  
    >
class fimpl_fill_single_char
{
public:
    typedef boost::stringify::width_t width_type;
    typedef boost::stringify::fill_tag<CharT> category;
    template <typename T> using accept_input_type = Filter<T>;

    fimpl_fill_single_char(CharT fill) : m_fill_char(fill)
    {
    }

    fimpl_fill_single_char(const fimpl_fill_single_char&) = default;


    template <class WidthAccumulator, class Output>
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

    CharT m_fill_char;
};


// template
//     < typename CharT
//     , template <class> class Filter = boost::stringify::accept_any_type  
//     >
// class fimpl_fill_str
// {
// public:
//     typedef boost::stringify::width_t width_type;
//     typedef boost::stringify::fill_tag<CharT> category;
//     template <typename T> using accept_input_type = Filter<T>;

//     fimpl_fill_str(const CharT* str) : m_fill(str)
//     {
//     }

//     fimpl_fill_str(std::basic_string<CharT> str) : m_fill(std::move(str))
//     {
//     }
    
//     fimpl_fill_str(const fimpl_fill_str&) = default;
    
//     template <class CharTraits,  class WidthAccumulator>
//     CharT* fill(CharT* out, width_type width) const noexcept
//     {
//         auto rep = repetitions<WidthAccumulator>(width);
//         std::size_t len = m_fill.lenght() * rep;
//         for(; rep > 0; --rep)
//         {
//             CharTraits::copy(out, m_fill, m_fill.length());
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

//     std::basic_string<CharT> m_fill;
// };

template <typename CharT>
struct fill_tag
{
    typedef 
        boost::stringify::fimpl_fill_static_single_char
            < CharT
            , boost::stringify::detail::the_space_character<CharT>()
            , boost::stringify::accept_any_type
            >
        default_impl;
};

template
    < typename CharT
    , CharT fillChar
    , template <class> class Filter = boost::stringify::accept_any_type
    >
auto fill()
{
    return fimpl_fill_static_single_char<CharT, fillChar, Filter>();
}

template
    < typename CharT
    , template <class> class Filter = boost::stringify::accept_any_type
    >
auto fill(CharT fillChar)
{
    return fimpl_fill_single_char<CharT, Filter>(fillChar);
}


template <typename CharT, typename InputType, typename Formating>
decltype(auto) get_filler(const Formating& fmt)
{
    return fmt.template get<boost::stringify::fill_tag<CharT>, InputType>();
}
    
} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_CUSTOM_FILL_HPP

