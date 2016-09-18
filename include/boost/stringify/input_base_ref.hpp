#ifndef BOOST_STRINGIFY_STR_WRITE_REF_HPP
#define BOOST_STRINGIFY_STR_WRITE_REF_HPP

#include <boost/stringify/input_base.hpp>

namespace boost {
namespace stringify {

template <typename charT, typename charTraits, typename Formating>
struct input_base_ref
{
    input_base_ref
        ( const boost::stringify::input_base<charT, Formating>& w
        )
        noexcept
        : writer(w)
    {
    }

    template <class T>
    using input_base_of
        = decltype(argf<charT, charTraits, Formating>(std::declval<const T>()));
           
    template <typename T, typename ... ExtraArgs>
    input_base_ref
        ( const T& value
        , input_base_of<T> && wt = input_base_of<T>()
        )
        noexcept
        : writer(wt)
    {
        wt.set(value);
    }

    /*   
    template <typename T, typename ... ExtraArgs>
    input_base_ref
        ( const T& value
        , const typename input_base_of<T>::local_formating& fmt
        , input_base_of<T> && wt = input_base_of<T, const char*>()
        ) noexcept
        : writer(wt)
    {
        wt.set(value, fmt);
    }
    */

    std::size_t length(const Formating& fmt) const noexcept
    {
        return writer.length(fmt);
    }

    charT* write_without_termination_char
        ( charT* out
        , const Formating& fmt
        ) const noexcept
    {
        return writer.write_without_termination_char(out, fmt);
    }
        
    charT* write
        ( charT* out
        , const Formating& fmt
        ) const noexcept
    {
        return writer.write(out, fmt);
    }
    
private:
   
    const boost::stringify::input_base<charT, Formating>& writer;

};


} // namespace stringify
} // namespace boost

#endif  /* BOOST_STRINGIFY_DETAIL_STR_WRITE_REF_HPP */

