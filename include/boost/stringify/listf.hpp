#ifndef BOOST_STRINGIFY_LISTF_HPP_INCLUDED
#define BOOST_STRINGIFY_LISTF_HPP_INCLUDED

#include <initializer_list>
#include <boost/stringify/detail/input_base_ref.hpp>

namespace boost
{
namespace stringify
{

template <typename charT, typename Formating>
class listf: public boost::stringify::input_base<charT, Formating>
{
    typedef 
    std::initializer_list
        <boost::stringify::detail::input_base_ref<charT, Formating > >
    initializer_list_type;

    const initializer_list_type inputs;

public:

    listf(const initializer_list_type& _inputs) noexcept:
    inputs(_inputs)
    {
    }

    virtual std::size_t minimal_length(const Formating& fmt) const noexcept
    {
        std::size_t sum=0;
        for(auto it = inputs.begin(); it != inputs.end(); ++it)
        {
            sum += it->writer.minimal_length(fmt);
        }
        return sum;
    }


    virtual charT* write_without_termination_char
        ( charT* out
        , const Formating& fmt
        ) const noexcept
    {
        for(auto it = inputs.begin(); it != inputs.end(); ++it)
        {
            out = it->writer.write_without_termination_char(out, fmt);
        }
        return out;
    }

    virtual void write
        ( simple_ostream<charT>& out
        , const Formating& fmt
        ) const
    {
        for(auto it = inputs.begin(); it != inputs.end() && out.good(); ++it)
        {
            it->writer.write(out, fmt);
        }
    }

};
} // namespace stringify
} // namespace boost


#endif
