#ifndef BOOST_STRINGIFY_V0_JOIN_HPP
#define BOOST_STRINGIFY_V0_JOIN_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/input_arg.hpp>
#include <boost/stringify/v0/facets/alignment.hpp>
#include <initializer_list>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN
namespace detail {
struct join_t
{
    boost::stringify::v0::alignment align;
    char32_t fillchar = char32_t();
    int width = 0;
    int num_leading_args = 1;
};

}

inline boost::stringify::v0::detail::join_t
join_left(int width, char32_t fillchar =char32_t())
{
    return {boost::stringify::v0::alignment::left, fillchar, width, 0};
}


inline boost::stringify::v0::detail::join_t
join_right(int width, char32_t fillchar =char32_t())
{
    return {boost::stringify::v0::alignment::right, fillchar, width, 0};
}

inline boost::stringify::v0::detail::join_t
join_internal(int width, int num_leading_args = 1, char32_t fillchar =char32_t())
{
    return {boost::stringify::v0::alignment::internal
            , fillchar, width, num_leading_args};
}

inline boost::stringify::v0::detail::join_t
join_internal(int width, char32_t fillchar)
{
    return {boost::stringify::v0::alignment::internal, fillchar, width, 1};
}

namespace detail {

template <typename CharT, typename FTuple>
class join_stringifier
{
    using width_calc_tag = boost::stringify::v0::width_calculator_tag;
    using from_utf32_tag = boost::stringify::v0::conversion_from_utf32_tag<CharT>;
    using input_arg = boost::stringify::v0::input_arg<CharT, FTuple>;
    using ini_list_type = std::initializer_list<input_arg>;

public:

    using char_type   = CharT;
    using input_type  = boost::stringify::v0::detail::join_t ;
    using writer_type = boost::stringify::v0::output_writer<CharT>;
    using ftuple_type = FTuple;
    using second_arg = ini_list_type;

    join_stringifier
        ( const FTuple& ft
        , const input_type& j
        , const second_arg& args
        )
        : m_ft(ft)
        , m_join(j)
        , m_args(args)
    {
        determinate_fill();
    }

    std::size_t length() const
    {
        return args_length() + fill_length();
    }

    void write(writer_type& out) const
    {
        if (m_fillcount <= 0)
        {
            write_args(out);
        }
        else
        {
            switch(m_join.align)
            {
                case boost::stringify::v0::alignment::left:
                    write_args(out);
                    write_fill(out);
                    break;
                case boost::stringify::v0::alignment::right:
                    write_fill(out);
                    write_args(out);
                    break;
                case boost::stringify::v0::alignment::internal:
                    write_splitted(out);
                    break;
            }
        }
    }

    int remaining_width(int w) const
    {
        if (m_fillcount > 0)
        {
            return (std::max)(0, w - m_join.width);
        }
        return remaining_width_from_arglist(w);
    }

private:

    const FTuple& m_ft;
    const input_type& m_join;
    const ini_list_type& m_args;
    char32_t m_fillchar = U' ';
    int m_fillcount = 0;

    template <typename Category>
    const auto& get_facet() const
    {
        return m_ft.template get_facet<Category, input_type>();
    }

    std::size_t args_length() const
    {
        std::size_t sum = 0;
        for(const auto& arg : m_args)
        {
            sum += arg.length(m_ft);
        }
        return sum;
    }

    std::size_t fill_length() const
    {
        if(m_fillcount > 0)
        {
            return m_fillcount * get_facet<from_utf32_tag>().length(m_fillchar);
        }
        return 0;
    }

    void determinate_fill()
    {
        int fill_width = remaining_width_from_arglist(m_join.width);
        if(fill_width > 0)
        {
            m_fillchar = determinate_fillchar();
            int fillchar_width = get_facet<width_calc_tag>().width_of(m_fillchar);
            m_fillcount = fill_width / fillchar_width;
        }
    }

    char32_t determinate_fillchar() const
    {
        if(m_join.fillchar != 0)
        {
            return m_join.fillchar;
        }
        return get_facet<fill_tag>().fill_char();
    }

    int remaining_width_from_arglist(int w) const
    {
        for(auto it = m_args.begin(); w > 0 && it != m_args.end(); ++it)
        {
            w = (*it).remaining_width(w, m_ft);
        }
        return w;
    }

    void write_splitted(writer_type& out) const
    {
        auto it = m_args.begin();
        for ( int count = m_join.num_leading_args
            ; count > 0 && it != m_args.end()
            ; --count, ++it)
        {
            (*it).write(out, m_ft);
        }
        write_fill(out);
        while(it != m_args.end())
        {
            (*it).write(out, m_ft);
            ++it;
        }
    }

    void write_args(writer_type& out) const
    {
        for(const auto& arg : m_args)
        {
            arg.write(out, m_ft);
        }
    }

    void write_fill(writer_type& out) const
    {
         get_facet<from_utf32_tag>().write(out, m_fillchar, m_fillcount);
    }

};


struct input_join_traits
{
    template <typename CharT, typename FTuple>
    using stringifier =
        boost::stringify::v0::detail::join_stringifier<CharT, FTuple>;
};


boost::stringify::v0::detail::input_join_traits
boost_stringify_input_traits_of(const boost::stringify::v0::detail::join_t&);


} // namespace detail

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_JOIN_HPP

