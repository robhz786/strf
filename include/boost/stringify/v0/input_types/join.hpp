#ifndef BOOST_STRINGIFY_V0_JOIN_HPP
#define BOOST_STRINGIFY_V0_JOIN_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/input_arg.hpp>
#include <initializer_list>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

enum class alignment{left, right, internal, center};

namespace detail {
struct join_t
{
    int width = 0;
    stringify::v0::alignment align = stringify::v0::alignment::right;
    char32_t fillchar = U' ';
    int num_leading_args = 1;
};
}

inline stringify::v0::detail::join_t
join( int width = 0
    , stringify::v0::alignment align = stringify::v0::alignment::right
    , char32_t fillchar = U' '
    , int num_leading_args = 0
    )
{
    return {width, align, fillchar, num_leading_args};
}

inline stringify::v0::detail::join_t
join_center(int width, char32_t fillchar = U' ')
{
    return {width, stringify::v0::alignment::center, fillchar, 0};
}

inline stringify::v0::detail::join_t
join_left(int width, char32_t fillchar = U' ')
{
    return {width, stringify::v0::alignment::left, fillchar, 0};
}


inline stringify::v0::detail::join_t
join_right(int width, char32_t fillchar = U' ')
{
    return {width, stringify::v0::alignment::right, fillchar, 0};
}

inline stringify::v0::detail::join_t
join_internal(int width, char32_t fillchar, int num_leading_args)
{
    return {width, stringify::v0::alignment::internal, fillchar, num_leading_args};
}

inline stringify::v0::detail::join_t
join_internal(int width, int num_leading_args)
{
    return {width, stringify::v0::alignment::internal, U' ', num_leading_args};
}

template <typename CharT, typename FTuple>
class join_formatter: public formatter<CharT>
{
    using width_calc_tag = stringify::v0::width_calculator_tag;
    using encoder_tag = stringify::v0::encoder_tag<CharT>;
    using input_arg = stringify::v0::input_arg<CharT, FTuple>;
    using ini_list_type = std::initializer_list<input_arg>;

public:

    using char_type   = CharT;
    using input_type  = stringify::v0::detail::join_t ;
    using writer_type = stringify::v0::output_writer<CharT>;
    using ftuple_type = FTuple;
    using second_arg = ini_list_type;

    join_formatter
        ( const FTuple& ft
        , const input_type& j
        , const second_arg& args
        )
        : m_ft(ft)
        , m_join(j)
        , m_args(args)
        , m_fillcount{remaining_width_from_arglist(m_join.width)}
    {
    }

    std::size_t length() const override
    {
        return args_length() + fill_length();
    }

    void write(writer_type& out) const override
    {
        if (m_fillcount <= 0)
        {
            write_args(out);
        }
        else
        {
            switch(m_join.align)
            {
                case stringify::v0::alignment::left:
                    write_args(out);
                    write_fill(out, m_fillcount);
                    break;
                case stringify::v0::alignment::right:
                    write_fill(out, m_fillcount);
                    write_args(out);
                    break;
                case stringify::v0::alignment::internal:
                    write_splitted(out);
                    break;
                case stringify::v0::alignment::center:
                {
                    auto half_fillcount = m_fillcount / 2;
                    write_fill(out, half_fillcount);
                    write_args(out);
                    write_fill(out, m_fillcount - half_fillcount);
                    break;
                }

            }
        }
    }

    int remaining_width(int w) const override
    {
        if (m_fillcount > 0)
        {
            return (std::max)(0, w - m_join.width);
        }
        return remaining_width_from_arglist(w);
    }

private:

    const FTuple& m_ft;
    input_type m_join;
    const ini_list_type m_args;
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
            return m_fillcount * get_facet<encoder_tag>().length(m_join.fillchar);
        }
        return 0;
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
        write_fill(out, m_fillcount);
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

    void write_fill(writer_type& out, int count) const
    {
         get_facet<encoder_tag>().encode(out, count, m_join.fillchar);
    }

};

namespace detail {

struct input_join_traits
{
    template <typename CharT, typename FTuple>
    using formatter =
        stringify::v0::join_formatter<CharT, FTuple>;
};

stringify::v0::detail::input_join_traits
boost_stringify_input_traits_of(const stringify::v0::detail::join_t&);

} // namespace detail

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_JOIN_HPP

