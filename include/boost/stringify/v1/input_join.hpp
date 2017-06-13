#ifndef BOOST_STRINGIFY_V1_JOIN_HPP
#define BOOST_STRINGIFY_V1_JOIN_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v1/input_arg.hpp>
#include <boost/stringify/v1/custom_alignment.hpp>
#include <initializer_list>

namespace boost {
namespace stringify {
inline namespace v1 {
namespace detail {
struct join_t
{
    boost::stringify::v1::alignment align;
    char32_t fillchar = char32_t();
    int width = 0;
    int num_leading_args = 1;
};

}

boost::stringify::v1::detail::join_t
join_left(int width, char32_t fillchar =char32_t())
{
    return {boost::stringify::v1::alignment::left, fillchar, width, 0};
}


boost::stringify::v1::detail::join_t
join_right(int width, char32_t fillchar =char32_t())
{
    return {boost::stringify::v1::alignment::right, fillchar, width, 0};
}

boost::stringify::v1::detail::join_t
join_internal(int width, int num_leading_args = 1, char32_t fillchar =char32_t())
{
    return {boost::stringify::v1::alignment::internal
            , fillchar, width, num_leading_args};
}

boost::stringify::v1::detail::join_t
join_internal(int width, char32_t fillchar)
{
    return {boost::stringify::v1::alignment::internal, fillchar, width, 1};
}

namespace detail {

template <typename CharT, typename Output, typename FTuple>
class join_stringifier
{
    using width_t = boost::stringify::v1::width_t;
    using width_tag = boost::stringify::v1::width_tag;
    using input_arg = boost::stringify::v1::input_arg<CharT, Output, FTuple>;

public:

    using input_type  = boost::stringify::v1::detail::join_t ;
    using char_type    = CharT ;
    using output_type = Output;
    using ftuple_type = FTuple;
    using arg_format_type = std::initializer_list<input_arg>;
    
    join_stringifier
        ( const FTuple& fmt
        , const input_type& j
        , const arg_format_type& args
        )
        : m_fmt(fmt)
        , m_join(j)
        , m_args(args)
        , m_fill_width(calc_fill_width(j.width))
    {
    }

    
    std::size_t length() const
    {
        return args_length() + m_fill_width;
    }


    void write(Output& out) const
    {
        if (m_fill_width <= 0)
        {
            write_args(out);
        }
        else
            switch(m_join.align)
            {
                case boost::stringify::v1::alignment::left:
                    write_args(out);
                    write_fill(out);
                    break;
                case boost::stringify::v1::alignment::right:
                    write_fill(out);
                    write_args(out);
                    break;
                case boost::stringify::v1::alignment::internal:
                    write_splitted(out);
                    break;
            }
    }

    
    int remaining_width(int w) const
    {
        if (m_fill_width > 0)
        {
            return std::max(0, w - m_join.width);
        }
        return calc_fill_width(w);
    }

   
private:

    const FTuple& m_fmt;
    const input_type& m_join;
    const std::initializer_list<input_arg>& m_args;
    const int m_fill_width;

    std::size_t args_length() const
    {
        std::size_t sum = 0;
        for(const auto& arg : m_args)
        {
            sum += arg.length(m_fmt);
        }
        return sum;
    }


    std::size_t fill_length() const
    {
        if(m_join.fillchar)
        {
            boost::stringify::v1::fill_impl<boost::stringify::v1::true_trait>
                fill_writer(m_join.fillchar);
            return fill_writer.length<CharT>(m_fill_width, m_fmt);
        }
        else
        {
            boost::stringify::v1::fill_length<CharT, input_type>(m_fill_width, m_fmt);
        }
    }


    int calc_fill_width(int total_width) const
    {
        int w = total_width;
        for(auto it = m_args.begin(); w > 0 && it != m_args.end(); ++it)
        {
            w = (*it).remaining_width(w, m_fmt);
        }
        return w;
    }


    void write_splitted(Output& out) const
    {
        auto it = m_args.begin();
        for ( int count = m_join.num_leading_args
            ; count > 0 && it != m_args.end()
            ; --count, ++it)      
        {
            (*it).write(out, m_fmt);
        }
        write_fill(out);
        while(it != m_args.end())
        {
            (*it).write(out, m_fmt);
            ++it;
        }
    }

    void write_args(Output& out) const
    {
        for(const auto& arg : m_args)
        {
            arg.write(out, m_fmt);
        }
    }
    
    void write_fill(Output& out) const
    {
        if(m_join.fillchar)
        {
            boost::stringify::v1::fill_impl<boost::stringify::v1::true_trait>
                fill_writer(m_join.fillchar);
            fill_writer.fill<CharT>(m_fill_width, out, m_fmt);
        }
        else
        {
            boost::stringify::v1::write_fill<CharT, input_type>(m_fill_width, out, m_fmt);
        }
    }
};


struct input_join_traits
{
    template <typename CharT, typename Output, typename FTuple>
    using stringifier =
        boost::stringify::v1::detail::join_stringifier<CharT, Output, FTuple>;
};

boost::stringify::v1::detail::input_join_traits
boost_stringify_input_traits_of(const boost::stringify::v1::detail::join_t&);


} // namespace detail


} // inline namespace v1
} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_V1_JOIN_HPP

