#ifndef BOOST_STRINGIFY_V0_JOIN_HPP
#define BOOST_STRINGIFY_V0_JOIN_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/input_arg.hpp>
#include <boost/stringify/v0/facets/alignment.hpp>
#include <initializer_list>

namespace boost {
namespace stringify {
inline namespace v0 {
namespace detail {
struct join_t
{
    boost::stringify::v0::alignment align;
    char32_t fillchar = char32_t();
    int width = 0;
    int num_leading_args = 1;
};

}

boost::stringify::v0::detail::join_t
join_left(int width, char32_t fillchar =char32_t())
{
    return {boost::stringify::v0::alignment::left, fillchar, width, 0};
}


boost::stringify::v0::detail::join_t
join_right(int width, char32_t fillchar =char32_t())
{
    return {boost::stringify::v0::alignment::right, fillchar, width, 0};
}

boost::stringify::v0::detail::join_t
join_internal(int width, int num_leading_args = 1, char32_t fillchar =char32_t())
{
    return {boost::stringify::v0::alignment::internal
            , fillchar, width, num_leading_args};
}

boost::stringify::v0::detail::join_t
join_internal(int width, char32_t fillchar)
{
    return {boost::stringify::v0::alignment::internal, fillchar, width, 1};
}

namespace detail {

template <typename Output, typename FTuple>
class join_stringifier
{
    using width_t = boost::stringify::v0::width_t;
    using width_tag = boost::stringify::v0::width_tag;
    using input_arg = boost::stringify::v0::input_arg<Output, FTuple>;
    using ini_list_type = std::initializer_list<input_arg>;
public:

    using char_type    = typename Output::char_type ;
    using input_type  = boost::stringify::v0::detail::join_t ;
    using output_type = Output;
    using ftuple_type = FTuple;
    using arg_format_type = ini_list_type;
    
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
    const ini_list_type& m_args;
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
            boost::stringify::v0::fill_impl<boost::stringify::v0::true_trait>
                fill_writer(m_join.fillchar);
            return fill_writer.length<char_type>(m_fill_width, m_fmt);
        }
        else
        {
            boost::stringify::v0::fill_length<char_type, input_type>(m_fill_width, m_fmt);
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
            boost::stringify::v0::fill_impl<boost::stringify::v0::true_trait>
                fill_writer(m_join.fillchar);
            fill_writer.fill<char_type>(m_fill_width, out, m_fmt);
        }
        else
        {
            boost::stringify::v0::write_fill<char_type, input_type>(m_fill_width, out, m_fmt);
        }
    }
};


struct input_join_traits
{
    template <typename Output, typename FTuple>
    using stringifier =
        boost::stringify::v0::detail::join_stringifier<Output, FTuple>;
};

boost::stringify::v0::detail::input_join_traits
boost_stringify_input_traits_of(const boost::stringify::v0::detail::join_t&);


} // namespace detail


} // inline namespace v0
} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_V0_JOIN_HPP

