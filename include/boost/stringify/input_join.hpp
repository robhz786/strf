#ifndef BOOST_STRINGIFY_JOIN_HPP
#define BOOST_STRINGIFY_JOIN_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/input_arg.hpp>
#include <boost/stringify/custom_alignment.hpp>
#include <initializer_list>

namespace boost {
namespace stringify {
namespace detail {

struct join_t
{
    boost::stringify::alignment align;
    char32_t fillchar;
    int width;    
};

struct join_generator
{
    constexpr join_generator(char32_t fillchar = U'\0')
        : m_fillchar(fillchar)
    {
    }

    constexpr join_generator(const join_generator&) = default;
    
    auto operator()(char32_t fill_char) const
    -> join_generator
    {
        return join_generator(fill_char);
    }

    boost::stringify::detail::join_t operator<(int width) const
    {
        return {boost::stringify::alignment::left, m_fillchar, width};
    }

    boost::stringify::detail::join_t operator>(int width) const
    {
        return {boost::stringify::alignment::right, m_fillchar, width};
    }

    boost::stringify::detail::join_t operator=(int width) const
    {
        return {boost::stringify::alignment::internal, m_fillchar, width};
    }

private:

    const char32_t m_fillchar;

};


template <typename CharT, typename Output, typename FTuple>
class join_stringifier
{
    using width_t = boost::stringify::width_t;
    using width_tag = boost::stringify::width_tag;
    using input_arg = boost::stringify::input_arg<CharT, Output, FTuple>;

public:

    using input_type  = boost::stringify::detail::join_t ;
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

    
    virtual std::size_t length() const
    {
        return args_length() + m_fill_width;
    }


    virtual void write(Output& out) const
    {
        if (m_fill_width <= 0)
        {
            write_args(out);
        }
        else
            switch(m_join.align)
            {
                case boost::stringify::alignment::left:
                    write_args(out);
                    write_fill(out);
                    break;
                case boost::stringify::alignment::right:
                    write_fill(out);
                    write_args(out);
                    break;
                case boost::stringify::alignment::internal:
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
            boost::stringify::fill_impl<boost::stringify::true_trait>
                fill_writer(m_join.fillchar);
            return fill_writer.length<CharT>(m_fill_width, m_fmt);
        }
        else
        {
            boost::stringify::fill_length<CharT, input_type>(m_fill_width, m_fmt);
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
        std::size_t reg_count = m_args.size() / 2;
        auto it = m_args.begin();
        while(reg_count && it != m_args.end())
        {
            (*it).write(out, m_fmt);
            ++it;
            --reg_count;
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
            boost::stringify::fill_impl<boost::stringify::true_trait>
                fill_writer(m_join.fillchar);
            fill_writer.fill<CharT>(m_fill_width, out, m_fmt);
        }
        else
        {
            boost::stringify::write_fill<CharT, input_type>(m_fill_width, out, m_fmt);
        }
    }
};


struct input_join_traits
{
    template <typename CharT, typename Output, typename FTuple>
    using stringifier =
        boost::stringify::detail::join_stringifier<CharT, Output, FTuple>;
};

boost::stringify::detail::input_join_traits
boost_stringify_input_traits_of(const boost::stringify::detail::join_t&);


} // namespace detail

constexpr boost::stringify::detail::join_generator join = U'\0';


} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_JOIN_HPP

