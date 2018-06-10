#ifndef BOOST_STRINGIFY_V0_JOIN_HPP
#define BOOST_STRINGIFY_V0_JOIN_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/facets/encodings.hpp>
#include <initializer_list>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

namespace detail {

template <typename ... Args> class args_tuple;

template <>
class args_tuple<>
{
public:
    args_tuple()
    {
    }
};

template <typename Arg, typename ... Args>
class args_tuple<Arg, Args...> : private args_tuple<Args...>
{
public:
    args_tuple(const Arg& arg, const Args& ... args)
        : args_tuple<Args...>(args...)
        , first_arg(arg)
    {
    }

    const Arg& first_arg;

    const args_tuple<Args...>& remove_first() const
    {
        return *this;
    }

    args_tuple<Args...>& remove_first()
    {
        return *this;
    }
};

template <typename CharT>
struct printer_ptr_range
{
    using fmt_ptr = const stringify::v0::printer<CharT>*;

    const fmt_ptr* begin() const
    {
        return m_begin;
    }
    const fmt_ptr* end() const
    {
        return m_end;
    }
    virtual std::size_t size() const
    {
        return m_end - m_begin;
    }

    fmt_ptr* m_begin = nullptr;
    fmt_ptr* m_end = nullptr;
};


template <typename CharT, typename FPack, typename ... Args>
class printers_tuple;

template <typename CharT, typename FPack>
class printers_tuple<CharT, FPack>
{
public:
    printers_tuple
        ( stringify::v0::output_writer<CharT>&
        , const FPack&
        , const stringify::v0::detail::args_tuple<>& )
    {
    }

    using fmt_ptr = const stringify::v0::printer<CharT>*;

    fmt_ptr* fill(fmt_ptr* out_it) const
    {
        return out_it;
    }
};


template <typename CharT, typename FPack, typename Arg, typename ... Args>
class printers_tuple<CharT, FPack, Arg, Args...>
{
    using printer_type
        = decltype
            ( stringify_make_printer<CharT, FPack>
                ( * std::declval<stringify::v0::output_writer<CharT>*> ()
                , std::declval<FPack>()
                , std::declval<const Arg>()));
public:

    printers_tuple
        ( stringify::v0::output_writer<CharT>& out
        , const FPack& ft
        , const stringify::v0::detail::args_tuple<Arg, Args...>& args
        )
        : m_printer
          (stringify_make_printer<CharT, FPack>(out, ft, args.first_arg))
        , m_rest(out, ft, args.remove_first())
    {
    }

    using fmt_ptr = const stringify::v0::printer<CharT>*;

    fmt_ptr* fill(fmt_ptr* out_it) const
    {
        *out_it = &m_printer;
        return m_rest.fill(++out_it);
    }

private:

    printer_type m_printer;
    printers_tuple<CharT, FPack, Args...> m_rest;
};



template <typename CharT, typename FPack, typename ... Args>
class printers_group
{
public:

    printers_group
        ( stringify::v0::output_writer<CharT>& out
        , const FPack& ft
        , const stringify::v0::detail::args_tuple<Args...>& args
        )
        : m_impl(out, ft, args)
    {
        m_range.m_end = m_impl.fill(m_array);
        m_range.m_begin = m_array;
    }

    virtual ~printers_group()
    {
    }

    const auto& range() const
    {
        return m_range;
    }

private:

    stringify::v0::detail::printers_tuple<CharT, FPack, Args...> m_impl;

    using fmt_ptr = const stringify::v0::printer<CharT>*;
    fmt_ptr m_array[sizeof...(Args)];
    stringify::v0::detail::printer_ptr_range<CharT> m_range;
};


struct join_t;

template <typename ... Args>
struct joined_args
{
    const stringify::v0::detail::join_t& join;
    stringify::v0::detail::args_tuple<Args...> args;
};


struct join_t
{
    int width = 0;
    stringify::v0::alignment align = stringify::v0::alignment::right;
    char32_t fillchar = U' ';
    int num_leading_args = 1;

    template <typename ... Args>
    stringify::v0::detail::joined_args<Args...> operator()(const Args& ... args) const
    {
        return {*this, stringify::v0::detail::args_tuple<Args...>{args...}};
    }
};


template <typename CharT>
class join_printer_impl: public printer<CharT>
{
    using printer_type = stringify::v0::printer<CharT>;
    using pp_range = stringify::v0::detail::printer_ptr_range<CharT>;

public:

    using char_type   = CharT;
    using input_type  = stringify::v0::detail::join_t ;
    using writer_type = stringify::v0::output_writer<CharT>;

    join_printer_impl
        ( stringify::v0::output_writer<CharT>& out
        , const stringify::v0::detail::printer_ptr_range<CharT>& pp_range
        , const stringify::v0::detail::join_t& j
        , const stringify::v0::encoding<CharT>& encoding
        )
        : m_out(out)
        , m_join(j)
        , m_encoder{encoding.encoder()}
        , m_keepsurr{out.allow_surrogates()}
        , m_args{pp_range}
    {
        m_fillcount = remaining_width_from_arglist(m_join.width);
    }

    ~join_printer_impl()
    {
    }

    std::size_t length() const override
    {
        return args_length() + fill_length();
    }

    void write() const override
    {
        if (m_fillcount <= 0)
        {
            write_args();
        }
        else
        {
            switch(m_join.align)
            {
                case stringify::v0::alignment::left:
                    write_args();
                    write_fill(m_fillcount);
                    break;
                case stringify::v0::alignment::right:
                    write_fill(m_fillcount);
                    write_args();
                    break;
                case stringify::v0::alignment::internal:
                    write_splitted();
                    break;
                case stringify::v0::alignment::center:
                {
                    auto half_fillcount = m_fillcount / 2;
                    write_fill(half_fillcount);
                    write_args();
                    write_fill(m_fillcount - half_fillcount);
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

    stringify::v0::output_writer<CharT>& m_out;    
    input_type m_join;
    int m_fillcount = 0;
    const stringify::v0::encoder<CharT>& m_encoder;
    bool m_keepsurr;
    pp_range m_args = nullptr;

    std::size_t args_length() const
    {
        std::size_t sum = 0;
        for(const auto* arg : m_args)
        {
            sum += arg->length();
        }
        return sum;
    }

    std::size_t fill_length() const
    {
        if(m_fillcount > 0)
        {
            return m_fillcount * m_encoder.length(m_join.fillchar, m_keepsurr);
        }
        return 0;
    }

    int remaining_width_from_arglist(int w) const
    {
        for(auto it = m_args.begin(); w > 0 && it != m_args.end(); ++it)
        {
            w = (*it) -> remaining_width(w);
        }
        return w;
    }

    void write_splitted() const
    {
        auto it = m_args.begin();
        for ( int count = m_join.num_leading_args
            ; count > 0 && it != m_args.end()
            ; --count, ++it)
        {
            (*it)->write();
        }
        write_fill(m_fillcount);
        while(it != m_args.end())
        {
            (*it)->write();
            ++it;
        }
    }

    void write_args() const
    {
        for(const auto& arg : m_args)
        {
            arg->write();
        }
    }

    void write_fill(int count) const
    {
        m_out.put32(count, m_join.fillchar);
    }
};


template <typename CharT, typename FPack, typename ... Args>
class join_printer
    : private stringify::v0::detail::printers_group<CharT, FPack, Args...>
    , public stringify::v0::detail::join_printer_impl<CharT>
{
    using fmt_group
    = stringify::v0::detail::printers_group<CharT, FPack, Args...>;

    using join_impl
    = stringify::v0::detail::join_printer_impl<CharT>;


public:

    using input_type  = stringify::v0::detail::join_t ;

    join_printer
        ( stringify::v0::output_writer<CharT>& out
        , const FPack& ft
        , const stringify::v0::detail::joined_args<Args...>& ja
        )
        : fmt_group(out, ft, ja.args)
        , join_impl{out, fmt_group::range(), ja.join, get_encoding(ft)}
    {
    }

    virtual ~join_printer()
    {
    }

private:

    static const auto& get_encoding(const FPack& ft)
    {
        using encoder_category = stringify::v0::encoding_category<CharT>;
        return ft.template get_facet<encoder_category, input_type>();
    }
};

// struct join_input_traits
// {
//     template <typename CharT, typename FPack, typename ... Args>
//     static inline stringify::v0::detail::join_printer<CharT, FPack, Args...>
//     make_printer
//         ( const FPack& ft
//         , const stringify::v0::detail::joined_args<Args...>& x
//         )
//     {
//         return {ft, x};
//     }
// };

} // namespace detail

// template <typename ... Args>
// stringify::v0::detail::join_input_traits stringify_get_input_traits
// (const stringify::v0::detail::joined_args<Args...>&);

template <typename CharT, typename FPack, typename ... Args>
inline stringify::v0::detail::join_printer<CharT, FPack, Args...>
stringify_make_printer
    ( stringify::v0::output_writer<CharT>& out
    , const FPack& ft
    , const stringify::v0::detail::joined_args<Args...>& x )
{
    return {out, ft, x};
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

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_JOIN_HPP

