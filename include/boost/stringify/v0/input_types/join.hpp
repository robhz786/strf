#ifndef BOOST_STRINGIFY_V0_JOIN_HPP
#define BOOST_STRINGIFY_V0_JOIN_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/facets/encoding.hpp>
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

    printers_tuple(const printers_tuple&) = default;

    printers_tuple(printers_tuple&&)
    {
    }

    printers_tuple
        ( const FPack&
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
            ( make_printer<CharT, FPack>
                ( std::declval<FPack>()
                , std::declval<const Arg>()));
public:

    printers_tuple(const printers_tuple&) = default;

    printers_tuple(const printers_tuple&& rval)
        : m_printer(std::move(rval.m_printer))
        , m_rest(std::move(rval.m_rest))
    {
    }

    printers_tuple
        ( const FPack& fp
        , const stringify::v0::detail::args_tuple<Arg, Args...>& args )
        : m_printer(make_printer<CharT, FPack>(fp, args.first_arg))
        , m_rest(fp, args.remove_first())
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
        ( const FPack& fp
        , const stringify::v0::detail::args_tuple<Args...>& args )
        : m_impl(fp, args)
    {
        m_range.m_end = m_impl.fill(m_array);
        m_range.m_begin = m_array;
    }

    printers_group(const printers_group& cp)
        : m_impl(cp.m_impl)
    {
        m_range.m_end = m_impl.fill(m_array);
        m_range.m_begin = m_array;
    }

    printers_group(printers_group&& rval)
        : m_impl(std::move(rval.m_impl))
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

    using printer_ptr = const stringify::v0::printer<CharT>*;
    printer_ptr m_array[sizeof...(Args)];
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
    stringify::v0::detail::joined_args<Args...> operator()
        (const Args& ... args) const
    {
        return {*this, {args...}};
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

    join_printer_impl
        ( const stringify::v0::detail::printer_ptr_range<CharT>& args
        , const stringify::v0::detail::join_t& j
        , const stringify::v0::encoding<CharT>& encoding
        , stringify::v0::error_handling err_hdl )
        : m_join{j}
        , m_args{args}
        , m_encoding(encoding)
        , m_err_hdl(err_hdl)
    {
        m_fillcount = remaining_width_from_arglist(m_join.width);
    }

    join_printer_impl( const join_printer_impl& cp ) = delete;

    join_printer_impl
        ( const join_printer_impl& cp
        , const stringify::v0::detail::printer_ptr_range<CharT>& args )
        : m_join{cp.m_join}
        , m_args{args}
        , m_encoding(cp.encoder)
    {
    }

    join_printer_impl
        ( join_printer_impl&& tmp
        , const stringify::v0::detail::printer_ptr_range<CharT>& args )
        : m_join{std::move(tmp.m_join)}
        , m_args{std::move(args)}
        , m_encoding(tmp.m_encoding)
    {
    }

    ~join_printer_impl()
    {
    }

    std::size_t necessary_size() const override
    {
        return args_length() + fill_length();
    }

    stringify::v0::expected_buff_it<CharT> write
        ( stringify::v0::buff_it<CharT> buff
        , stringify::buffer_recycler<CharT>& recycler ) const
    {
        if (m_fillcount <= 0)
        {
            return write_args(buff, recycler);
        }
        else
        {
            switch(m_join.align)
            {
                case stringify::v0::alignment::left:
                {
                    auto x = write_args(buff, recycler);
                    return x ? write_fill(*x, recycler, m_fillcount): x;
                }
                case stringify::v0::alignment::right:
                {
                    auto x = write_fill(buff, recycler, m_fillcount);
                    return x ? write_args(*x, recycler) : x;
                }
                case stringify::v0::alignment::internal:
                {
                    return write_splitted(buff, recycler);
                }
                default:
                {
                    BOOST_ASSERT(m_join.align == stringify::v0::alignment::center);
                    auto half_fillcount = m_fillcount / 2;
                    auto x = write_fill(buff, recycler, half_fillcount);
                    if(x)
                    {
                        x = write_args(*x, recycler);
                    };
                    return x ? write_fill(*x, recycler, m_fillcount - half_fillcount) : x;
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

    input_type m_join;
    int m_fillcount = 0;
    pp_range m_args = nullptr;
    const stringify::v0::encoding<CharT>& m_encoding;
    stringify::v0::error_handling m_err_hdl;

    std::size_t args_length() const
    {
        std::size_t sum = 0;
        for(const auto* arg : m_args)
        {
            sum += arg->necessary_size();
        }
        return sum;
    }

    std::size_t fill_length() const
    {
        if(m_fillcount > 0)
        {
            return m_fillcount * m_encoding.char_size(m_join.fillchar, m_err_hdl);
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

    stringify::v0::expected_buff_it<CharT> write_splitted
        ( stringify::v0::buff_it<CharT> buff
        , stringify::buffer_recycler<CharT>& recycler ) const
    {
        stringify::v0::expected_buff_it<CharT> x { stringify::v0::in_place_t{}
                                                   , buff };
        auto it = m_args.begin();
        for ( int count = m_join.num_leading_args
            ; count > 0 && it != m_args.end()
            ; --count, ++it)
        {
            x = (*it)->write(*x, recycler);
            BOOST_STRINGIFY_RETURN_ON_ERROR(x);
        }
        x = write_fill(*x, recycler, m_fillcount);
        while(it != m_args.end())
        {
            BOOST_STRINGIFY_RETURN_ON_ERROR(x);
            x = (*it)->write(*x, recycler);
            ++it;
        }
        return x;
    }

    stringify::v0::expected_buff_it<CharT> write_args
        ( stringify::v0::buff_it<CharT> buff
        , stringify::buffer_recycler<CharT>& recycler ) const
    {
        stringify::v0::expected_buff_it<CharT> x { stringify::v0::in_place_t{}
                                                 , buff };
        for(const auto& arg : m_args)
        {
            x = arg->write(*x, recycler);
            BOOST_STRINGIFY_RETURN_ON_ERROR(x);
        }
        return x;
    }

    stringify::v0::expected_buff_it<CharT> write_fill
        ( stringify::v0::buff_it<CharT> buff
        , stringify::buffer_recycler<CharT>& recycler
        , int count ) const
    {
        std::size_t count2 = count;
        while(true)
        {
            auto res = m_encoding.encode_fill( &buff.it, buff.end
                                             , count2, m_join.fillchar, m_err_hdl );
            if (res == stringify::v0::cv_result::success)
            {
                return {stringify::v0::in_place_t{}, buff};
            }
            if (res == stringify::v0::cv_result::invalid_char)
            {
                return {stringify::v0::unexpect_t{}, stringify::v0::encoding_error()};
            }
            BOOST_ASSERT(res == stringify::v0::cv_result::insufficient_space);
            auto x = recycler.recycle(buff.it);
            BOOST_STRINGIFY_RETURN_ON_ERROR(x);
            buff = *x;
        }
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
        ( const FPack& fp
        , const stringify::v0::detail::joined_args<Args...>& ja )
        : fmt_group(fp, ja.args)
        , join_impl{ fmt_group::range()
                   , ja.join
                   , stringify::v0::get_facet
                       < stringify::v0::encoding_category<CharT>, void > (fp)
                   , stringify::v0::get_facet
                       < stringify::v0::encoding_policy_category, void > (fp)
                       .err_hdl() }
    {
    }

    join_printer(const join_printer& cp)
        : fmt_group(cp)
        , join_impl(cp, fmt_group::range())
    {
    }

    // join_printer(join_printer&& tmp)
    //     : fmt_group(std::move(tmp))
    //     , join_impl(std::move(tmp), fmt_group::range())
    // {
    // }

    virtual ~join_printer()
    {
    }
};

} // namespace detail

template <typename CharT, typename FPack, typename ... Args>
inline stringify::v0::detail::join_printer<CharT, FPack, Args...>
make_printer
    ( const FPack& fp
    , const stringify::v0::detail::joined_args<Args...>& x )
{
    return {fp, x};
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

