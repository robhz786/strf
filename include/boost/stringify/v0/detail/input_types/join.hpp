#ifndef BOOST_STRINGIFY_V0_JOIN_HPP
#define BOOST_STRINGIFY_V0_JOIN_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/detail/facets/encoding.hpp>

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
        return _begin;
    }
    const fmt_ptr* end() const
    {
        return _end;
    }
    virtual std::size_t size() const
    {
        return _end - _begin;
    }

    fmt_ptr* _begin = nullptr;
    fmt_ptr* _end = nullptr;
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
        : _printer(std::move(rval._printer))
        , _rest(std::move(rval._rest))
    {
    }

    printers_tuple
        ( const FPack& fp
        , const stringify::v0::detail::args_tuple<Arg, Args...>& args )
        : _printer(make_printer<CharT, FPack>(fp, args.first_arg))
        , _rest(fp, args.remove_first())
    {
    }

    using fmt_ptr = const stringify::v0::printer<CharT>*;

    fmt_ptr* fill(fmt_ptr* out_it) const
    {
        *out_it = &_printer;
        return _rest.fill(++out_it);
    }

private:

    printer_type _printer;
    printers_tuple<CharT, FPack, Args...> _rest;
};



template <typename CharT, typename FPack, typename ... Args>
class printers_group
{
public:

    printers_group
        ( const FPack& fp
        , const stringify::v0::detail::args_tuple<Args...>& args )
        : _impl(fp, args)
    {
        _range._end = _impl.fill(_array);
        _range._begin = _array;
    }

    printers_group(const printers_group& cp)
        : _impl(cp._impl)
    {
        _range._end = _impl.fill(_array);
        _range._begin = _array;
    }

    printers_group(printers_group&& rval)
        : _impl(std::move(rval._impl))
    {
        _range._end = _impl.fill(_array);
        _range._begin = _array;
    }


    virtual ~printers_group()
    {
    }

    const auto& range() const
    {
        return _range;
    }

private:

    stringify::v0::detail::printers_tuple<CharT, FPack, Args...> _impl;

    using _printer_ptr = const stringify::v0::printer<CharT>*;
    _printer_ptr _array[sizeof...(Args)];
    stringify::v0::detail::printer_ptr_range<CharT> _range;
};

} // namespace detail


template <typename ... Args>
struct join_t
{
    stringify::v0::detail::args_tuple<Args...> args;
};

struct aligned_join_t;

template <typename ... Args>
struct aligned_joined_args
{
    const stringify::v0::aligned_join_t& join;
    stringify::v0::detail::args_tuple<Args...> args;
};


struct aligned_join_t
{
    int width = 0;
    stringify::v0::text_alignment align = stringify::v0::text_alignment::right;
    char32_t fillchar = U' ';
    int num_leading_args = 1;

    template <typename ... Args>
    stringify::v0::aligned_joined_args<Args...> operator()
        (const Args& ... args) const
    {
        return {*this, {args...}};
    }
};

namespace detail {

template <typename CharT>
class aligned_join_printer_impl: public printer<CharT>
{
    using printer_type = stringify::v0::printer<CharT>;
    using pp_range = stringify::v0::detail::printer_ptr_range<CharT>;

public:

    using input_type  = stringify::v0::aligned_join_t ;

    aligned_join_printer_impl
        ( const stringify::v0::detail::printer_ptr_range<CharT>& args
        , const stringify::v0::aligned_join_t& j
        , stringify::v0::encoding<CharT> encoding
        , stringify::v0::encoding_error enc_err
        , stringify::v0::surrogate_policy allow_surr )
        : _join{j}
        , _args{args}
        , _encoding(encoding)
        , _enc_err(enc_err)
        , _allow_surr(allow_surr)
    {
        auto w = _arglist_width(_join.width);
        _fillcount = ( _join.width > w
                     ? _join.width - w
                     : 0 );
    }

    aligned_join_printer_impl( const aligned_join_printer_impl& cp ) = delete;

    aligned_join_printer_impl
        ( const aligned_join_printer_impl& cp
        , const stringify::v0::detail::printer_ptr_range<CharT>& args )
        : _join{cp._join}
        , _args{args}
        , _encoding(cp._encoding)
        , _fillcount(cp._fillcount)
        , _enc_err(cp._enc_err)
        , _allow_surr(cp._allow_surr)
    {
    }

    aligned_join_printer_impl
        ( aligned_join_printer_impl&& tmp
        , const stringify::v0::detail::printer_ptr_range<CharT>& args )
        : _join{std::move(tmp._join)}
        , _args{std::move(args)}
        , _encoding(tmp._encoding)
        , _fillcount(tmp._fillcount)
        , _enc_err(tmp._enc_err)
        , _allow_surr(tmp._allow_surr)
    {
    }

    ~aligned_join_printer_impl()
    {
    }

    std::size_t necessary_size() const override
    {
        return _args_length() + _fill_length();
    }

    void write(stringify::v0::basic_outbuf<CharT>& ob) const override
    {
        if (_fillcount <= 0)
        {
            return _write_args(ob);
        }
        else
        {
            switch(_join.align)
            {
                case stringify::v0::text_alignment::left:
                {
                    _write_args(ob);
                    _write_fill(ob, _fillcount);
                    break;
                }
                case stringify::v0::text_alignment::right:
                {
                    _write_fill(ob, _fillcount);
                    _write_args(ob);
                    break;
                }
                case stringify::v0::text_alignment::split:
                {
                    _write_split(ob);
                    break;
                }
                default:
                {
                    BOOST_ASSERT(_join.align == stringify::v0::text_alignment::center);
                    auto half_fillcount = _fillcount / 2;
                    _write_fill(ob, half_fillcount);
                    _write_args(ob);
                    _write_fill(ob, _fillcount - half_fillcount);
                    break;
                }
            }
        }
    }

    int width(int limit) const override
    {
        if (_fillcount > 0)
        {
            return _join.width;
        }
        return _arglist_width(limit);
    }
private:

    input_type _join;
    pp_range _args = nullptr;
    const stringify::v0::encoding<CharT> _encoding;
    int _fillcount = 0;
    stringify::v0::encoding_error _enc_err;
    stringify::v0::surrogate_policy _allow_surr;

    std::size_t _args_length() const
    {
        std::size_t sum = 0;
        for(const auto* arg : _args)
        {
            sum += arg->necessary_size();
        }
        return sum;
    }

    std::size_t _fill_length() const
    {
        if(_fillcount > 0)
        {
            return _fillcount * _encoding.char_size( _join.fillchar
                                                   , _enc_err);
        }
        return 0;
    }

    int _arglist_width(int limit) const
    {
        int sum = 0;
        for(auto it = _args.begin(); sum < limit && it != _args.end(); ++it)
        {
            sum += (*it) -> width(limit - sum);
        }
        return sum;
    }

    void _write_split(stringify::v0::basic_outbuf<CharT>& ob) const
    {
        auto it = _args.begin();
        for ( int count = _join.num_leading_args
            ; count > 0 && it != _args.end()
            ; --count, ++it)
        {
            (*it)->write(ob);
        }
        _write_fill(ob, _fillcount);
        while(it != _args.end())
        {
            (*it)->write(ob);
            ++it;
        }
    }

    void _write_args(stringify::v0::basic_outbuf<CharT>& ob) const
    {
        for(const auto& arg : _args)
        {
            arg->write(ob);
        }
    }

    void _write_fill( stringify::v0::basic_outbuf<CharT>& ob
                    , int count ) const
    {
        _encoding.encode_fill( ob, count, _join.fillchar
                             , _enc_err, _allow_surr );
    }
};


template <typename CharT, typename FPack, typename ... Args>
class aligned_join_printer
    : private stringify::v0::detail::printers_group<CharT, FPack, Args...>
    , public stringify::v0::detail::aligned_join_printer_impl<CharT>
{
    using _fmt_group
    = stringify::v0::detail::printers_group<CharT, FPack, Args...>;

    using _aligned_join_impl
    = stringify::v0::detail::aligned_join_printer_impl<CharT>;

    template <typename Category>
    static decltype(auto) _get_facet(const FPack& fp)
    {
        return fp.template get_facet<Category, stringify::v0::aligned_join_t>();
    }

public:

    aligned_join_printer
        ( const FPack& fp
        , const stringify::v0::aligned_joined_args<Args...>& ja )
        : _fmt_group(fp, ja.args)
        , _aligned_join_impl
            ( _fmt_group::range()
            , ja.join
            , _get_facet<stringify::v0::encoding_c<CharT>>(fp)
            , _get_facet<stringify::v0::encoding_error_c>(fp)
            , _get_facet<stringify::v0::surrogate_policy_c>(fp) )
    {
    }

    aligned_join_printer(const aligned_join_printer& cp)
        : _fmt_group(cp)
        , _aligned_join_impl(cp, _fmt_group::range())
    {
    }

    virtual ~aligned_join_printer()
    {
    }
};


template <typename CharT>
class join_printer_impl: public printer<CharT>
{
    using printer_type = stringify::v0::printer<CharT>;
    using pp_range = stringify::v0::detail::printer_ptr_range<CharT>;

public:

    join_printer_impl
        ( const stringify::v0::detail::printer_ptr_range<CharT>& args )
        : _args{args}
    {
    }

    ~join_printer_impl()
    {
    }

    std::size_t necessary_size() const override
    {
        return _args_length();
    }

    void write(stringify::v0::basic_outbuf<CharT>& ob) const override
    {
        return _write_args(ob);
    }

    int width(int limit) const override
    {
        return _arglist_width(limit);
    }

private:

    pp_range _args = nullptr;

    std::size_t _args_length() const
    {
        std::size_t sum = 0;
        for(const auto* arg : _args)
        {
            sum += arg->necessary_size();
        }
        return sum;
    }

    int _arglist_width(int limit) const
    {
        int sum = 0;
        for(auto it = _args.begin(); sum < limit && it != _args.end(); ++it)
        {
            sum += (*it) -> width(limit - sum);
        }
        return sum;
    }

    void _write_args(stringify::v0::basic_outbuf<CharT>& ob) const
    {
        for(const auto& arg : _args)
        {
            arg->write(ob);
        }
    }
};

template <typename CharT, typename FPack, typename ... Args>
class join_printer
    : private stringify::v0::detail::printers_group<CharT, FPack, Args...>
    , public stringify::v0::detail::join_printer_impl<CharT>
{
    using _fmt_group
    = stringify::v0::detail::printers_group<CharT, FPack, Args...>;

    using _join_impl
    = stringify::v0::detail::join_printer_impl<CharT>;

public:

    join_printer
        ( const FPack& fp
        , const stringify::v0::join_t<Args...>& j )
        : _fmt_group(fp, j.args)
        , _join_impl( _fmt_group::range() )
    {
    }

    join_printer(const join_printer& cp)
        : _fmt_group(cp)
        , _join_impl(_fmt_group::range())
    {
    }

    virtual ~join_printer()
    {
    }
};

} // namespace detail

template <typename CharT, typename FPack, typename... Args>
inline stringify::v0::detail::join_printer<CharT, FPack, Args...>
make_printer( const FPack& fp
            , const stringify::v0::join_t<Args...>& args )
{
    return {fp, args};
}

template <typename ... Args>
stringify::v0::join_t<Args...> join(const Args& ... args)
{
    return {{args...}};
}

template <typename CharT, typename FPack, typename ... Args>
inline stringify::v0::detail::aligned_join_printer<CharT, FPack, Args...>
make_printer
    ( const FPack& fp
    , const stringify::v0::aligned_joined_args<Args...>& x )
{
    return {fp, x};
}

constexpr stringify::v0::aligned_join_t
join_align( int width
          , stringify::v0::text_alignment align
          , char32_t fillchar = U' '
          , int num_leading_args = 0 )
{
    return {width, align, fillchar, num_leading_args};
}

constexpr stringify::v0::aligned_join_t
join_center(int width, char32_t fillchar = U' ') noexcept
{
    return {width, stringify::v0::text_alignment::center, fillchar, 0};
}

constexpr stringify::v0::aligned_join_t
join_left(int width, char32_t fillchar = U' ') noexcept
{
    return {width, stringify::v0::text_alignment::left, fillchar, 0};
}


constexpr stringify::v0::aligned_join_t
join_right(int width, char32_t fillchar = U' ') noexcept
{
    return {width, stringify::v0::text_alignment::right, fillchar, 0};
}

constexpr stringify::v0::aligned_join_t
join_split(int width, char32_t fillchar, int num_leading_args) noexcept
{
    return {width, stringify::v0::text_alignment::split, fillchar, num_leading_args};
}

constexpr stringify::v0::aligned_join_t
join_split(int width, int num_leading_args) noexcept
{
    return {width, stringify::v0::text_alignment::split, U' ', num_leading_args};
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_JOIN_HPP

