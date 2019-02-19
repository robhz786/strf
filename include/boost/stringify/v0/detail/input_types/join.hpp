#ifndef BOOST_STRINGIFY_V0_JOIN_HPP
#define BOOST_STRINGIFY_V0_JOIN_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/detail/facets/encoding.hpp>
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
        , stringify::v0::encoding<CharT> encoding
        , stringify::v0::encoding_policy epoli )
        : _join{j}
        , _args{args}
        , _encoding(encoding)
        , _epoli(epoli)
    {
        _fillcount = _remaining_width_from_arglist(_join.width);
    }

    join_printer_impl( const join_printer_impl& cp ) = delete;

    join_printer_impl
        ( const join_printer_impl& cp
        , const stringify::v0::detail::printer_ptr_range<CharT>& args )
        : _join{cp._join}
        , _args{args}
        , _encoding(cp._encoding)
        , _epoli(cp._epoli)
        , _fillcount(cp._fillcount)
    {
    }

    join_printer_impl
        ( join_printer_impl&& tmp
        , const stringify::v0::detail::printer_ptr_range<CharT>& args )
        : _join{std::move(tmp._join)}
        , _args{std::move(args)}
        , _encoding(tmp._encoding)
        , _epoli(tmp._epoli)
        , _fillcount(tmp._fillcount)
    {
    }

    ~join_printer_impl()
    {
    }

    std::size_t necessary_size() const override
    {
        return _args_length() + _fill_length();
    }

    bool write(stringify::v0::output_buffer<CharT>& ob) const override
    {
        if (_fillcount <= 0)
        {
            return _write_args(ob);
        }
        else
        {
            switch(_join.align)
            {
                case stringify::v0::alignment::left:
                {
                    return _write_args(ob)
                        && _write_fill(ob, _fillcount);
                }
                case stringify::v0::alignment::right:
                {
                    return _write_fill(ob, _fillcount)
                        && _write_args(ob);
                }
                case stringify::v0::alignment::internal:
                {
                    return _write_splitted(ob);
                }
                default:
                {
                    BOOST_ASSERT(_join.align == stringify::v0::alignment::center);
                    auto half_fillcount = _fillcount / 2;
                    return _write_fill(ob, half_fillcount)
                        && _write_args(ob)
                        && _write_fill(ob, _fillcount - half_fillcount);
                }
            }
        }
    }

    int remaining_width(int w) const override
    {
        if (_fillcount > 0)
        {
            return (std::max)(0, w - _join.width);
        }
        return _remaining_width_from_arglist(w);
    }


private:

    input_type _join;
    pp_range _args = nullptr;
    const stringify::v0::encoding<CharT> _encoding;
    stringify::v0::encoding_policy _epoli;
    int _fillcount = 0;

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
                                                   , _epoli.err_hdl());
        }
        return 0;
    }

    int _remaining_width_from_arglist(int w) const
    {
        for(auto it = _args.begin(); w > 0 && it != _args.end(); ++it)
        {
            w = (*it) -> remaining_width(w);
        }
        return w;
    }

    bool _write_splitted(stringify::v0::output_buffer<CharT>& ob) const
    {
        auto it = _args.begin();
        for ( int count = _join.num_leading_args
            ; count > 0 && it != _args.end()
            ; --count, ++it)
        {
            if (! (*it)->write(ob))
            {
                return false;
            }
        }
        if (! _write_fill(ob, _fillcount))
        {
            return false;
        }
        while(it != _args.end())
        {
            if (! (*it)->write(ob))
            {
                return false;
            }
            ++it;
        }
        return true;
    }

    bool _write_args(stringify::v0::output_buffer<CharT>& ob) const
    {
        for(const auto& arg : _args)
        {
            if (! arg->write(ob))
            {
                return false;
            }
        }
        return true;;
    }

    bool _write_fill
        ( stringify::v0::output_buffer<CharT>& ob
        , int count ) const
    {
        return stringify::v0::detail::write_fill
            ( _encoding, ob, count, _join.fillchar, _epoli.err_hdl() );
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

    using input_type  = stringify::v0::detail::join_t ;

    join_printer
        ( const FPack& fp
        , const stringify::v0::detail::joined_args<Args...>& ja )
        : _fmt_group(fp, ja.args)
        , _join_impl{ _fmt_group::range()
                    , ja.join
                    , stringify::v0::get_facet
                        < stringify::v0::encoding_category<CharT>, void > (fp)
                    , stringify::v0::get_facet
                        < stringify::v0::encoding_policy_category, void > (fp) }
    {
    }

    join_printer(const join_printer& cp)
        : _fmt_group(cp)
        , _join_impl(cp, _fmt_group::range())
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

