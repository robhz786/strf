#ifndef STRF_DESTINATION_HPP
#define STRF_DESTINATION_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/tr_string.hpp>
#include <strf/detail/printers_tuple.hpp>
#include <strf/facets_pack.hpp>

STRF_NAMESPACE_BEGIN

template < typename OutbufCreator
         , typename FPack = strf::facets_pack<> >
class destination_with_given_size;

template < typename OutbufCreator
         , typename FPack = strf::facets_pack<> >
class destination_calc_size;

template < typename OutbufCreator
         , typename FPack = strf::facets_pack<> >
class destination_no_reserve;

namespace detail {

struct destination_tag {};

template < template <typename, typename> class DestinationTmpl
         , class OutbufCreator
         , class FPack
         , class PreviewType
         , class CharT = typename OutbufCreator::char_type >
class destination_common
{
    using _destination_type = DestinationTmpl<OutbufCreator, FPack>;
public:

    template <typename ... FPE>
    STRF_NODISCARD constexpr auto with(FPE&& ... fpe) const &
    {
        static_assert( std::is_copy_constructible<OutbufCreator>::value
                     , "OutbufCreator must be copy constructible" );

        const auto& self = static_cast<const _destination_type&>(*this);

        using NewFPack = decltype
            ( strf::pack( std::declval<const FPack&>()
                                 , std::forward<FPE>(fpe) ...) );

        return DestinationTmpl<OutbufCreator, NewFPack>
        { self, detail::destination_tag{}, std::forward<FPE>(fpe) ...};
    }

    template <typename ... FPE>
    STRF_NODISCARD constexpr auto with(FPE&& ... fpe) &&
    {
        static_assert( std::is_move_constructible<OutbufCreator>::value
                     , "OutbufCreator must be move constructible" );

        auto& self = static_cast<const _destination_type&>(*this);

        using NewFPack = decltype
            ( strf::pack( std::declval<FPack>()
                                 , std::forward<FPE>(fpe) ...) );

        return DestinationTmpl<OutbufCreator, NewFPack>
        { std::move(self), detail::destination_tag{}, std::forward<FPE>(fpe) ...};
    }

    constexpr strf::destination_no_reserve<OutbufCreator, FPack>
    no_reserve() const &
    {
        const auto& self = static_cast<const _destination_type&>(*this);
        return { strf::detail::destination_tag{}
               , self._outbuf_creator
               , self._fpack };
    }

    constexpr strf::destination_no_reserve<OutbufCreator, FPack>
    no_reserve() &&
    {
        auto& self = static_cast<_destination_type&>(*this);
        return { strf::detail::destination_tag{}
               , std::move(self._outbuf_creator)
               , std::move(self._fpack) };
    }

    constexpr strf::destination_calc_size<OutbufCreator, FPack>
    reserve_calc() const &
    {
        const auto& self = static_cast<const _destination_type&>(*this);
        return { strf::detail::destination_tag{}
               , self._outbuf_creator
               , self._fpack };
    }

    strf::destination_calc_size<OutbufCreator, FPack>
    reserve_calc() &&
    {
        auto& self = static_cast<_destination_type&>(*this);
        return { strf::detail::destination_tag{}
               , std::move(self._outbuf_creator)
               , std::move(self._fpack) };
    }

    constexpr strf::destination_with_given_size<OutbufCreator, FPack>
    reserve(std::size_t size) const &
    {
        const auto& self = static_cast<const _destination_type&>(*this);
        return { strf::detail::destination_tag{}
               , size
               , self._outbuf_creator
               , self._fpack };
    }

    constexpr strf::destination_with_given_size<OutbufCreator, FPack>
    reserve(std::size_t size) &&
    {
        auto& self = static_cast<_destination_type&>(*this);
        return { strf::detail::destination_tag{}
               , size
               , std::move(self._outbuf_creator)
               , std::move(self._fpack) };
    }

    template <typename ... Args>
    decltype(auto) operator()(const Args& ... args) const &
    {
        const auto& self = static_cast<const _destination_type&>(*this);
        PreviewType preview;
        return self._write
            ( preview
            , _as_printer_cref(make_printer<CharT, FPack>( self._fpack
                                                         , preview
                                                         , args ))... );
    }

#if defined(STRF_HAS_STD_STRING_VIEW)

    template <typename ... Args>
    decltype(auto) tr
        ( const std::basic_string_view<CharT>& str
        , const Args& ... args ) const &
    {

       _tr_write(str.begin, str.end(), args...);
    }

    // template <typename ... Args>
    // decltype(auto) tr
    //     ( const std::basic_string_view<CharT>& str
    //     , const Args& ... args ) &&
    // {
    //     return std::move(*this)._tr_write(str.begin, str.end(), args...);
    // }

#else

    template <typename ... Args>
    decltype(auto) tr(const CharT* str, const Args& ... args) const &
    {
        return _tr_write
            ( str, str + std::char_traits<CharT>::length(str), args... );
    }

    // template <typename ... Args>
    // decltype(auto) tr(const CharT* str, const Args& ... args) &&
    // {
    //     return std::move(*this)._tr_write
    //         ( str, str + std::char_traits<CharT>::length(str), args... );
    // }

#endif

private:

    static inline const strf::printer<CharT>&
    _as_printer_cref(const strf::printer<CharT>& p)
    {
        return p;
    }
    static inline const strf::printer<CharT>*
    _as_printer_cptr(const strf::printer<CharT>& p)
    {
         return &p;
    }

    template < typename ... Args >
    decltype(auto) _tr_write( const CharT* str
                            , const CharT* str_end
                            , const Args& ... args) const &
    {
        return _tr_write_2
            (str, str_end, std::make_index_sequence<sizeof...(args)>(), args...);
    }

    template < std::size_t ... I, typename ... Args >
    decltype(auto) _tr_write_2( const CharT* str
                              , const CharT* str_end
                              , std::index_sequence<I...>
                              , const Args& ... args) const &
    {
        PreviewType preview_arr[sizeof...(args)];
        const auto& self = static_cast<const _destination_type&>(*this);
        return _tr_write_3
            ( str
            , str_end
            , preview_arr
            , { _as_printer_cptr( make_printer<CharT, FPack>( self._fpack
                                                            , preview_arr[I]
                                                            , args ))... } );
    }

    template < typename Preview, typename ... Args >
    decltype(auto) _tr_write_3
        ( const CharT* str
        , const CharT* str_end
        , Preview* preview_arr
        , std::initializer_list<const strf::printer<CharT>*> args ) const &
    {
        const auto& self = static_cast<const _destination_type&>(*this);

        using catenc = strf::encoding_c<CharT>;
        using caterr = strf::tr_invalid_arg_c;
        decltype(auto) enc = strf::get_facet<catenc, void>(self._fpack);
        decltype(auto) arg_err = strf::get_facet<caterr, void>(self._fpack);

        PreviewType preview;
        strf::detail::tr_string_printer<CharT> tr_printer
            (preview, preview_arr, args, str, str_end, enc, arg_err);

        return self._write(preview, tr_printer);
    }
};

}// namespace detail

template < typename OutbufCreator, typename FPack >
class destination_no_reserve
    : private strf::detail::destination_common
        < strf::destination_no_reserve
        , OutbufCreator
        , FPack
        , strf::print_preview<false, false> >
{
    using _common = strf::detail::destination_common
        < strf::destination_no_reserve
        , OutbufCreator
        , FPack
        , strf::print_preview<false, false> >;

    template <template <typename, typename> class, class, class, class, class>
    friend class strf::detail::destination_common;

    using _preview_type = strf::print_preview<false, false>;

public:

    using char_type = typename OutbufCreator::char_type;

    template < typename ... Args
             , std::enable_if_t
                 < std::is_constructible<OutbufCreator, Args...>::value
                 , int > = 0 >
    constexpr destination_no_reserve(Args&&... args)
        : _outbuf_creator(std::forward<Args>(args)...)
    {
    }

    template < typename T = OutbufCreator
             , std::enable_if_t
                 < std::is_copy_constructible<T>::value, int > = 0 >
    constexpr destination_no_reserve( strf::detail::destination_tag
                                    , const OutbufCreator& oc
                                    , const FPack& fp )
        : _outbuf_creator(oc)
        , _fpack(fp)
    {
    }

    constexpr destination_no_reserve( strf::detail::destination_tag
                                    , OutbufCreator&& oc
                                    , FPack&& fp )
        : _outbuf_creator(std::move(oc))
        , _fpack(std::move(fp))
    {
    }

    constexpr destination_no_reserve(const destination_no_reserve&) = default;
    constexpr destination_no_reserve(destination_no_reserve&&) = default;

    using _common::with;
    using _common::operator();
    using _common::tr;
    using _common::reserve_calc;
    using _common::reserve;

    constexpr destination_no_reserve& no_reserve() &
    {
        return *this;
    }
    constexpr const destination_no_reserve& no_reserve() const &
    {
        return *this;
    }
    constexpr destination_no_reserve&& no_reserve() &&
    {
        return std::move(*this);
    }
    constexpr const destination_no_reserve&& no_reserve() const &&
    {
        return std::move(*this);
    }

private:

    template <class, class>
    friend class destination_no_reserve;

    template < typename OtherFPack
             , typename ... FPE
             , typename T = OutbufCreator
             , typename = std::enable_if_t
                 < std::is_copy_constructible<T>::value > >
    constexpr destination_no_reserve
        ( const destination_no_reserve<OutbufCreator, OtherFPack>& other
        , detail::destination_tag
        , FPE&& ... fpe )
        : _outbuf_creator(other._outbuf_creator)
        , _fpack(other._fpack, std::forward<FPE>(fpe)...)
    {
    }

    template < typename OtherFPack, typename ... FPE >
    constexpr destination_no_reserve
        ( destination_no_reserve<OutbufCreator, OtherFPack>&& other
        , detail::destination_tag
        , FPE&& ... fpe )
        : _outbuf_creator(std::move(other._outbuf_creator))
        , _fpack(std::move(other._fpack), std::forward<FPE>(fpe)...)
    {
    }

    template <typename ... Printers>
    decltype(auto) _write
        ( const strf::print_preview<false, false>&
        , const Printers& ... printers) const
    {
        decltype(auto) ob = _outbuf_creator.create();
        strf::detail::write_args(ob, printers...);
        return ob.finish();
    }

    OutbufCreator _outbuf_creator;
    FPack _fpack;
};

template < typename OutbufCreator, typename FPack >
class destination_with_given_size
    : public strf::detail::destination_common
        < strf::destination_with_given_size
        , OutbufCreator
        , FPack
        , strf::print_preview<false, false> >
{
    using _common = strf::detail::destination_common
        < strf::destination_with_given_size
        , OutbufCreator
        , FPack
        , strf::print_preview<false, false> >;

    template < template <typename, typename> class, class,class, class, class>
    friend class strf::detail::destination_common;

    using _preview_type = strf::print_preview<false, false>;

public:

    using char_type = typename OutbufCreator::char_type;

    template < typename ... Args
             , std::enable_if_t
                 < std::is_constructible<OutbufCreator, Args...>::value
                 , int > = 0 >
    constexpr destination_with_given_size(std::size_t size, Args&&... args)
        : _size(size)
        , _outbuf_creator(std::forward<Args>(args)...)
    {
    }

    template < typename T = OutbufCreator
             , std::enable_if_t<std::is_copy_constructible<T>::value, int> = 0 >
    constexpr destination_with_given_size( strf::detail::destination_tag
                                         , std::size_t size
                                         , const OutbufCreator& oc
                                         , const FPack& fp )
        : _size(size)
        , _outbuf_creator(oc)
        , _fpack(fp)
    {
    }

    constexpr destination_with_given_size( strf::detail::destination_tag
                                         , std::size_t size
                                         , OutbufCreator&& oc
                                         , FPack&& fp )
        : _size(size)
        , _outbuf_creator(std::move(oc))
        , _fpack(std::move(fp))
    {
    }

    constexpr destination_with_given_size(const destination_with_given_size&) = default;
    constexpr destination_with_given_size(destination_with_given_size&&) = default;

    using _common::with;
    using _common::operator();
    using _common::tr;
    using _common::reserve_calc;
    using _common::no_reserve;

    constexpr destination_with_given_size& reserve(std::size_t size) &
    {
        _size = size;
        return *this;
    }
    constexpr destination_with_given_size&& reserve(std::size_t size) &&
    {
        _size = size;
        return std::move(*this);
    }

private:

    template <class, class>
    friend class destination_with_given_size;

    template < typename OtherFPack
             , typename ... FPE
             , typename T = OutbufCreator
             , typename = std::enable_if_t
                 < std::is_copy_constructible<T>::value > >
    constexpr destination_with_given_size
        ( const destination_with_given_size<OutbufCreator, OtherFPack>& other
        , detail::destination_tag
        , FPE&& ... fpe )
        : _size(other._size)
        , _outbuf_creator(other._outbuf_creator)
        , _fpack(other._fpack, std::forward<FPE>(fpe)...)
    {
    }

    template < typename OtherFPack, typename ... FPE >
    constexpr destination_with_given_size
        ( destination_with_given_size<OutbufCreator, OtherFPack>&& other
        , detail::destination_tag
        , FPE&& ... fpe )
        : _size(other.size)
        , _outbuf_creator(std::move(other._outbuf_creator))
        , _fpack(std::move(other._fpack), std::forward<FPE>(fpe)...)
    {
    }

    template <typename ... Printers>
    decltype(auto) _write
        ( const strf::print_preview<false, false>&
        , const Printers& ... printers) const
    {
        decltype(auto) ob = _outbuf_creator.create(_size);
        strf::detail::write_args(ob, printers...);
        return ob.finish();
    }

    std::size_t _size;
    OutbufCreator _outbuf_creator;
    FPack _fpack;
};

template < typename OutbufCreator, typename FPack >
class destination_calc_size
    : public strf::detail::destination_common
        < strf::destination_calc_size
        , OutbufCreator
        , FPack
        , strf::print_preview<true, false> >
{
    using _common = strf::detail::destination_common
        < strf::destination_calc_size
        , OutbufCreator
        , FPack
        , strf::print_preview<true, false> >;

    template < template <typename, typename> class, class, class, class, class>
    friend class strf::detail::destination_common;

    using _preview_type = strf::print_preview<true, false>;

public:

    using char_type = typename OutbufCreator::char_type;

    template < typename ... Args
             , std::enable_if_t
                 < std::is_constructible<OutbufCreator, Args...>::value
                 , int > = 0 >
    constexpr destination_calc_size(Args&&... args)
        : _outbuf_creator(std::forward<Args>(args)...)
    {
    }

    template < typename T = OutbufCreator
             , std::enable_if_t
                 < std::is_copy_constructible<T>::value, int > = 0 >
    constexpr destination_calc_size( strf::detail::destination_tag
                                   , const OutbufCreator& oc
                                   , const FPack& fp )
        : _outbuf_creator(oc)
        , _fpack(fp)
    {
    }

    constexpr destination_calc_size( strf::detail::destination_tag
                                   , OutbufCreator&& oc
                                   , FPack&& fp )
        : _outbuf_creator(std::move(oc))
        , _fpack(std::move(fp))
    {
    }

    constexpr destination_calc_size(const destination_calc_size&) = default;
    constexpr destination_calc_size(destination_calc_size&&) = default;

    using _common::with;
    using _common::operator();
    using _common::tr;
    using _common::no_reserve;
    using _common::reserve;

    constexpr const destination_calc_size & reserve_calc() const &
    {
        return *this;
    }
    constexpr destination_calc_size & reserve_calc() &
    {
        return *this;
    }
    constexpr const destination_calc_size && reserve_calc() const &&
    {
        return std::move(*this);
    }
    constexpr destination_calc_size && reserve_calc() &&
    {
        return std::move(*this);
    }

private:

    template <typename, typename>
    friend class destination_calc_size;

    template < typename OtherFPack
             , typename ... FPE
             , typename T = OutbufCreator
             , typename = std::enable_if_t
                 < std::is_copy_constructible<T>::value > >
    destination_calc_size
        ( const destination_calc_size<OutbufCreator, OtherFPack>& other
        , detail::destination_tag
        , FPE&& ... fpe )
        : _outbuf_creator(other._outbuf_creator)
        , _fpack(other._fpack, std::forward<FPE>(fpe)...)
    {
    }

    template < typename OtherFPack, typename ... FPE >
    destination_calc_size
        ( destination_calc_size<OutbufCreator, OtherFPack>&& other
        , detail::destination_tag
        , FPE&& ... fpe )
        : _outbuf_creator(std::move(other._outbuf_creator))
        , _fpack(std::move(other._fpack), std::forward<FPE>(fpe)...)
    {
    }

    template <typename ... Printers>
    decltype(auto) _write
        ( const strf::print_preview<true, false>& preview
        , const Printers& ... printers ) const
    {
        decltype(auto) ob = _outbuf_creator.create(preview.get_size());
        strf::detail::write_args(ob, printers...);
        return ob.finish();
    }

    OutbufCreator _outbuf_creator;
    FPack _fpack;
};


template <typename CharOut, typename FPack, typename Preview, typename Arg>
using printer_impl
= decltype(make_printer<CharOut, FPack>( std::declval<const FPack&>()
                                       , std::declval<Preview&>()
                                       , std::declval<const Arg&>() ) );

STRF_NAMESPACE_END

#endif  // STRF_DESTINATION_HPP
