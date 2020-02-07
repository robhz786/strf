#ifndef STRF_DESTINATION_HPP
#define STRF_DESTINATION_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/tr_string.hpp>
#include <strf/facets_pack.hpp>

namespace strf {

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
    using destination_type_ = DestinationTmpl<OutbufCreator, FPack>;
public:

    template <typename ... FPE>
    STRF_NODISCARD constexpr STRF_HD auto with(FPE&& ... fpe) const &
    {
        static_assert( std::is_copy_constructible<OutbufCreator>::value
                     , "OutbufCreator must be copy constructible" );

        const auto& self = static_cast<const destination_type_&>(*this);

        using NewFPack = decltype( strf::pack( std::declval<const FPack&>()
                                             , std::forward<FPE>(fpe) ...) );

        return DestinationTmpl<OutbufCreator, NewFPack>
        { self, detail::destination_tag{}, std::forward<FPE>(fpe) ...};
    }

    template <typename ... FPE>
    STRF_NODISCARD constexpr STRF_HD auto with(FPE&& ... fpe) &&
    {
        static_assert( std::is_move_constructible<OutbufCreator>::value
                     , "OutbufCreator must be move constructible" );

        auto& self = static_cast<const destination_type_&>(*this);

        using NewFPack = decltype( strf::pack( std::declval<FPack>()
                                             , std::forward<FPE>(fpe) ...) );

        return DestinationTmpl<OutbufCreator, NewFPack>
        { std::move(self), detail::destination_tag{}, std::forward<FPE>(fpe) ...};
    }

    constexpr STRF_HD strf::destination_no_reserve<OutbufCreator, FPack>
    no_reserve() const &
    {
        const auto& self = static_cast<const destination_type_&>(*this);
        return { strf::detail::destination_tag{}
               , self.outbuf_creator_
               , self.fpack_ };
    }

    constexpr STRF_HD strf::destination_no_reserve<OutbufCreator, FPack>
    no_reserve() &&
    {
        auto& self = static_cast<destination_type_&>(*this);
        return { strf::detail::destination_tag{}
               , std::move(self.outbuf_creator_)
               , std::move(self.fpack_) };
    }

    constexpr STRF_HD strf::destination_calc_size<OutbufCreator, FPack>
    reserve_calc() const &
    {
        const auto& self = static_cast<const destination_type_&>(*this);
        return { strf::detail::destination_tag{}
               , self.outbuf_creator_
               , self.fpack_ };
    }

    strf::destination_calc_size<OutbufCreator, FPack>
    STRF_HD reserve_calc() &&
    {
        auto& self = static_cast<destination_type_&>(*this);
        return { strf::detail::destination_tag{}
               , std::move(self.outbuf_creator_)
               , std::move(self.fpack_) };
    }

    constexpr STRF_HD strf::destination_with_given_size<OutbufCreator, FPack>
    reserve(std::size_t size) const &
    {
        const auto& self = static_cast<const destination_type_&>(*this);
        return { strf::detail::destination_tag{}
               , size
               , self.outbuf_creator_
               , self.fpack_ };
    }

    constexpr STRF_HD strf::destination_with_given_size<OutbufCreator, FPack>
    reserve(std::size_t size) &&
    {
        auto& self = static_cast<destination_type_&>(*this);
        return { strf::detail::destination_tag{}
               , size
               , std::move(self.outbuf_creator_)
               , std::move(self.fpack_) };
    }

    template <typename ... Args>
    decltype(auto) STRF_HD operator()(const Args& ... args) const &
    {
        const auto& self = static_cast<const destination_type_&>(*this);
        PreviewType preview;
        return self.write_
            ( preview
            , as_printer_cref_(make_printer<CharT, FPack>( strf::rank<5>{}
                                                         , self.fpack_
                                                         , preview
                                                         , args ))... );
    }

#if defined(STRF_HAS_STD_STRING_VIEW)

    template <typename ... Args>
    decltype(auto) STRF_HD tr
        ( const std::basic_string_view<CharT>& str
        , const Args& ... args ) const &
    {
        return tr_write_(str.data(), str.size(), args...);
    }

#else

    template <typename ... Args>
    decltype(auto) STRF_HD tr(const CharT* str, const Args& ... args) const &
    {
        return tr_write_(str, std::char_traits<CharT>::length(str), args...);
    }

#endif

private:

    static inline const strf::printer<sizeof(CharT)>&
    STRF_HD as_printer_cref_(const strf::printer<sizeof(CharT)>& p)
    {
        return p;
    }
    static inline const strf::printer<sizeof(CharT)>*
    STRF_HD as_printer_cptr_(const strf::printer<sizeof(CharT)>& p)
    {
         return &p;
    }

    template < typename ... Args >
    decltype(auto) STRF_HD tr_write_( const CharT* str
                                    , std::size_t str_len
                                    , const Args& ... args) const &
    {
        return tr_write_2_
            (str, str + str_len, std::make_index_sequence<sizeof...(args)>(), args...);
    }

    template < std::size_t ... I, typename ... Args >
    decltype(auto) STRF_HD tr_write_2_( const CharT* str
                              , const CharT* str_end
                              , std::index_sequence<I...>
                              , const Args& ... args) const &
    {
        PreviewType preview_arr[sizeof...(args)];
        const auto& self = static_cast<const destination_type_&>(*this);
        return tr_write_3_
            ( str
            , str_end
            , preview_arr
            , { as_printer_cptr_( make_printer<CharT, FPack>( strf::rank<5>{}
                                                            , self.fpack_
                                                            , preview_arr[I]
                                                            , args ))... } );
    }

    template < typename Preview, typename ... Args >
    decltype(auto) STRF_HD tr_write_3_
        ( const CharT* str
        , const CharT* str_end
        , Preview* preview_arr
        , std::initializer_list<const strf::printer<sizeof(CharT)>*> args ) const &
    {
        const auto& self = static_cast<const destination_type_&>(*this);

        using catenc = strf::encoding_c<CharT>;
        decltype(auto) enc = strf::get_facet<catenc, void>(self.fpack_);

        using caterr = strf::tr_invalid_arg_c;
        decltype(auto) arg_err = strf::get_facet<caterr, void>(self.fpack_);

        using uchar_type = strf::underlying_char_type<sizeof(CharT)>;
        auto ustr = reinterpret_cast<const uchar_type*>(str);
        auto ustr_end = reinterpret_cast<const uchar_type*>(str_end);

        PreviewType preview;
        strf::detail::tr_string_printer<sizeof(CharT)> tr_printer
            (preview, preview_arr, args, ustr, ustr_end, enc, arg_err);

        return self.write_(preview, tr_printer);
    }
};

template < typename OB >
inline STRF_HD decltype(std::declval<OB&>().finish())
    finish(strf::rank<2>, OB& ob)
{
    return ob.finish();
}

template < typename OB >
inline STRF_HD void finish(strf::rank<1>, OB&)
{
}

}// namespace detail

template < typename OutbufCreator, typename FPack >
class destination_no_reserve
    : private strf::detail::destination_common
        < strf::destination_no_reserve
        , OutbufCreator
        , FPack
        , strf::print_preview<strf::preview_size::no, strf::preview_width::no> >
{
    using common_ = strf::detail::destination_common
        < strf::destination_no_reserve
        , OutbufCreator
        , FPack
        , strf::print_preview<strf::preview_size::no, strf::preview_width::no> >;

    template <template <typename, typename> class, class, class, class, class>
    friend class strf::detail::destination_common;

    using preview_type_
        = strf::print_preview<strf::preview_size::no, strf::preview_width::no>;

public:

    using char_type = typename OutbufCreator::char_type;

    template < typename ... Args
             , std::enable_if_t
                 < std::is_constructible<OutbufCreator, Args...>::value
                 , int > = 0 >
    constexpr STRF_HD destination_no_reserve(Args&&... args)
        : outbuf_creator_(std::forward<Args>(args)...)
    {
    }

    template < typename T = OutbufCreator
             , std::enable_if_t
                 < std::is_copy_constructible<T>::value, int > = 0 >
    constexpr STRF_HD destination_no_reserve( strf::detail::destination_tag
                                            , const OutbufCreator& oc
                                            , const FPack& fp )
        : outbuf_creator_(oc)
        , fpack_(fp)
    {
    }

    constexpr STRF_HD destination_no_reserve( strf::detail::destination_tag
                                            , OutbufCreator&& oc
                                            , FPack&& fp )
        : outbuf_creator_(std::move(oc))
        , fpack_(std::move(fp))
    {
    }

    using common_::with;
    using common_::operator();
    using common_::tr;
    using common_::reserve_calc;
    using common_::reserve;

    constexpr STRF_HD destination_no_reserve& no_reserve() &
    {
        return *this;
    }
    constexpr STRF_HD const destination_no_reserve& no_reserve() const &
    {
        return *this;
    }
    constexpr STRF_HD destination_no_reserve&& no_reserve() &&
    {
        return std::move(*this);
    }
    constexpr STRF_HD const destination_no_reserve&& no_reserve() const &&
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
    constexpr STRF_HD destination_no_reserve
        ( const destination_no_reserve<OutbufCreator, OtherFPack>& other
        , detail::destination_tag
        , FPE&& ... fpe )
        : outbuf_creator_(other.outbuf_creator_)
        , fpack_(other.fpack_, std::forward<FPE>(fpe)...)
    {
    }

    template < typename OtherFPack, typename ... FPE >
    constexpr STRF_HD destination_no_reserve
        ( destination_no_reserve<OutbufCreator, OtherFPack>&& other
        , detail::destination_tag
        , FPE&& ... fpe )
        : outbuf_creator_(std::move(other.outbuf_creator_))
        , fpack_(std::move(other.fpack_), std::forward<FPE>(fpe)...)
    {
    }

    template <typename ... Printers>
    decltype(auto) STRF_HD write_
        ( const preview_type_&
        , const Printers& ... printers) const
    {
        decltype(auto) ob = outbuf_creator_.create();
        strf::detail::write_args(ob.as_underlying(), printers...);
        return strf::detail::finish(strf::rank<2>(), ob);
    }

    OutbufCreator outbuf_creator_;
    FPack fpack_;
};

template < typename OutbufCreator, typename FPack >
class destination_with_given_size
    : public strf::detail::destination_common
        < strf::destination_with_given_size
        , OutbufCreator
        , FPack
        , strf::print_preview<strf::preview_size::no, strf::preview_width::no> >
{
    using common_ = strf::detail::destination_common
        < strf::destination_with_given_size
        , OutbufCreator
        , FPack
        , strf::print_preview<strf::preview_size::no, strf::preview_width::no> >;

    template < template <typename, typename> class, class,class, class, class>
    friend class strf::detail::destination_common;

    using preview_type_
        = strf::print_preview<strf::preview_size::no, strf::preview_width::no>;

public:

    using char_type = typename OutbufCreator::char_type;

    template < typename ... Args
             , std::enable_if_t
                 < std::is_constructible<OutbufCreator, Args...>::value
                 , int > = 0 >
    constexpr STRF_HD destination_with_given_size(std::size_t size, Args&&... args)
        : size_(size)
        , outbuf_creator_(std::forward<Args>(args)...)
    {
    }

    template < typename T = OutbufCreator
             , std::enable_if_t<std::is_copy_constructible<T>::value, int> = 0 >
    constexpr STRF_HD destination_with_given_size( strf::detail::destination_tag
                                                 , std::size_t size
                                                 , const OutbufCreator& oc
                                                 , const FPack& fp )
        : size_(size)
        , outbuf_creator_(oc)
        , fpack_(fp)
    {
    }

    constexpr STRF_HD destination_with_given_size( strf::detail::destination_tag
                                                 , std::size_t size
                                                 , OutbufCreator&& oc
                                                 , FPack&& fp )
        : size_(size)
        , outbuf_creator_(std::move(oc))
        , fpack_(std::move(fp))
    {
    }

    using common_::with;
    using common_::operator();
    using common_::tr;
    using common_::reserve_calc;
    using common_::no_reserve;

    constexpr STRF_HD destination_with_given_size& reserve(std::size_t size) &
    {
        size_ = size;
        return *this;
    }
    constexpr STRF_HD destination_with_given_size&& reserve(std::size_t size) &&
    {
        size_ = size;
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
    constexpr STRF_HD destination_with_given_size
        ( const destination_with_given_size<OutbufCreator, OtherFPack>& other
        , detail::destination_tag
        , FPE&& ... fpe )
        : size_(other.size_)
        , outbuf_creator_(other.outbuf_creator_)
        , fpack_(other.fpack_, std::forward<FPE>(fpe)...)
    {
    }

    template < typename OtherFPack, typename ... FPE >
    constexpr STRF_HD destination_with_given_size
        ( destination_with_given_size<OutbufCreator, OtherFPack>&& other
        , detail::destination_tag
        , FPE&& ... fpe )
        : size_(other.size)
        , outbuf_creator_(std::move(other.outbuf_creator_))
        , fpack_(std::move(other.fpack_), std::forward<FPE>(fpe)...)
    {
    }

    template <typename ... Printers>
    decltype(auto) STRF_HD write_
        ( const preview_type_&
        , const Printers& ... printers) const
    {
        decltype(auto) ob = outbuf_creator_.create(size_);
        strf::detail::write_args(ob.as_underlying(), printers...);
        return strf::detail::finish(strf::rank<2>(), ob);
    }

    std::size_t size_;
    OutbufCreator outbuf_creator_;
    FPack fpack_;
};

template < typename OutbufCreator, typename FPack >
class destination_calc_size
    : public strf::detail::destination_common
        < strf::destination_calc_size
        , OutbufCreator
        , FPack
        , strf::print_preview<strf::preview_size::yes, strf::preview_width::no> >
{
    using common_ = strf::detail::destination_common
        < strf::destination_calc_size
        , OutbufCreator
        , FPack
        , strf::print_preview<strf::preview_size::yes, strf::preview_width::no> >;

    template < template <typename, typename> class, class, class, class, class>
    friend class strf::detail::destination_common;

    using preview_type_
        = strf::print_preview<strf::preview_size::yes, strf::preview_width::no>;

public:

    using char_type = typename OutbufCreator::char_type;

    template < typename ... Args
             , std::enable_if_t
                 < std::is_constructible<OutbufCreator, Args...>::value
                 , int > = 0 >
    constexpr STRF_HD destination_calc_size(Args&&... args)
        : outbuf_creator_(std::forward<Args>(args)...)
    {
    }

    template < typename T = OutbufCreator
             , std::enable_if_t
                 < std::is_copy_constructible<T>::value, int > = 0 >
    constexpr STRF_HD destination_calc_size( strf::detail::destination_tag
                                           , const OutbufCreator& oc
                                           , const FPack& fp )
        : outbuf_creator_(oc)
        , fpack_(fp)
    {
    }

    constexpr STRF_HD destination_calc_size( strf::detail::destination_tag
                                           , OutbufCreator&& oc
                                           , FPack&& fp )
        : outbuf_creator_(std::move(oc))
        , fpack_(std::move(fp))
    {
    }

    using common_::with;
    using common_::operator();
    using common_::tr;
    using common_::no_reserve;
    using common_::reserve;

    constexpr STRF_HD const destination_calc_size & reserve_calc() const &
    {
        return *this;
    }
    constexpr STRF_HD destination_calc_size & reserve_calc() &
    {
        return *this;
    }
    constexpr STRF_HD const destination_calc_size && reserve_calc() const &&
    {
        return std::move(*this);
    }
    constexpr STRF_HD destination_calc_size && reserve_calc() &&
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
    STRF_HD destination_calc_size
        ( const destination_calc_size<OutbufCreator, OtherFPack>& other
        , detail::destination_tag
        , FPE&& ... fpe )
        : outbuf_creator_(other.outbuf_creator_)
        , fpack_(other.fpack_, std::forward<FPE>(fpe)...)
    {
    }

    template < typename OtherFPack, typename ... FPE >
    STRF_HD destination_calc_size
        ( destination_calc_size<OutbufCreator, OtherFPack>&& other
        , detail::destination_tag
        , FPE&& ... fpe )
        : outbuf_creator_(std::move(other.outbuf_creator_))
        , fpack_(std::move(other.fpack_), std::forward<FPE>(fpe)...)
    {
    }

    template <typename ... Printers>
    decltype(auto) STRF_HD write_
        ( const preview_type_& preview
        , const Printers& ... printers ) const
    {
        decltype(auto) ob = outbuf_creator_.create(preview.get_size());
        strf::detail::write_args(ob.as_underlying(), printers...);
        return strf::detail::finish(strf::rank<2>(), ob);
    }

    OutbufCreator outbuf_creator_;
    FPack fpack_;
};

template <typename CharOut, typename FPack, typename Preview, typename Arg>
inline STRF_HD auto make_printer
    ( strf::rank<1>
    , const FPack& fp
    , Preview& preview
    , std::reference_wrapper<Arg> arg)
{
    return make_printer<CharOut, FPack>
        ( strf::rank<5>{}, fp, preview, arg.get() );
}

template <typename CharOut, typename FPack, typename Preview, typename Arg>
using printer_impl
= decltype(make_printer<CharOut, FPack>( strf::rank<5>{}
                                       , std::declval<const FPack&>()
                                       , std::declval<Preview&>()
                                       , std::declval<const Arg&>() ) );
namespace detail {

template <typename CharT>
class outbuf_reference
{
public:

    using char_type = CharT;

    explicit STRF_HD outbuf_reference(strf::basic_outbuf<CharT>& ob) noexcept
        : ob_(ob)
    {
    }

    STRF_HD strf::basic_outbuf<CharT>& create() const
    {
        return ob_;
    }

private:
    strf::basic_outbuf<CharT>& ob_;
};


} // namespace detail

template <typename CharT>
auto STRF_HD to(strf::basic_outbuf<CharT>& ob)
{
    return strf::destination_no_reserve<strf::detail::outbuf_reference<CharT>>(ob);
}

} // namespace strf

#endif  // STRF_DESTINATION_HPP
