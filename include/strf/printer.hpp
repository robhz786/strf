#ifndef STRF_PRINTER_HPP
#define STRF_PRINTER_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/printable_traits.hpp>
#include <strf/detail/tr_printer.hpp>

namespace strf {

template < typename DestinationCreator
         , typename FPack = strf::facets_pack<> >
class printer_with_given_size;

template < typename DestinationCreator
         , typename FPack = strf::facets_pack<> >
class printer_with_size_calc;

template < typename DestinationCreator
         , typename FPack = strf::facets_pack<> >
class printer_no_reserve;

namespace detail {

template <typename Dest>
inline STRF_HD decltype(std::declval<Dest&>().finish())
    finish(strf::rank<2>, Dest& dest)
{
    return dest.finish();
}

template <typename Dest>
inline STRF_HD void finish(strf::rank<1>, Dest&)
{
}

template <typename DestinationCreator, bool Sized>
struct destination_creator_traits;

template <typename DestinationCreator>
struct destination_creator_traits<DestinationCreator, false>
{
    using destination_type = typename DestinationCreator::destination_type;
    using finish_return_type =
        decltype(strf::detail::finish(strf::rank<2>(), std::declval<destination_type&>()));
};

template <typename DestinationCreator>
struct destination_creator_traits<DestinationCreator, true>
{
    using destination_type = typename DestinationCreator::sized_destination_type;
    using finish_return_type =
        decltype(strf::detail::finish(strf::rank<2>(), std::declval<destination_type&>()));
};

template <typename DestinationCreator, bool Sized>
using destination_finish_return_type = typename
    destination_creator_traits<DestinationCreator, Sized>::finish_return_type;

struct destination_tag {};

template < template <typename, typename> class DestinationTmpl
         , bool Sized, class DestinationCreator, class PrePrinting, class FPack >
class printer_common
{
    using destination_type_ = DestinationTmpl<DestinationCreator, FPack>;

    using char_type_ = typename DestinationCreator::char_type;

    template <typename Arg>
    using stringifier_ = strf::stringifier_type<char_type_, PrePrinting, FPack, Arg>;

    using finish_return_type_ = destination_finish_return_type<DestinationCreator, Sized>;

    template <typename... FPE>
    using with_return_type =
        DestinationTmpl
            < DestinationCreator
            , decltype(strf::pack(std::declval<const FPack&>(), std::declval<FPE>()...)) >;

public:

    template <typename... FPE>
    STRF_NODISCARD constexpr STRF_HD auto with(FPE&&... fpe) const &
        noexcept(noexcept(with_return_type<FPE...>
                          ( std::declval<const destination_type_&>()
                          , detail::destination_tag()
                          , std::declval<FPE>()... )))
        -> with_return_type<FPE...>
    {
        static_assert( std::is_copy_constructible<DestinationCreator>::value
                     , "DestinationCreator must be copy constructible" );

        return { static_cast<const destination_type_&>(*this)
               , detail::destination_tag{}, std::forward<FPE>(fpe) ...};
    }

    template <typename... FPE>
    STRF_NODISCARD STRF_CONSTEXPR_IN_CXX14 STRF_HD auto with(FPE&& ... fpe) &&
        noexcept(noexcept(with_return_type<FPE...>
                          ( std::declval<destination_type_>()
                          , detail::destination_tag()
                          , std::declval<FPE>()... )))
        -> with_return_type<FPE...>
    {
        static_assert( std::is_move_constructible<DestinationCreator>::value
                     , "DestinationCreator must be move constructible" );

        return { std::move(static_cast<const destination_type_&>(*this))
               , detail::destination_tag{}
               , std::forward<FPE>(fpe)...};
    }

    constexpr STRF_HD strf::printer_no_reserve<DestinationCreator, FPack>
    no_reserve() const &
        noexcept(noexcept(strf::printer_no_reserve<DestinationCreator, FPack>
                          ( strf::detail::destination_tag()
                          , std::declval<const DestinationCreator&>()
                          , std::declval<const FPack&>() )))
    {
        return { strf::detail::destination_tag{}
               , static_cast<const destination_type_*>(this)->destination_creator_
               , static_cast<const destination_type_*>(this)->fpack_ };
    }

    STRF_CONSTEXPR_IN_CXX14 STRF_HD strf::printer_no_reserve<DestinationCreator, FPack>
    no_reserve() &&
        noexcept(noexcept(strf::printer_no_reserve<DestinationCreator, FPack>
                          ( strf::detail::destination_tag()
                          , std::declval<DestinationCreator>()
                          , std::declval<FPack>() )))
    {
        return { strf::detail::destination_tag{}
               , std::move(static_cast<destination_type_*>(this)->destination_creator_)
               , std::move(static_cast<destination_type_*>(this)->fpack_) };
    }

    constexpr STRF_HD strf::printer_with_size_calc<DestinationCreator, FPack>
    reserve_calc() const &
        noexcept(noexcept(strf::printer_with_size_calc<DestinationCreator, FPack>
                          ( strf::detail::destination_tag()
                          , std::declval<const DestinationCreator&>()
                          , std::declval<const FPack&>() )))
    {
        return { strf::detail::destination_tag{}
               , static_cast<const destination_type_*>(this)->destination_creator_
               , static_cast<const destination_type_*>(this)->fpack_ };
    }

    STRF_CONSTEXPR_IN_CXX14 strf::printer_with_size_calc<DestinationCreator, FPack>
    STRF_HD reserve_calc() &&
        noexcept(noexcept(strf::printer_with_size_calc<DestinationCreator, FPack>
                          ( strf::detail::destination_tag()
                          , std::declval<DestinationCreator>()
                          , std::declval<FPack>() )))
    {
        auto& self = static_cast<destination_type_&>(*this);
        return { strf::detail::destination_tag{}
               , std::move(self.destination_creator_)
               , std::move(self.fpack_) };
    }

    constexpr STRF_HD strf::printer_with_given_size<DestinationCreator, FPack>
    reserve(std::size_t size) const &
        noexcept(noexcept(strf::printer_with_given_size<DestinationCreator, FPack>
                          ( strf::detail::destination_tag()
                          , std::size_t()
                          , std::declval<const DestinationCreator&>()
                          , std::declval<const FPack&>() )))
    {
        return { strf::detail::destination_tag{}
               , size
               , static_cast<const destination_type_*>(this)->destination_creator_
               , static_cast<const destination_type_*>(this)->fpack_ };
    }

    STRF_CONSTEXPR_IN_CXX14 STRF_HD strf::printer_with_given_size<DestinationCreator, FPack>
    reserve(std::size_t size) &&
        noexcept(noexcept(strf::printer_with_given_size<DestinationCreator, FPack>
                          ( strf::detail::destination_tag()
                          , std::size_t()
                          , std::declval<DestinationCreator>()
                          , std::declval<FPack>() )))
    {
        auto& self = static_cast<destination_type_&>(*this);
        return { strf::detail::destination_tag{}
               , size
               , std::move(self.destination_creator_)
               , std::move(self.fpack_) };
    }

    template <typename ... Args>
    finish_return_type_ STRF_HD operator()(const Args& ... args) const &
    {
        const auto& self = static_cast<const destination_type_&>(*this);
        PrePrinting pre;
        return self.write_
            ( pre
            , as_stringifier_cref_
              ( stringifier_<Args>
                ( strf::make_stringifier_input<char_type_>
                  ( pre, self.fpack_, args ) ) )... );
    }

#if defined(STRF_HAS_STD_STRING_VIEW)

    template <typename ... Args>
    finish_return_type_ STRF_HD tr
        ( const std::basic_string_view<char_type_>& str
        , const Args& ... args ) const &
    {
        return tr_write_(str.data(), str.size(), args...);
    }

#else

    template <typename ... Args>
    finish_return_type_ STRF_HD tr(const char_type_* str, const Args& ... args) const &
    {
        return tr_write_(str, strf::detail::str_length<char_type_>(str), args...);
    }

#endif

private:

    static inline STRF_HD const strf::stringifier<char_type_>&
    as_stringifier_cref_(const strf::stringifier<char_type_>& p) noexcept
    {
        return p;
    }
    static inline STRF_HD const strf::stringifier<char_type_>*
    as_stringifier_cptr_(const strf::stringifier<char_type_>& p) noexcept
    {
         return &p;
    }

    template < typename ... Args >
    finish_return_type_ STRF_HD tr_write_
        ( const char_type_* str
        , std::size_t str_len
        , const Args& ... args) const &
    {
        return tr_write_2_
            ( str, str + str_len
            , strf::detail::make_index_sequence<sizeof...(args)>()
            , args...);
    }

    template < std::size_t ... I, typename ... Args >
    finish_return_type_ STRF_HD tr_write_2_
        ( const char_type_* str
        , const char_type_* str_end
        , strf::detail::index_sequence<I...>
        , const Args& ... args) const &
    {
        constexpr std::size_t args_count = sizeof...(args);
        PrePrinting pre_arr[args_count ? args_count : 1];
        const auto& fpack = static_cast<const destination_type_&>(*this).fpack_;
        (void)fpack;
        return tr_write_3_
            ( str
            , str_end
            , pre_arr
            , { as_stringifier_cptr_
                ( stringifier_<Args>
                  ( strf::make_stringifier_input<char_type_>
                    ( pre_arr[I], fpack, args ) ) )... } );
    }

    template <typename ... Args>
    finish_return_type_ STRF_HD tr_write_3_
        ( const char_type_* str
        , const char_type_* str_end
        , PrePrinting* pre_arr
        , std::initializer_list<const strf::stringifier<char_type_>*> args ) const &
    {
        const auto& self = static_cast<const destination_type_&>(*this);

        using catenc = strf::charset_c<char_type_>;
        auto charset = strf::use_facet<catenc, void>(self.fpack_);

        using caterr = strf::tr_error_notifier_c;
        auto&& err_hdl = strf::use_facet<caterr, void>(self.fpack_);
        using err_hdl_type = strf::detail::remove_cvref_t<decltype(err_hdl)>;

        PrePrinting pre;
        strf::detail::tr_string_printer<decltype(charset), err_hdl_type>
            tr_printer(pre, pre_arr, args, str, str_end, charset, err_hdl);

        return self.write_(pre, tr_printer);
    }
};

}// namespace detail

template < typename DestinationCreator, typename FPack >
class printer_no_reserve
    : private strf::detail::printer_common
        < strf::printer_no_reserve
        , false
        , DestinationCreator
        , strf::no_preprinting
        , FPack >
{
    using common_ = strf::detail::printer_common
        < strf::printer_no_reserve
        , false
        , DestinationCreator
        , strf::no_preprinting
        , FPack >;

    template <template <typename, typename> class, bool, class, class, class>
    friend class strf::detail::printer_common;

    using preprinting_type_ = strf::no_preprinting;
    using finish_return_type_ = strf::detail::destination_finish_return_type<DestinationCreator, false>;

public:

    using char_type = typename DestinationCreator::char_type;

    template < typename ... Args
             , typename FP = FPack
             , strf::detail::enable_if_t
                 < std::is_constructible<DestinationCreator, Args...>::value
                && std::is_default_constructible<FP>::value
                 , int > = 0 >
    constexpr STRF_HD printer_no_reserve(Args&&... args)
        noexcept( noexcept(DestinationCreator(std::declval<Args>()...))
               && noexcept(FP()) )
        : destination_creator_(std::forward<Args>(args)...)
    {
    }

    template < typename T = DestinationCreator
             , strf::detail::enable_if_t
                 < std::is_copy_constructible<T>::value, int > = 0 >
    constexpr STRF_HD printer_no_reserve
        ( strf::detail::destination_tag
        , const DestinationCreator& oc
        , const FPack& fp )
        noexcept( noexcept(DestinationCreator(std::declval<const DestinationCreator&>()))
               && noexcept(FPack(std::declval<const FPack&>())) )
        : destination_creator_(oc)
        , fpack_(fp)
    {
    }

    constexpr STRF_HD printer_no_reserve
        ( strf::detail::destination_tag
        , DestinationCreator&& oc
        , FPack&& fp )
        noexcept( noexcept(DestinationCreator(std::declval<DestinationCreator>()))
               && noexcept(FPack(std::declval<FPack>())) )
        : destination_creator_(std::move(oc))
        , fpack_(std::move(fp))
    {
    }

    using common_::with;
    using common_::operator();
    using common_::tr;
    using common_::reserve_calc;
    using common_::reserve;

    STRF_CONSTEXPR_IN_CXX14 STRF_HD printer_no_reserve& no_reserve() & noexcept
    {
        return *this;
    }
    constexpr STRF_HD const printer_no_reserve& no_reserve() const & noexcept
    {
        return *this;
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD printer_no_reserve&& no_reserve() && noexcept
    {
        return std::move(*this);
    }
    constexpr STRF_HD const printer_no_reserve&& no_reserve() const && noexcept
    {
        return std::move(*this);
    }

private:

    template <class, class>
    friend class printer_no_reserve;

    template < typename OtherFPack
             , typename ... FPE
             , typename T = DestinationCreator
             , strf::detail::enable_if_t<std::is_copy_constructible<T>::value, int> = 0 >
    constexpr STRF_HD printer_no_reserve
        ( const printer_no_reserve<DestinationCreator, OtherFPack>& other
        , detail::destination_tag
        , FPE&& ... fpe )
        noexcept( noexcept(DestinationCreator(std::declval<const DestinationCreator&>()))
               && noexcept(FPack(std::declval<const OtherFPack&>(), std::declval<FPE>()...)) )
        : destination_creator_(other.destination_creator_)
        , fpack_(other.fpack_, std::forward<FPE>(fpe)...)
    {
    }

    template < typename OtherFPack, typename ... FPE >
    constexpr STRF_HD printer_no_reserve
        ( printer_no_reserve<DestinationCreator, OtherFPack>&& other
        , detail::destination_tag
        , FPE&& ... fpe )
        noexcept( noexcept(DestinationCreator(std::declval<DestinationCreator>()))
               && noexcept(FPack(std::declval<OtherFPack>(), std::declval<FPE>()...)) )
        : destination_creator_(std::move(other.destination_creator_))
        , fpack_(std::move(other.fpack_), std::forward<FPE>(fpe)...)
    {
    }

    template <typename ... Stringifiers>
    finish_return_type_ STRF_HD write_
        ( const preprinting_type_&
        , const Stringifiers& ... stringifiers) const
    {
        typename DestinationCreator::destination_type dest{destination_creator_.create()};
        strf::detail::write_args(dest, stringifiers...);
        return strf::detail::finish(strf::rank<2>(), dest);
    }

    DestinationCreator destination_creator_;
    FPack fpack_;
};

template < typename DestinationCreator, typename FPack >
class printer_with_given_size
    : public strf::detail::printer_common
        < strf::printer_with_given_size
        , true
        , DestinationCreator
        , strf::no_preprinting
        , FPack >
{
    using common_ = strf::detail::printer_common
        < strf::printer_with_given_size
        , true
        , DestinationCreator
        , strf::no_preprinting
        , FPack >;

    template < template <typename, typename> class, bool, class, class, class>
    friend class strf::detail::printer_common;

    using preprinting_type_ = strf::no_preprinting;
    using finish_return_type_ = strf::detail::destination_finish_return_type<DestinationCreator, true>;

public:

    using char_type = typename DestinationCreator::char_type;

    template < typename ... Args
             , typename FP = FPack
             , strf::detail::enable_if_t
                 < std::is_constructible<DestinationCreator, Args...>::value
                && std::is_default_constructible<FP>::value
                 , int > = 0 >
    constexpr STRF_HD printer_with_given_size(std::size_t size, Args&&... args)
        noexcept( noexcept(DestinationCreator(std::declval<Args>()...))
               && noexcept(FP()) )
        : size_(size)
        , destination_creator_(std::forward<Args>(args)...)
    {
    }

    template < typename T = DestinationCreator
             , strf::detail::enable_if_t<std::is_copy_constructible<T>::value, int> = 0 >
    constexpr STRF_HD printer_with_given_size
        ( strf::detail::destination_tag
        , std::size_t size
        , const DestinationCreator& oc
        , const FPack& fp )
        noexcept( noexcept(DestinationCreator(std::declval<const DestinationCreator&>()))
               && noexcept(FPack(std::declval<const FPack&>())) )
        : size_(size)
        , destination_creator_(oc)
        , fpack_(fp)
    {
    }

    constexpr STRF_HD printer_with_given_size
        ( strf::detail::destination_tag
        , std::size_t size
        , DestinationCreator&& oc
        , FPack&& fp )
        noexcept( noexcept(DestinationCreator(std::declval<DestinationCreator>()))
               && noexcept(FPack(std::declval<FPack>())) )
        : size_(size)
        , destination_creator_(std::move(oc))
        , fpack_(std::move(fp))
    {
    }

    using common_::with;
    using common_::operator();
    using common_::tr;
    using common_::reserve_calc;
    using common_::no_reserve;

    STRF_CONSTEXPR_IN_CXX14 STRF_HD printer_with_given_size& reserve(std::size_t size) & noexcept
    {
        size_ = size;
        return *this;
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD printer_with_given_size&& reserve(std::size_t size) && noexcept
    {
        size_ = size;
        return std::move(*this);
    }

private:

    template <class, class>
    friend class printer_with_given_size;

    template < typename OtherFPack
             , typename ... FPE
             , typename T = DestinationCreator
             , strf::detail::enable_if_t<std::is_copy_constructible<T>::value, int> = 0>
    constexpr STRF_HD printer_with_given_size
        ( const printer_with_given_size<DestinationCreator, OtherFPack>& other
        , detail::destination_tag
        , FPE&& ... fpe )
        noexcept( noexcept(DestinationCreator(std::declval<const DestinationCreator&>()))
               && noexcept(FPack(std::declval<const FPack&>(), std::declval<const FPE&>()...)) )
        : size_(other.size_)
        , destination_creator_(other.destination_creator_)
        , fpack_(other.fpack_, std::forward<FPE>(fpe)...)
    {
    }

    template < typename OtherFPack, typename ... FPE >
    constexpr STRF_HD printer_with_given_size
        ( printer_with_given_size<DestinationCreator, OtherFPack>&& other
        , detail::destination_tag
        , FPE&& ... fpe )
        noexcept( noexcept(DestinationCreator(std::declval<DestinationCreator>()))
               && noexcept(FPack(std::declval<FPack>(), std::declval<FPE>()...)) )
        : size_(other.size)
        , destination_creator_(std::move(other.destination_creator_))
        , fpack_(std::move(other.fpack_), std::forward<FPE>(fpe)...)
    {
    }

    template <typename ... Stringifiers>
    STRF_HD finish_return_type_ write_
        ( const preprinting_type_&
        , const Stringifiers& ... stringifiers) const
    {
        typename DestinationCreator::sized_destination_type dest{destination_creator_.create(size_)};
        strf::detail::write_args(dest, stringifiers...);
        return strf::detail::finish(strf::rank<2>(), dest);
    }

    std::size_t size_;
    DestinationCreator destination_creator_;
    FPack fpack_;
};

template < typename DestinationCreator, typename FPack >
class printer_with_size_calc
    : public strf::detail::printer_common
        < strf::printer_with_size_calc
        , true
        , DestinationCreator
        , strf::preprinting<strf::precalc_size::yes, strf::precalc_width::no>
        , FPack >
{
    using common_ = strf::detail::printer_common
        < strf::printer_with_size_calc
        , true
        , DestinationCreator
        , strf::preprinting<strf::precalc_size::yes, strf::precalc_width::no>
        , FPack >;

    template < template <typename, typename> class, bool, class, class, class>
    friend class strf::detail::printer_common;

    using preprinting_type_
        = strf::preprinting<strf::precalc_size::yes, strf::precalc_width::no>;
    using finish_return_type_ = strf::detail::destination_finish_return_type<DestinationCreator, true>;

public:

    using char_type = typename DestinationCreator::char_type;

    template < typename ... Args
             , typename FP = FPack
             , strf::detail::enable_if_t
                 < std::is_constructible<DestinationCreator, Args...>::value
                && std::is_default_constructible<FP>::value
                 , int > = 0 >
    constexpr STRF_HD printer_with_size_calc(Args&&... args)
        noexcept( noexcept(DestinationCreator(std::declval<Args>()...))
               && noexcept(FP()) )
        : destination_creator_(std::forward<Args>(args)...)
    {
    }

    template < typename T = DestinationCreator
             , strf::detail::enable_if_t
                 < std::is_copy_constructible<T>::value, int > = 0 >
    constexpr STRF_HD printer_with_size_calc
        ( strf::detail::destination_tag
        , const DestinationCreator& oc
        , const FPack& fp )
        noexcept( noexcept(DestinationCreator(std::declval<const DestinationCreator&>()))
               && noexcept(FPack(std::declval<const FPack&>())) )
        : destination_creator_(oc)
        , fpack_(fp)
    {
    }

    constexpr STRF_HD printer_with_size_calc
        ( strf::detail::destination_tag
        , DestinationCreator&& oc
        , FPack&& fp )
        noexcept( noexcept(DestinationCreator(std::declval<DestinationCreator>()))
               && noexcept(FPack(std::declval<FPack>())) )
        : destination_creator_(std::move(oc))
        , fpack_(std::move(fp))
    {
    }

    using common_::with;
    using common_::operator();
    using common_::tr;
    using common_::no_reserve;
    using common_::reserve;

    constexpr STRF_HD const printer_with_size_calc & reserve_calc() const& noexcept
    {
        return *this;
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD printer_with_size_calc & reserve_calc() & noexcept
    {
        return *this;
    }
    constexpr STRF_HD const printer_with_size_calc && reserve_calc() const&& noexcept
    {
        return std::move(*this);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD printer_with_size_calc && reserve_calc() && noexcept
    {
        return std::move(*this);
    }

private:

    template <typename, typename>
    friend class printer_with_size_calc;

    template < typename OtherFPack
             , typename ... FPE
             , typename T = DestinationCreator
             , strf::detail::enable_if_t<std::is_copy_constructible<T>::value, int> = 0 >
    STRF_HD printer_with_size_calc
        ( const printer_with_size_calc<DestinationCreator, OtherFPack>& other
        , detail::destination_tag
        , FPE&& ... fpe )
        noexcept( noexcept(DestinationCreator(std::declval<const DestinationCreator&>()))
               && noexcept(FPack(std::declval<const OtherFPack&>(), std::declval<FPE>()...)) )
        : destination_creator_(other.destination_creator_)
        , fpack_(other.fpack_, std::forward<FPE>(fpe)...)
    {
    }

    template < typename OtherFPack, typename ... FPE >
    STRF_HD printer_with_size_calc
        ( printer_with_size_calc<DestinationCreator, OtherFPack>&& other
        , detail::destination_tag
        , FPE&& ... fpe )
        noexcept( noexcept(DestinationCreator(std::declval<DestinationCreator>()))
               && noexcept(FPack(std::declval<OtherFPack>(), std::declval<FPE>()...)) )
        : destination_creator_(std::move(other.destination_creator_))
        , fpack_(std::move(other.fpack_), std::forward<FPE>(fpe)...)
    {
    }

    template <typename ... Stringifiers>
    finish_return_type_ STRF_HD write_
        ( const preprinting_type_& pre
        , const Stringifiers& ... stringifiers ) const
    {
        std::size_t size = pre.accumulated_size();
        typename DestinationCreator::sized_destination_type dest{destination_creator_.create(size)};
        strf::detail::write_args(dest, stringifiers...);
        return strf::detail::finish(strf::rank<2>(), dest);
    }

    DestinationCreator destination_creator_;
    FPack fpack_;
};

namespace detail {

template <typename CharT>
class destination_reference
{
public:

    using char_type = CharT;
    using destination_type = strf::destination<CharT>&;

    explicit STRF_HD destination_reference(strf::destination<CharT>& dest) noexcept
        : dest_(dest)
    {
    }

    STRF_HD strf::destination<CharT>& create() const noexcept
    {
        return dest_;
    }

private:
    strf::destination<CharT>& dest_;
};


} // namespace detail

template <typename CharT>
strf::printer_no_reserve<strf::detail::destination_reference<CharT>>
STRF_HD to(strf::destination<CharT>& dest) noexcept
{
    return strf::printer_no_reserve<strf::detail::destination_reference<CharT>>(dest);
}

namespace detail {

template <typename CharT>
class basic_cstr_destination_creator
{
public:

    using char_type = CharT;
    using finish_type = typename basic_cstr_destination<CharT>::result;
    using destination_type = basic_cstr_destination<CharT>;

    STRF_CONSTEXPR_IN_CXX14 STRF_HD
    basic_cstr_destination_creator(CharT* dest, CharT* dest_end) noexcept
        : dest_(dest)
        , dest_end_(dest_end)
    {
        STRF_ASSERT(dest < dest_end);
    }

    STRF_HD typename basic_cstr_destination<CharT>::range create() const noexcept
    {
        return typename basic_cstr_destination<CharT>::range{dest_, dest_end_};
    }

private:

    CharT* dest_;
    CharT* dest_end_;
};

template <typename CharT>
class array_destination_creator
{
public:

    using char_type = CharT;
    using finish_type = typename array_destination<CharT>::result;
    using destination_type = array_destination<CharT>;

    constexpr STRF_HD
    array_destination_creator(CharT* dest, CharT* dest_end) noexcept
        : dest_(dest)
        , dest_end_(dest_end)
    {
        STRF_ASSERT_IN_CONSTEXPR(dest < dest_end);
    }

    STRF_HD typename array_destination<CharT>::range create() const noexcept
    {
        return typename array_destination<CharT>::range{dest_, dest_end_};
    }

private:

    CharT* dest_;
    CharT* dest_end_;
};

}

#if defined(__cpp_char8_t)

template<std::size_t N>
inline STRF_HD auto to(char8_t (&dest)[N]) noexcept
{
    return strf::printer_no_reserve
        < strf::detail::basic_cstr_destination_creator<char8_t> >
        (dest, dest + N);
}

inline STRF_HD auto to(char8_t* dest, char8_t* end) noexcept
{
    return strf::printer_no_reserve
        < strf::detail::basic_cstr_destination_creator<char8_t> >
        (dest, end);
}

inline STRF_HD auto to(char8_t* dest, std::size_t count) noexcept
{
    return strf::printer_no_reserve
        < strf::detail::basic_cstr_destination_creator<char8_t> >
        (dest, dest + count);
}

#endif

template<std::size_t N>
inline STRF_HD auto to(char (&dest)[N]) noexcept
    -> strf::printer_no_reserve
        < strf::detail::basic_cstr_destination_creator<char> >
{
    return strf::printer_no_reserve
        < strf::detail::basic_cstr_destination_creator<char> >
        (dest, dest + N);
}

inline STRF_HD auto to(char* dest, char* end) noexcept
    -> strf::printer_no_reserve
        < strf::detail::basic_cstr_destination_creator<char> >
{
    return strf::printer_no_reserve
        < strf::detail::basic_cstr_destination_creator<char> >
        (dest, end);
}

inline STRF_HD auto to(char* dest, std::size_t count) noexcept
    -> strf::printer_no_reserve
        < strf::detail::basic_cstr_destination_creator<char> >
{
    return strf::printer_no_reserve
        < strf::detail::basic_cstr_destination_creator<char> >
        (dest, dest + count);
}

template<std::size_t N>
inline STRF_HD auto to(char16_t (&dest)[N]) noexcept
    -> strf::printer_no_reserve
        < strf::detail::basic_cstr_destination_creator<char16_t> >
{
    return strf::printer_no_reserve
        < strf::detail::basic_cstr_destination_creator<char16_t> >
        (dest, dest + N);
}

inline STRF_HD auto to(char16_t* dest, char16_t* end) noexcept
    -> strf::printer_no_reserve
        < strf::detail::basic_cstr_destination_creator<char16_t> >
{
    return strf::printer_no_reserve
        < strf::detail::basic_cstr_destination_creator<char16_t> >
        (dest, end);
}

inline STRF_HD auto to(char16_t* dest, std::size_t count) noexcept
    -> strf::printer_no_reserve
        < strf::detail::basic_cstr_destination_creator<char16_t> >
{
    return strf::printer_no_reserve
        < strf::detail::basic_cstr_destination_creator<char16_t> >
        (dest, dest + count);
}

template<std::size_t N>
inline STRF_HD auto to(char32_t (&dest)[N]) noexcept
    -> strf::printer_no_reserve
        < strf::detail::basic_cstr_destination_creator<char32_t> >
{
    return strf::printer_no_reserve
        < strf::detail::basic_cstr_destination_creator<char32_t> >
        (dest, dest + N);
}

inline STRF_HD auto to(char32_t* dest, char32_t* end) noexcept
    -> strf::printer_no_reserve
        < strf::detail::basic_cstr_destination_creator<char32_t> >
{
    return strf::printer_no_reserve
        < strf::detail::basic_cstr_destination_creator<char32_t> >
        (dest, end);
}

inline STRF_HD auto to(char32_t* dest, std::size_t count) noexcept
    -> strf::printer_no_reserve
        < strf::detail::basic_cstr_destination_creator<char32_t> >
{
    return strf::printer_no_reserve
        < strf::detail::basic_cstr_destination_creator<char32_t> >
        (dest, dest + count);
}

template<std::size_t N>
inline STRF_HD auto to(wchar_t (&dest)[N]) noexcept
    -> strf::printer_no_reserve
        < strf::detail::basic_cstr_destination_creator<wchar_t> >
{
    return strf::printer_no_reserve
        < strf::detail::basic_cstr_destination_creator<wchar_t> >
        (dest, dest + N);
}

inline STRF_HD auto to(wchar_t* dest, wchar_t* end) noexcept
    -> strf::printer_no_reserve
        < strf::detail::basic_cstr_destination_creator<wchar_t> >
{
    return strf::printer_no_reserve
        < strf::detail::basic_cstr_destination_creator<wchar_t> >
        (dest, end);
}

inline STRF_HD auto to(wchar_t* dest, std::size_t count) noexcept
    -> strf::printer_no_reserve
        < strf::detail::basic_cstr_destination_creator<wchar_t> >
{
    return strf::printer_no_reserve
        < strf::detail::basic_cstr_destination_creator<wchar_t> >
        (dest, dest + count);
}

template<typename CharT, std::size_t N>
inline STRF_HD auto to_range(CharT (&dest)[N]) noexcept
    -> strf::printer_no_reserve
        < strf::detail::array_destination_creator<CharT> >
{
    return strf::printer_no_reserve
        < strf::detail::array_destination_creator<CharT> >
        (dest, dest + N);
}

template<typename CharT>
inline STRF_HD auto to_range(CharT* dest, CharT* end) noexcept
    -> strf::printer_no_reserve
        < strf::detail::array_destination_creator<CharT> >
{
    return strf::printer_no_reserve
        < strf::detail::array_destination_creator<CharT> >
        (dest, end);
}

template<typename CharT>
inline STRF_HD auto to_range(CharT* dest, std::size_t count) noexcept
    -> strf::printer_no_reserve
        < strf::detail::array_destination_creator<CharT> >
{
    return strf::printer_no_reserve
        < strf::detail::array_destination_creator<CharT> >
        (dest, dest + count);
}

} // namespace strf

#endif // STRF_PRINTER_HPP
