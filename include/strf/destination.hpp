#ifndef STRF_DESTINATION_HPP
#define STRF_DESTINATION_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/tr_string.hpp>
#include <strf/facets_pack.hpp>

namespace strf {

template < typename OutbuffCreator
         , typename FPack = strf::facets_pack<> >
class destination_with_given_size;

template < typename OutbuffCreator
         , typename FPack = strf::facets_pack<> >
class destination_calc_size;

template < typename OutbuffCreator
         , typename FPack = strf::facets_pack<> >
class destination_no_reserve;

namespace detail {

struct destination_tag {};

template < template <typename, typename> class DestinationTmpl
         , class OutbuffCreator, class Preview, class FPack >
class destination_common
{
    using destination_type_ = DestinationTmpl<OutbuffCreator, FPack>;

    using char_type_ = typename OutbuffCreator::char_type;

    template <typename Arg>
    using printer_ = strf::printer_type<char_type_, Preview, FPack, Arg>;

public:

    template <typename ... FPE>
    STRF_NODISCARD constexpr STRF_HD auto with(FPE&& ... fpe) const &
    {
        static_assert( std::is_copy_constructible<OutbuffCreator>::value
                     , "OutbuffCreator must be copy constructible" );

        const auto& self = static_cast<const destination_type_&>(*this);

        using NewFPack = decltype( strf::pack( std::declval<const FPack&>()
                                             , std::forward<FPE>(fpe) ...) );

        return DestinationTmpl<OutbuffCreator, NewFPack>
        { self, detail::destination_tag{}, std::forward<FPE>(fpe) ...};
    }

    template <typename ... FPE>
    STRF_NODISCARD constexpr STRF_HD auto with(FPE&& ... fpe) &&
    {
        static_assert( std::is_move_constructible<OutbuffCreator>::value
                     , "OutbuffCreator must be move constructible" );

        auto& self = static_cast<const destination_type_&>(*this);

        using NewFPack = decltype( strf::pack( std::declval<FPack>()
                                             , std::forward<FPE>(fpe) ...) );

        return DestinationTmpl<OutbuffCreator, NewFPack>
        { std::move(self), detail::destination_tag{}, std::forward<FPE>(fpe) ...};
    }

    constexpr STRF_HD strf::destination_no_reserve<OutbuffCreator, FPack>
    no_reserve() const &
    {
        const auto& self = static_cast<const destination_type_&>(*this);
        return { strf::detail::destination_tag{}
               , self.outbuff_creator_
               , self.fpack_ };
    }

    constexpr STRF_HD strf::destination_no_reserve<OutbuffCreator, FPack>
    no_reserve() &&
    {
        auto& self = static_cast<destination_type_&>(*this);
        return { strf::detail::destination_tag{}
               , std::move(self.outbuff_creator_)
               , std::move(self.fpack_) };
    }

    constexpr STRF_HD strf::destination_calc_size<OutbuffCreator, FPack>
    reserve_calc() const &
    {
        const auto& self = static_cast<const destination_type_&>(*this);
        return { strf::detail::destination_tag{}
               , self.outbuff_creator_
               , self.fpack_ };
    }

    strf::destination_calc_size<OutbuffCreator, FPack>
    STRF_HD reserve_calc() &&
    {
        auto& self = static_cast<destination_type_&>(*this);
        return { strf::detail::destination_tag{}
               , std::move(self.outbuff_creator_)
               , std::move(self.fpack_) };
    }

    constexpr STRF_HD strf::destination_with_given_size<OutbuffCreator, FPack>
    reserve(std::size_t size) const &
    {
        const auto& self = static_cast<const destination_type_&>(*this);
        return { strf::detail::destination_tag{}
               , size
               , self.outbuff_creator_
               , self.fpack_ };
    }

    constexpr STRF_HD strf::destination_with_given_size<OutbuffCreator, FPack>
    reserve(std::size_t size) &&
    {
        auto& self = static_cast<destination_type_&>(*this);
        return { strf::detail::destination_tag{}
               , size
               , std::move(self.outbuff_creator_)
               , std::move(self.fpack_) };
    }

    template <typename ... Args>
    decltype(auto) STRF_HD operator()(const Args& ... args) const &
    {
        const auto& self = static_cast<const destination_type_&>(*this);
        Preview preview;
        return self.write_
            ( preview
            , as_printer_cref_
              ( printer_<Args>
                ( strf::make_printer_input<char_type_>
                  ( preview, self.fpack_, args ) ) )... );
    }

#if defined(STRF_HAS_STD_STRING_VIEW)

    template <typename ... Args>
    decltype(auto) STRF_HD tr
        ( const std::basic_string_view<char_type_>& str
        , const Args& ... args ) const &
    {
        return tr_write_(str.data(), str.size(), args...);
    }

#else

    template <typename ... Args>
    decltype(auto) STRF_HD tr(const char_type_* str, const Args& ... args) const &
    {
        return tr_write_(str, strf::detail::str_length<char_type_>(str), args...);
    }

#endif

private:

    static inline const strf::printer<char_type_>&
    STRF_HD as_printer_cref_(const strf::printer<char_type_>& p)
    {
        return p;
    }
    static inline const strf::printer<char_type_>*
    STRF_HD as_printer_cptr_(const strf::printer<char_type_>& p)
    {
         return &p;
    }

    template < typename ... Args >
    decltype(auto) STRF_HD tr_write_
        ( const char_type_* str
        , std::size_t str_len
        , const Args& ... args) const &
    {
        return tr_write_2_
            (str, str + str_len, std::make_index_sequence<sizeof...(args)>(), args...);
    }

    template < std::size_t ... I, typename ... Args >
    decltype(auto) STRF_HD tr_write_2_
        ( const char_type_* str
        , const char_type_* str_end
        , std::index_sequence<I...>
        , const Args& ... args) const &
    {
        constexpr std::size_t args_count = sizeof...(args);
        Preview preview_arr[args_count ? args_count : 1];
        const auto& fpack = static_cast<const destination_type_&>(*this).fpack_;
        (void)fpack;
        return tr_write_3_
            ( str
            , str_end
            , preview_arr
            , { as_printer_cptr_
                ( printer_<Args>
                  ( strf::make_printer_input<char_type_>
                    ( preview_arr[I], fpack, args ) ) )... } );
    }

    template <typename ... Args>
    decltype(auto) STRF_HD tr_write_3_
        ( const char_type_* str
        , const char_type_* str_end
        , Preview* preview_arr
        , std::initializer_list<const strf::printer<char_type_>*> args ) const &
    {
        const auto& self = static_cast<const destination_type_&>(*this);

        using catenc = strf::charset_c<char_type_>;
        auto charset = strf::get_facet<catenc, void>(self.fpack_);

        using caterr = strf::tr_error_notifier_c;
        decltype(auto) err_hdl = strf::get_facet<caterr, void>(self.fpack_);
        using err_hdl_type = std::remove_cv_t<std::remove_reference_t<decltype(err_hdl)>>;

        Preview preview;
        strf::detail::tr_string_printer<decltype(charset), err_hdl_type>
            tr_printer(preview, preview_arr, args, str, str_end, charset, err_hdl);

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

template < typename OutbuffCreator, typename FPack >
class destination_no_reserve
    : private strf::detail::destination_common
        < strf::destination_no_reserve
        , OutbuffCreator
        , strf::no_print_preview
        , FPack >
{
    using common_ = strf::detail::destination_common
        < strf::destination_no_reserve
        , OutbuffCreator
        , strf::no_print_preview
        , FPack >;

    template <template <typename, typename> class, class, class, class>
    friend class strf::detail::destination_common;

    using preview_type_ = strf::no_print_preview;

public:

    using char_type = typename OutbuffCreator::char_type;

    template < typename ... Args
             , std::enable_if_t
                 < std::is_constructible<OutbuffCreator, Args...>::value
                 , int > = 0 >
    constexpr STRF_HD destination_no_reserve(Args&&... args)
        : outbuff_creator_(std::forward<Args>(args)...)
    {
    }

    template < typename T = OutbuffCreator
             , std::enable_if_t
                 < std::is_copy_constructible<T>::value, int > = 0 >
    constexpr STRF_HD destination_no_reserve( strf::detail::destination_tag
                                            , const OutbuffCreator& oc
                                            , const FPack& fp )
        : outbuff_creator_(oc)
        , fpack_(fp)
    {
    }

    constexpr STRF_HD destination_no_reserve( strf::detail::destination_tag
                                            , OutbuffCreator&& oc
                                            , FPack&& fp )
        : outbuff_creator_(std::move(oc))
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
             , typename T = OutbuffCreator
             , std::enable_if_t<std::is_copy_constructible<T>::value, int> = 0 >
    constexpr STRF_HD destination_no_reserve
        ( const destination_no_reserve<OutbuffCreator, OtherFPack>& other
        , detail::destination_tag
        , FPE&& ... fpe )
        : outbuff_creator_(other.outbuff_creator_)
        , fpack_(other.fpack_, std::forward<FPE>(fpe)...)
    {
    }

    template < typename OtherFPack, typename ... FPE >
    constexpr STRF_HD destination_no_reserve
        ( destination_no_reserve<OutbuffCreator, OtherFPack>&& other
        , detail::destination_tag
        , FPE&& ... fpe )
        : outbuff_creator_(std::move(other.outbuff_creator_))
        , fpack_(std::move(other.fpack_), std::forward<FPE>(fpe)...)
    {
    }

    template <typename ... Printers>
    decltype(auto) STRF_HD write_
        ( const preview_type_&
        , const Printers& ... printers) const
    {
        typename OutbuffCreator::outbuff_type ob{outbuff_creator_.create()};
        strf::detail::write_args(ob, printers...);
        return strf::detail::finish(strf::rank<2>(), ob);
    }

    OutbuffCreator outbuff_creator_;
    FPack fpack_;
};

template < typename OutbuffCreator, typename FPack >
class destination_with_given_size
    : public strf::detail::destination_common
        < strf::destination_with_given_size
        , OutbuffCreator
        , strf::no_print_preview
        , FPack >
{
    using common_ = strf::detail::destination_common
        < strf::destination_with_given_size
        , OutbuffCreator
        , strf::no_print_preview
        , FPack >;

    template < template <typename, typename> class, class,class, class>
    friend class strf::detail::destination_common;

    using preview_type_ = strf::no_print_preview;

public:

    using char_type = typename OutbuffCreator::char_type;

    template < typename ... Args
             , std::enable_if_t
                 < std::is_constructible<OutbuffCreator, Args...>::value
                 , int > = 0 >
    constexpr STRF_HD destination_with_given_size(std::size_t size, Args&&... args)
        : size_(size)
        , outbuff_creator_(std::forward<Args>(args)...)
    {
    }

    template < typename T = OutbuffCreator
             , std::enable_if_t<std::is_copy_constructible<T>::value, int> = 0 >
    constexpr STRF_HD destination_with_given_size( strf::detail::destination_tag
                                                 , std::size_t size
                                                 , const OutbuffCreator& oc
                                                 , const FPack& fp )
        : size_(size)
        , outbuff_creator_(oc)
        , fpack_(fp)
    {
    }

    constexpr STRF_HD destination_with_given_size( strf::detail::destination_tag
                                                 , std::size_t size
                                                 , OutbuffCreator&& oc
                                                 , FPack&& fp )
        : size_(size)
        , outbuff_creator_(std::move(oc))
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
             , typename T = OutbuffCreator
             , std::enable_if_t<std::is_copy_constructible<T>::value, int> = 0>
    constexpr STRF_HD destination_with_given_size
        ( const destination_with_given_size<OutbuffCreator, OtherFPack>& other
        , detail::destination_tag
        , FPE&& ... fpe )
        : size_(other.size_)
        , outbuff_creator_(other.outbuff_creator_)
        , fpack_(other.fpack_, std::forward<FPE>(fpe)...)
    {
    }

    template < typename OtherFPack, typename ... FPE >
    constexpr STRF_HD destination_with_given_size
        ( destination_with_given_size<OutbuffCreator, OtherFPack>&& other
        , detail::destination_tag
        , FPE&& ... fpe )
        : size_(other.size)
        , outbuff_creator_(std::move(other.outbuff_creator_))
        , fpack_(std::move(other.fpack_), std::forward<FPE>(fpe)...)
    {
    }

    template <typename ... Printers>
    decltype(auto) STRF_HD write_
        ( const preview_type_&
        , const Printers& ... printers) const
    {
        typename OutbuffCreator::sized_outbuff_type ob{outbuff_creator_.create(size_)};
        strf::detail::write_args(ob, printers...);
        return strf::detail::finish(strf::rank<2>(), ob);
    }

    std::size_t size_;
    OutbuffCreator outbuff_creator_;
    FPack fpack_;
};

template < typename OutbuffCreator, typename FPack >
class destination_calc_size
    : public strf::detail::destination_common
        < strf::destination_calc_size
        , OutbuffCreator
        , strf::print_preview<strf::preview_size::yes, strf::preview_width::no>
        , FPack >
{
    using common_ = strf::detail::destination_common
        < strf::destination_calc_size
        , OutbuffCreator
        , strf::print_preview<strf::preview_size::yes, strf::preview_width::no>
        , FPack >;

    template < template <typename, typename> class, class, class, class>
    friend class strf::detail::destination_common;

    using preview_type_
        = strf::print_preview<strf::preview_size::yes, strf::preview_width::no>;

public:

    using char_type = typename OutbuffCreator::char_type;

    template < typename ... Args
             , std::enable_if_t
                 < std::is_constructible<OutbuffCreator, Args...>::value
                 , int > = 0 >
    constexpr STRF_HD destination_calc_size(Args&&... args)
        : outbuff_creator_(std::forward<Args>(args)...)
    {
    }

    template < typename T = OutbuffCreator
             , std::enable_if_t
                 < std::is_copy_constructible<T>::value, int > = 0 >
    constexpr STRF_HD destination_calc_size( strf::detail::destination_tag
                                           , const OutbuffCreator& oc
                                           , const FPack& fp )
        : outbuff_creator_(oc)
        , fpack_(fp)
    {
    }

    constexpr STRF_HD destination_calc_size( strf::detail::destination_tag
                                           , OutbuffCreator&& oc
                                           , FPack&& fp )
        : outbuff_creator_(std::move(oc))
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
             , typename T = OutbuffCreator
             , std::enable_if_t<std::is_copy_constructible<T>::value, int> = 0 >
    STRF_HD destination_calc_size
        ( const destination_calc_size<OutbuffCreator, OtherFPack>& other
        , detail::destination_tag
        , FPE&& ... fpe )
        : outbuff_creator_(other.outbuff_creator_)
        , fpack_(other.fpack_, std::forward<FPE>(fpe)...)
    {
    }

    template < typename OtherFPack, typename ... FPE >
    STRF_HD destination_calc_size
        ( destination_calc_size<OutbuffCreator, OtherFPack>&& other
        , detail::destination_tag
        , FPE&& ... fpe )
        : outbuff_creator_(std::move(other.outbuff_creator_))
        , fpack_(std::move(other.fpack_), std::forward<FPE>(fpe)...)
    {
    }

    template <typename ... Printers>
    decltype(auto) STRF_HD write_
        ( const preview_type_& preview
        , const Printers& ... printers ) const
    {
        std::size_t size = preview.accumulated_size();
        typename OutbuffCreator::sized_outbuff_type ob{outbuff_creator_.create(size)};
        strf::detail::write_args(ob, printers...);
        return strf::detail::finish(strf::rank<2>(), ob);
    }

    OutbuffCreator outbuff_creator_;
    FPack fpack_;
};

namespace detail {

template <typename CharT>
class outbuff_reference
{
public:

    using char_type = CharT;
    using outbuff_type = strf::basic_outbuff<CharT>&;

    explicit STRF_HD outbuff_reference(strf::basic_outbuff<CharT>& ob) noexcept
        : ob_(ob)
    {
    }

    STRF_HD strf::basic_outbuff<CharT>& create() const
    {
        return ob_;
    }

private:
    strf::basic_outbuff<CharT>& ob_;
};


} // namespace detail

template <typename CharT>
auto STRF_HD to(strf::basic_outbuff<CharT>& ob)
{
    return strf::destination_no_reserve<strf::detail::outbuff_reference<CharT>>(ob);
}

namespace detail {

template <typename CharT>
class basic_cstr_writer_creator
{
public:

    using char_type = CharT;
    using finish_type = typename basic_cstr_writer<CharT>::result;
    using outbuff_type = basic_cstr_writer<CharT>;

    constexpr STRF_HD
    basic_cstr_writer_creator(CharT* dest, CharT* dest_end) noexcept
        : dest_(dest)
        , dest_end_(dest_end)
    {
        STRF_ASSERT(dest < dest_end);
    }

    STRF_HD typename basic_cstr_writer<CharT>::range create() const noexcept
    {
        return typename basic_cstr_writer<CharT>::range{dest_, dest_end_};
    }

private:

    CharT* dest_;
    CharT* dest_end_;
};

template <typename CharT>
class basic_char_array_writer_creator
{
public:

    using char_type = CharT;
    using finish_type = typename basic_char_array_writer<CharT>::result;
    using outbuff_type = basic_char_array_writer<CharT>;

    constexpr STRF_HD
    basic_char_array_writer_creator(CharT* dest, CharT* dest_end) noexcept
        : dest_(dest)
        , dest_end_(dest_end)
    {
        STRF_ASSERT(dest < dest_end);
    }

    STRF_HD typename basic_char_array_writer<CharT>::range create() const noexcept
    {
        return typename basic_char_array_writer<CharT>::range{dest_, dest_end_};
    }

private:

    CharT* dest_;
    CharT* dest_end_;
};

}

#if defined(__cpp_char8_t)

template<std::size_t N>
inline STRF_HD auto to(char8_t (&dest)[N])
{
    return strf::destination_no_reserve
        < strf::detail::basic_cstr_writer_creator<char8_t> >
        (dest, dest + N);
}

inline STRF_HD auto to(char8_t* dest, char8_t* end)
{
    return strf::destination_no_reserve
        < strf::detail::basic_cstr_writer_creator<char8_t> >
        (dest, end);
}

inline STRF_HD auto to(char8_t* dest, std::size_t count)
{
    return strf::destination_no_reserve
        < strf::detail::basic_cstr_writer_creator<char8_t> >
        (dest, dest + count);
}

#endif

template<std::size_t N>
inline STRF_HD auto to(char (&dest)[N])
{
    return strf::destination_no_reserve
        < strf::detail::basic_cstr_writer_creator<char> >
        (dest, dest + N);
}

inline STRF_HD auto to(char* dest, char* end)
{
    return strf::destination_no_reserve
        < strf::detail::basic_cstr_writer_creator<char> >
        (dest, end);
}

inline STRF_HD auto to(char* dest, std::size_t count)
{
    return strf::destination_no_reserve
        < strf::detail::basic_cstr_writer_creator<char> >
        (dest, dest + count);
}

template<std::size_t N>
inline STRF_HD auto to(char16_t (&dest)[N])
{
    return strf::destination_no_reserve
        < strf::detail::basic_cstr_writer_creator<char16_t> >
        (dest, dest + N);
}

inline STRF_HD auto to(char16_t* dest, char16_t* end)
{
    return strf::destination_no_reserve
        < strf::detail::basic_cstr_writer_creator<char16_t> >
        (dest, end);
}

inline STRF_HD auto to(char16_t* dest, std::size_t count)
{
    return strf::destination_no_reserve
        < strf::detail::basic_cstr_writer_creator<char16_t> >
        (dest, dest + count);
}

template<std::size_t N>
inline STRF_HD auto to(char32_t (&dest)[N])
{
    return strf::destination_no_reserve
        < strf::detail::basic_cstr_writer_creator<char32_t> >
        (dest, dest + N);
}

inline STRF_HD auto to(char32_t* dest, char32_t* end)
{
    return strf::destination_no_reserve
        < strf::detail::basic_cstr_writer_creator<char32_t> >
        (dest, end);
}

inline STRF_HD auto to(char32_t* dest, std::size_t count)
{
    return strf::destination_no_reserve
        < strf::detail::basic_cstr_writer_creator<char32_t> >
        (dest, dest + count);
}

template<std::size_t N>
inline STRF_HD auto to(wchar_t (&dest)[N])
{
    return strf::destination_no_reserve
        < strf::detail::basic_cstr_writer_creator<wchar_t> >
        (dest, dest + N);
}

inline STRF_HD auto to(wchar_t* dest, wchar_t* end)
{
    return strf::destination_no_reserve
        < strf::detail::basic_cstr_writer_creator<wchar_t> >
        (dest, end);
}

inline STRF_HD auto to(wchar_t* dest, std::size_t count)
{
    return strf::destination_no_reserve
        < strf::detail::basic_cstr_writer_creator<wchar_t> >
        (dest, dest + count);
}

template<typename CharT, std::size_t N>
inline STRF_HD auto to_range(CharT (&dest)[N])
{
    return strf::destination_no_reserve
        < strf::detail::basic_char_array_writer_creator<CharT> >
        (dest, dest + N);
}

template<typename CharT>
inline STRF_HD auto to_range(CharT* dest, CharT* end)
{
    return strf::destination_no_reserve
        < strf::detail::basic_char_array_writer_creator<CharT> >
        (dest, end);
}

template<typename CharT>
inline STRF_HD auto to_range(CharT* dest, std::size_t count)
{
    return strf::destination_no_reserve
        < strf::detail::basic_char_array_writer_creator<CharT> >
        (dest, dest + count);
}


} // namespace strf

#endif  // STRF_DESTINATION_HPP
