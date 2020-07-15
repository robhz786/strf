#ifndef STRF_PRINTER_HPP
#define STRF_PRINTER_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/outbuff.hpp>
#include <strf/width_t.hpp>
#include <strf/facets_pack.hpp>

namespace strf {

template <typename CharT>
class printer
{
public:

    using char_type = CharT;

    STRF_HD virtual ~printer()
    {
    }

    STRF_HD virtual void print_to(strf::basic_outbuff<CharT>& ob) const = 0;
};

struct string_input_tag_base
{
};

template <typename CharIn>
struct string_input_tag: string_input_tag_base
{
};

template <typename CharT>
struct is_string_of
{
    template <typename T>
    using fn = std::is_base_of<string_input_tag<CharT>, T>;
};

template <typename T>
using is_string = std::is_base_of<string_input_tag_base, T>;

template <typename CharIn>
struct tr_string_input_tag: strf::string_input_tag<CharIn>
{
};

template <typename CharIn>
struct range_separator_input_tag: strf::string_input_tag<CharIn>
{
};

template <typename CharIn>
struct is_tr_string_of
{
    template <typename T>
    using fn = std::is_same<strf::tr_string_input_tag<CharIn>, T>;
};

template <typename T>
struct is_tr_string: std::false_type
{
};

template <typename CharIn>
struct is_tr_string<strf::is_tr_string_of<CharIn>> : std::true_type
{
};

template <bool Active>
class width_preview;

template <>
class width_preview<true>
{
public:

    explicit constexpr STRF_HD width_preview(strf::width_t initial_width) noexcept
        : width_(initial_width)
    {}

    STRF_HD width_preview(const width_preview&) = delete;

    constexpr STRF_HD void subtract_width(strf::width_t w) noexcept
    {
        width_ -= w;
    }

    constexpr STRF_HD void checked_subtract_width(strf::width_t w) noexcept
    {
        if (w < width_) {
            width_ -= w;
        } else {
            width_ = 0;
        }
    }

    constexpr STRF_HD void checked_subtract_width(std::ptrdiff_t w) noexcept
    {
        if (w < width_.ceil()) {
            width_ -= static_cast<std::int16_t>(w);
        } else {
            width_ = 0;
        }
    }

    constexpr STRF_HD void clear_remaining_width() noexcept
    {
        width_ = 0;
    }

    constexpr STRF_HD strf::width_t remaining_width() const noexcept
    {
        return width_;
    }

private:

    strf::width_t width_;
};

template <>
class width_preview<false>
{
public:

    constexpr STRF_HD width_preview() noexcept
    {
    }

    constexpr STRF_HD void subtract_width(strf::width_t) noexcept
    {
    }

    constexpr STRF_HD void checked_subtract_width(strf::width_t) noexcept
    {
    }

    constexpr STRF_HD void checked_subtract_width(std::ptrdiff_t) noexcept
    {
    }

    constexpr STRF_HD void clear_remaining_width() noexcept
    {
    }

    constexpr STRF_HD strf::width_t remaining_width() const noexcept
    {
        return 0;
    }
};

template <bool Active>
class size_preview;

template <>
class size_preview<true>
{
public:
    explicit constexpr STRF_HD size_preview(std::size_t initial_size = 0) noexcept
        : size_(initial_size)
    {
    }

    STRF_HD size_preview(const size_preview&) = delete;

    constexpr STRF_HD void add_size(std::size_t s) noexcept
    {
        size_ += s;
    }

    constexpr STRF_HD std::size_t get_size() const noexcept
    {
        return size_;
    }

private:

    std::size_t size_;
};

template <>
class size_preview<false>
{
public:

    constexpr STRF_HD size_preview() noexcept
    {
    }

    constexpr STRF_HD void add_size(std::size_t) noexcept
    {
    }

    constexpr STRF_HD std::size_t get_size() const noexcept
    {
        return 0;
    }
};

enum class preview_width: bool { no = false, yes = true };
enum class preview_size : bool { no = false, yes = true };

template <strf::preview_size SizeRequired, strf::preview_width WidthRequired>
class print_preview
    : public strf::size_preview<static_cast<bool>(SizeRequired)>
    , public strf::width_preview<static_cast<bool>(WidthRequired)>
{
public:

    static constexpr bool size_required = static_cast<bool>(SizeRequired);
    static constexpr bool width_required = static_cast<bool>(WidthRequired);
    static constexpr bool nothing_required = ! size_required && ! width_required;

    template <strf::preview_width W = WidthRequired>
    STRF_HD constexpr explicit print_preview
        ( std::enable_if_t<static_cast<bool>(W), strf::width_t> initial_width ) noexcept
        : strf::width_preview<true>{initial_width}
    {
    }

    constexpr STRF_HD print_preview() noexcept
    {
    }
};

using no_print_preview = strf::print_preview<strf::preview_size::no, strf::preview_width::no>;

namespace detail {

#if defined(__cpp_fold_expressions)

template <typename CharT, typename ... Printers>
inline STRF_HD void write_args( strf::basic_outbuff<CharT>& ob
                              , const Printers& ... printers )
{
    (... , printers.print_to(ob));
}

#else // defined(__cpp_fold_expressions)

template <typename CharT>
inline STRF_HD void write_args(strf::basic_outbuff<CharT>&)
{
}

template <typename CharT, typename Printer, typename ... Printers>
inline STRF_HD void write_args
    ( strf::basic_outbuff<CharT>& ob
    , const Printer& printer
    , const Printers& ... printers )
{
    printer.print_to(ob);
    if (ob.good()) {
        write_args<CharT>(ob, printers ...);
    }
}

#endif // defined(__cpp_fold_expressions)

} // namespace detail

template <typename CharT>
struct printer_input_tag
{
private:
    static const printer_input_tag<CharT>& tag_();

public:

    template <typename Arg, typename FPack, typename Preview>
    constexpr STRF_HD auto operator()(Arg&& arg, const FPack& fp, Preview& preview) const
        noexcept(noexcept(strf::detail::tag_invoke(tag_(), arg, fp, preview)))
        -> decltype(strf::detail::tag_invoke(tag_(), arg, fp, preview))
    {
        return strf::detail::tag_invoke(*this, arg, fp, preview);
    }
};


struct printing_c;

struct default_printing_facet
{
#if defined(__cpp_inline_variable)

    template <typename CharT>
    constexpr static printer_input_tag<CharT> make_printer_input = {};

#else

private:

    template <typename CharT>
    using tag_ = printer_input_tag<CharT>;

public:

    template <typename CharT, typename Arg, typename FPack, typename Preview>
    constexpr STRF_HD auto make_printer_input(Arg&& arg, const FPack& fp, Preview& preview) const
        noexcept(noexcept(strf::detail::tag_invoke(tag_<CharT>(), arg, fp, preview)))
        -> decltype(strf::detail::tag_invoke(tag_<CharT>(), arg, fp, preview))
    {
        return strf::detail::tag_invoke(tag_<CharT>(), arg, fp, preview);
    }

#endif
};

struct printing_c
{
    static constexpr STRF_HD strf::default_printing_facet get_default() noexcept
    {
        return {};
    }
};

#if defined (STRF_NO_GLOBAL_CONSTEXPR_VARIABLE)

template <typename CharT, typename Arg, typename FPack, typename Preview>
constexpr STRF_HD decltype(auto) make_default_printer_input
    (Arg&& arg, const FPack& fp, Preview& preview)
{
    strf::printer_input_tag<CharT> tag;
    return tag(arg, fp, preview);
}


template <typename CharT, typename Arg, typename FPack, typename Preview>
constexpr STRF_HD decltype(auto) make_printer_input
    (Arg&& arg, const FPack& fp, Preview& preview)
{
    return strf::get_facet<strf::printing_c, Arg>(fp)
        .template make_printer_input<CharT>(arg, fp, preview);
}

#else

namespace detail {

template <typename CharT>
struct make_printer_input_impl
{
    template <typename Arg, typename FPack, typename Preview>
    constexpr STRF_HD decltype(auto) operator()
        (const Arg& arg, const FPack& fp, Preview& preview) const
        noexcept(noexcept(strf::get_facet<strf::printing_c, Arg>(fp)
                          .template make_printer_input<CharT>(arg, fp, preview)))
    {
        return strf::get_facet<strf::printing_c, Arg>(fp)
            .template make_printer_input<CharT>(arg, fp, preview);
    }
};

} // namespace detail

template <typename CharT>
constexpr strf::printer_input_tag<CharT> make_default_printer_input = {};

template <typename CharT>
constexpr strf::detail::make_printer_input_impl<CharT> make_printer_input = {};

#endif // defined (STRF_NO_GLOBAL_CONSTEXPR_VARIABLE)

namespace detail {

template <typename CharT, typename FPack, typename Preview, typename Arg>
struct printer_impl_helper
{
    static const FPack& fp();
    static Preview& preview();
    static const Arg& arg();

    using default_printer_input = decltype
        ( strf::make_default_printer_input<CharT>(arg(), fp(), preview()) );

    using printer_input = decltype
        ( strf::make_printer_input<CharT>(arg(), fp(), preview()) );

    using default_printer = typename default_printer_input::printer_type;
    using printer = typename printer_input::printer_type;
};

} // namespace detail

template <typename CharT, typename FPack, typename Preview, typename Arg>
using default_printer_impl = typename strf::detail::printer_impl_helper
    < CharT, FPack, Preview, Arg >
    ::default_printer;

template <typename CharT, typename FPack, typename Preview, typename Arg>
using printer_impl = typename strf::detail::printer_impl_helper
    < CharT, FPack, Preview, Arg >
    ::printer;

template < typename CharT, typename FPack, typename Preview
         , typename Printer, typename Arg >
struct usual_printer_input
{
    using fpack_type = FPack;
    using preview_type = Preview;
    using printer_type = Printer;

    FPack fp;
    Preview& preview;
    Arg arg;
};

} // namespace strf

#endif // STRF_PRINTER_HPP
