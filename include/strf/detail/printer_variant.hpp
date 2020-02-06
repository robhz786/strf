#ifndef STRF_DETAIL_PRINTER_VARIANT_HPP
#define STRF_DETAIL_PRINTER_VARIANT_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/printer.hpp>
#include <type_traits>
#include <algorithm>

namespace strf {

namespace detail {

template <typename Printer0, typename ... Printers>
class printer_variant
{
    static constexpr std::size_t max_size_
    = std::max({sizeof(Printer0), sizeof(Printers)...});

public:

    static constexpr std::size_t char_size = Printer0::char_size;

    template <typename P, typename ... Args>
    printer_variant(strf::tag<P>, Args&& ... args)
    {
        static_assert( std::is_base_of<strf::printer<char_size>, P>::value
                     , "Invalid printer type" );
        static_assert(sizeof(P) <= max_size_, "Invalid printer type");
        new ((void*)&pool_) P{std::forward<Args>(args)...};
    }

#if defined(STRF_NO_CXX17_COPY_ELISION)

    printer_variant(printer_variant&&);

#endif // defined(STRF_NO_CXX17_COPY_ELISION)

    ~printer_variant()
    {
        get_ptr_()->~printer();
    }

    operator const strf::printer<char_size>& () const
    {
        return *get_ptr_();
    }

    const strf::printer<char_size>& get() const
    {
        return *get_ptr_();
    }

private:

    const strf::printer<char_size>* get_ptr_() const
    {
        return reinterpret_cast<const strf::printer<char_size>*>(&pool_);
    }

    using storage_type_ = typename std::aligned_storage_t
        < max_size_, alignof(strf::printer<char_size>)>;

    storage_type_ pool_;
};

} // namespace detail

} // namespace strf

#endif  // STRF_DETAIL_PRINTER_VARIANT_HPP

