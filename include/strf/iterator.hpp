#ifndef STRF_ITERATOR_HPP
#define STRF_ITERATOR_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/destination.hpp>
#include <iterator> // std::output_iterator_tag;

namespace strf {

template <typename T>
class output_buffer_iterator {
public:
    using iterator_category = std::output_iterator_tag;
    using value_type = void;
    using difference_type = std::ptrdiff_t;
    using pointer = void;
    using reference = void;

    constexpr STRF_HD output_buffer_iterator() = delete;
    constexpr STRF_HD explicit output_buffer_iterator(strf::output_buffer<T, 0>& d) noexcept
        : dst_(&d)
    {
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD output_buffer_iterator& operator++() noexcept {
        return *this;
    }
    // NOLINTNEXTLINE(cert-dcl21-cpp)
    STRF_CONSTEXPR_IN_CXX14 STRF_HD output_buffer_iterator& operator++(int) noexcept {
        return *this;
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD output_buffer_iterator& operator*() noexcept {
        return *this;
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD output_buffer_iterator& operator=(T value) {
        strf::put(*dst_, value);
        return *this;
    }
    constexpr STRF_HD bool failed() const noexcept {
        return !dst_->good();
    }

private:
    strf::output_buffer<T, 0>* dst_;
};

template <typename T>
constexpr STRF_HD output_buffer_iterator<T> make_iterator(strf::output_buffer<T, 0>& dst) noexcept {
    return output_buffer_iterator<T>{dst};
}

} // namespace strf

#endif  // STRF_ITERATOR_HPP

