//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define  _CRT_SECURE_NO_WARNINGS

#include "test_utils.hpp"

class reservation_tester : public strf::print_dest<char>
{
    constexpr static std::size_t buff_size_ = strf::min_space_after_recycle<char>();
    char buff_[buff_size_];

public:

    STRF_HD reservation_tester(strf::tag<void>)
        : strf::print_dest<char>{ buff_, buff_ + buff_size_ }
        , buff_{0}
    {
    }

    STRF_HD reservation_tester(std::size_t size)
        : strf::print_dest<char>{ buff_, buff_ + buff_size_ }
        , buff_{0}
        , reserved_size_{size}
    {
    }

    reservation_tester(const reservation_tester&) = delete;
    reservation_tester(reservation_tester&&) = delete;

    void STRF_HD recycle() override
    {
        this->set_buffer_ptr(buff_);
    }

    std::size_t STRF_HD finish()
    {
        return reserved_size_;
    }

private:

    std::size_t reserved_size_ = 0;
};


class reservation_tester_creator
{
public:

    using char_type = char;
    using destination_type = reservation_tester;
    using sized_destination_type = reservation_tester;

    strf::tag<void> STRF_HD create() const
    {
        return strf::tag<void>{};
    }
    std::size_t STRF_HD create(std::size_t size) const
    {
        return size;
    }
};

constexpr auto STRF_HD reservation_test()
    -> strf::destination_no_reserve<reservation_tester_creator>
{
    return {};
}


STRF_TEST_FUNC void test_reserve()
{
    // on non-const rval ref
    constexpr std::size_t not_reserved = 0;

    {
        auto size = reservation_test()  ("abcd");
        TEST_EQ(size, not_reserved);
    }
    {
        auto size = reservation_test() .reserve(5555) ("abcd");
        TEST_EQ(size, 5555);
    }
    {
        auto size = reservation_test() .reserve_calc() ("abcd");
        TEST_EQ(size, 4);
    }

    // // on non-const ref

    {
        auto tester = reservation_test();
        auto size = tester ("abcd");
        TEST_EQ(size, not_reserved);
    }
    {
        auto tester = reservation_test();
        auto size = tester.reserve(5555) ("abcd");
        TEST_EQ(size, 5555);
    }
    {
        auto tester = reservation_test();
        auto size = tester.reserve_calc() ("abcd");
        TEST_EQ(size, 4);
    }

    // // on const ref

    {
        const auto tester = reservation_test();
        auto size = tester ("abcd");
        TEST_EQ(size, not_reserved);
    }
    {
        const auto tester = reservation_test();
        auto size = tester.reserve(5555) ("abcd");
        TEST_EQ(size, 5555);
    }
    {
        const auto tester = reservation_test() .reserve(5555);
        auto size = tester.reserve_calc() ("abcd");
        TEST_EQ(size, 4);
    }

    // // on const rval ref

    {
        const auto tester = reservation_test();
        auto size = std::move(tester) ("abcd");
        TEST_EQ(size, not_reserved);
    }
    {
        const auto tester = reservation_test();
        auto size = std::move(tester).reserve(5555) ("abcd");
        TEST_EQ(size, 5555);
    }
    {
        const auto tester = reservation_test() .reserve(5555);
        auto size = std::move(tester).reserve_calc() ("abcd");
        TEST_EQ(size, 4);
    }
}

REGISTER_STRF_TEST(test_reserve);

