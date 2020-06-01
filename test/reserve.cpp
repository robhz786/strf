//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define  _CRT_SECURE_NO_WARNINGS

#include "test_utils.hpp"
#include <strf.hpp>

class reservation_tester : public strf::basic_outbuff<char>
{
    constexpr static std::size_t buff_size_ = strf::min_size_after_recycle<1>();
    char buff_[buff_size_];

public:

    reservation_tester()
        : strf::basic_outbuff<char>{ buff_, buff_ + buff_size_ }
        , buff_{0}
    {
    }

    reservation_tester(std::size_t size)
        : strf::basic_outbuff<char>{ buff_, buff_ + buff_size_ }
        , buff_{0}
        , reserved_size_{size}
    {
    }

#if defined(STRF_NO_CXX17_COPY_ELISION)

    reservation_tester(reservation_tester&& other);

#else // defined(STRF_NO_CXX17_COPY_ELISION)

    reservation_tester(const reservation_tester&) = delete;
    reservation_tester(reservation_tester&&) = delete;

#endif // defined(STRF_NO_CXX17_COPY_ELISION)

    void recycle() override
    {
        this->set_pointer(buff_);
    }

    std::size_t finish()
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

    template <typename ... Printers>
    std::size_t write(const Printers& ... printers) const
    {
        reservation_tester ob;
        strf::detail::write_args(ob, printers...);;
        return ob.finish();
    }

    template <typename ... Printers>
    std::size_t sized_write(std::size_t size, const Printers& ... printers) const
    {
        reservation_tester ob{size};
        strf::detail::write_args(ob, printers...);;
        return ob.finish();
    }
    reservation_tester create() const
    {
        return reservation_tester{};
    }
    reservation_tester create(std::size_t size) const
    {
        return reservation_tester{size};
    }
};

constexpr auto reservation_test()
{
    return strf::destination_no_reserve<reservation_tester_creator>();
}


int main()
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

    return test_finish();
}
