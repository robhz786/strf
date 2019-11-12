//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

template <bool NoExcept, typename CharT >
using string_maker = typename std::conditional
    < NoExcept
    , strf::basic_string_maker_noexcept<CharT>
    , strf::basic_string_maker<CharT> >
    :: type;

template <bool NoExcept, typename CharT >
using string_appender = typename std::conditional
    < NoExcept
    , strf::basic_string_appender_noexcept<CharT>
    , strf::basic_string_appender<CharT> >
    :: type;

template <bool NoExcept, typename CharT>
void test_successfull_append()
{
    auto tiny_str = test_utils::make_tiny_string<CharT>();
    auto double_str = test_utils::make_double_string<CharT>();
    auto expected_content = tiny_str + double_str;

    std::basic_string<CharT> str;
    string_appender<NoExcept, CharT> ob(str);
    write(ob, tiny_str.c_str(), tiny_str.size());
    write(ob, double_str.c_str(), double_str.size());
    ob.finish();
    BOOST_TEST(str == expected_content);
}

template <bool NoExcept, typename CharT>
void test_successfull_make()
{
    auto tiny_str = test_utils::make_tiny_string<CharT>();
    auto double_str = test_utils::make_double_string<CharT>();
    auto expected_content = tiny_str + double_str;

    string_maker<NoExcept, CharT> ob;
    write(ob, tiny_str.c_str(), tiny_str.size());
    write(ob, double_str.c_str(), double_str.size());
    BOOST_TEST(ob.finish() == expected_content);
}

template <bool NoExcept, typename CharT>
class string_maker_that_throws_impl
    : public strf::detail::basic_outbuf_noexcept_switch<NoExcept, CharT>
    , protected strf::detail::string_writer_mixin
        < string_maker_that_throws_impl<NoExcept, CharT>, NoExcept, CharT >
{
public:

    using string_type = std::basic_string<CharT>;

    string_maker_that_throws_impl()
        : strf::detail::basic_outbuf_noexcept_switch<NoExcept, CharT>
            ( strf::outbuf_garbage_buf<CharT>()
            , strf::outbuf_garbage_buf_end<CharT>() )
    {
        this->set_pos(this->buf_begin());
        this->set_end(this->buf_end());
    }

    string_maker_that_throws_impl(const string_maker_that_throws_impl&) = delete;
    string_maker_that_throws_impl(string_maker_that_throws_impl&&) = delete;
    ~string_maker_that_throws_impl() = default;

    string_type finish()
    {
        this->do_finish();
        return std::move(_str);
    }

    void throw_on_next_append(bool throw_= true)
    {
        _throw = throw_;
    }

private:

    template <typename, bool, typename>
    friend class strf::detail::string_writer_mixin;

    void _append(const CharT* begin, const CharT* end)
    {
        if (_throw)
        {
            throw std::bad_alloc();
        }
        _str.append(begin, end);
    }

    string_type _str;
    bool _throw = false;
};

template <bool NoExcept, typename CharT>
class string_maker_that_throws;

template <typename CharT>
class string_maker_that_throws<true, CharT>
    : public string_maker_that_throws_impl<true, CharT>
{
public:

    void recycle() noexcept override
    {
        this->do_recycle();
    }
};

template <typename CharT>
class string_maker_that_throws<false, CharT>
    : public string_maker_that_throws_impl<false, CharT>
{
public:

    void recycle() override
    {
        this->do_recycle();
    }
};

template <typename CharT>
void test_recycle_catches_exception()
{
    string_maker_that_throws<true, CharT> ob;
    auto double_str = test_utils::make_double_string<CharT>();
    auto half_str = test_utils::make_half_string<CharT>();

    write(ob, double_str.c_str(), double_str.size());
    ob.recycle();
    ob.throw_on_next_append();

    write(ob, half_str.c_str(), half_str.size());
    ob.recycle();
    BOOST_TEST(!ob.good());

    write(ob, half_str.c_str(), half_str.size());
    ob.recycle();
    BOOST_TEST(!ob.good());

    BOOST_TEST_THROWS((void)ob.finish(), std::bad_alloc);
    BOOST_TEST(!ob.good());

    BOOST_TEST_THROWS((void)ob.finish(), std::bad_alloc);
    BOOST_TEST(!ob.good());
}

template <typename CharT>
void test_recycle_that_throws()
{
    string_maker_that_throws<false, CharT> ob;
    auto double_str = test_utils::make_double_string<CharT>();
    auto half_str = test_utils::make_half_string<CharT>();
    auto expected_content = double_str;
    write(ob, double_str.c_str(), double_str.size());
    ob.recycle();
    ob.throw_on_next_append();

    write(ob, half_str.c_str(), half_str.size());
    BOOST_TEST_THROWS(ob.recycle(), std::bad_alloc);
    BOOST_TEST(!ob.good());

    ob.throw_on_next_append(false);

    write(ob, half_str.c_str(), half_str.size());
    ob.recycle(); // must be no-op
    BOOST_TEST(!ob.good());

    BOOST_TEST(ob.finish() == expected_content);
    BOOST_TEST(!ob.good());
}

template <typename CharT>
void test_dispatchers()
{
    auto double_str = test_utils::make_double_string<CharT>();
    auto half_str = test_utils::make_half_string<CharT>();
    {
        auto s = strf::to_basic_string<CharT>(half_str);
        BOOST_TEST(s == half_str);
    }
    {
        auto s = strf::to_basic_string<CharT>.reserve(20) (half_str);
        BOOST_TEST(s == half_str);
    }
    {
        auto s = strf::to_basic_string<CharT>.reserve_calc() (half_str);
        BOOST_TEST(s == half_str);
    }
    {
        auto s = double_str;
        strf::append(s) (half_str);
        BOOST_TEST(s == double_str + half_str);
    }
    {
        auto s = double_str;
        strf::append(s).reserve(20) (half_str);
        BOOST_TEST(s == double_str + half_str);
    }
    {
        auto s = double_str;
        strf::append(s).reserve_calc() (half_str);
        BOOST_TEST(s == double_str + half_str);
    }
}


int main()
{
    test_dispatchers<char>();
    test_dispatchers<char16_t>();

    test_successfull_append<true, char>();
    test_successfull_append<true, char16_t>();
    test_successfull_append<false, char>();
    test_successfull_append<false, char16_t>();

    test_successfull_make<true, char>();
    test_successfull_make<true, char16_t>();
    test_successfull_make<false, char>();
    test_successfull_make<false, char16_t>();

    test_recycle_catches_exception<char32_t>();
    test_recycle_catches_exception<wchar_t>();
    test_recycle_that_throws<char>();
    test_recycle_that_throws<char16_t>();

    return boost::report_errors();
}
