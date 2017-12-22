//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define  _CRT_SECURE_NO_WARNINGS

#include <boost/detail/lightweight_test.hpp>
#include "test_utils.hpp"
#include "error_code_emitter_arg.hpp"
#include <boost/stringify.hpp>

namespace strf = boost::stringify::v0;

int main()
{
    {
        char buff[200];
        using t1 = decltype(strf::write_to(buff) &= { "aaaa", 555 });
        using t2 = decltype(strf::write_to(buff) ["{}...{}"] &= { "aaaa", 555 });
        using t3 = decltype(strf::make_string() &= { "aaaa", 555 });
        using t4 = decltype(strf::make_string ["{}{}"] &= { "aaaa", 555 });

        static_assert(std::is_same<t1, void>::value, "expected void type");
        static_assert(std::is_same<t2, void>::value, "expected void type");
        static_assert(std::is_same<t3, std::string>::value, "expected std::string");
        static_assert(std::is_same<t4, std::string>::value, "expected std::string");
    }

    {
        char buff[200];
        strf::write_to(buff) &= { "aaaa", 555 };
        BOOST_TEST(0 == strcmp(buff, "aaaa555"));
    }

    {
        char buff[200];
        strf::write_to(buff) ["{}...{}"] &= { "aaaa", 555 };
        BOOST_TEST(0 == strcmp(buff, "aaaa...555"));
    }

    {
        auto str = strf::make_string() &= { "aaaa", 555 };
        BOOST_TEST(str == "aaaa555");
    }

    {
        auto str = strf::make_string ["{} {}"]  &= { "aaa", "bbb" };
        BOOST_TEST(str == "aaa bbb");
    }

    error_tag erroneous_arg =
        { std::make_error_code(std::errc::address_family_not_supported) };

    {
        std::error_code err_code;
        char buff [200] = "aaaa";
        try
        {
            strf::write_to(buff) &= {"aaa", erroneous_arg};
        }
        catch (const std::system_error& sys_err)
        {
            err_code = sys_err.code();
        }

        BOOST_TEST(err_code == erroneous_arg.ec);
        BOOST_TEST(buff[0] == '\0');
    }

    {
        std::error_code err_code;
        char buff [200] = "aaaa";
        try
        {
            strf::write_to(buff) ["{}...{}"] &= {"aaa", erroneous_arg};
        }
        catch (const std::system_error& sys_err)
        {
            err_code = sys_err.code();
        }

        BOOST_TEST(err_code == erroneous_arg.ec);
        BOOST_TEST(buff[0] == '\0');
    }

    {
        std::error_code err_code;
        try
        {
            auto str = strf::make_string() &= {"aaa", erroneous_arg};
        }
        catch (const std::system_error& sys_err)
        {
            err_code = sys_err.code();
        }

        BOOST_TEST(err_code == erroneous_arg.ec);
    }

    {
        std::error_code err_code;
        try
        {
            auto str = strf::make_string ["{} {}"] &= {"aaa", erroneous_arg};
        }
        catch (const std::system_error& sys_err)
        {
            err_code = sys_err.code();
        }

        BOOST_TEST(err_code == erroneous_arg.ec);
    }


    return report_errors() || boost::report_errors();
}
