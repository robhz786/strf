//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"
#include <boost/stringify.hpp>

template <int N> struct fcategory;

struct ctor_log
{
    int cp_count = 0;
    int mv_count = 0;
};

template <int N, bool BV = true>
class facet
{
public:
    constexpr facet(int value_, ctor_log* log_ = nullptr)
        : value(value_)
        , _log(log_) {}

    constexpr facet(const facet& f)
        : value(f.value)
        , _log(f._log)
    {
        if(_log)
        {
            ++ _log->cp_count;
        }
    }

    constexpr facet(facet&& f)
        : value(f.value)
        , _log(f._log)
    {
        if(_log)
        {
            ++ _log->mv_count;
        }
    }

    using category = fcategory<N>;
    static constexpr bool store_by_value = BV;
    int value = 0;

private:

    ctor_log* _log;
};

template <int N> struct fcategory
{
    constexpr static bool constrainable = true;

    constexpr static facet<N> get_default() noexcept
    {
        return facet<N>{-1};
    }
};

template <typename T>
using is_64 = std::integral_constant<bool, sizeof(T) == 8>;

namespace strf = boost::stringify;

int main()
{
    { // check constexpr

        constexpr facet<0> f {10};
        constexpr auto c = strf::constrain<is_64>(f);
        constexpr auto c2 = c;
        constexpr auto c3 = std::move(c2);

        constexpr auto fp = pack(c3);
        constexpr decltype(fp) fp2{fp};
        constexpr decltype(fp) fp3{std::move(fp2)};

        static_assert(strf::get_facet<fcategory<0>, double>(fp).value == 10, " ");
        static_assert(strf::get_facet<fcategory<0>, float>(fp).value == -1, " ");

    }
    {   // constrain a facet copy
        ctor_log log;
        facet<0> f{10, &log};
        auto c = strf::constrain<is_64>(f);
        auto c2 = strf::constrain<std::is_integral>(c);
        auto c3 = strf::constrain<std::is_signed>(c2);
        auto fp = pack(c3);
        decltype(fp) fp2{fp};
        const auto fp3 = pack(fp2);

        BOOST_TEST_EQ(log.cp_count, 6);
        BOOST_TEST_EQ(log.mv_count, 0);
        BOOST_TEST_EQ(10, (strf::get_facet<fcategory<0>, std::int64_t>(fp3).value));
        BOOST_TEST_EQ(-1, (strf::get_facet<fcategory<0>, std::int32_t>(fp3).value));
        BOOST_TEST_EQ(-1, (strf::get_facet<fcategory<0>, std::uint64_t>(fp3).value));
        BOOST_TEST_EQ(-1, (strf::get_facet<fcategory<0>, double>(fp3).value));
    }

    {   // constrain a facet with store_by_value=false
        ctor_log log;
        facet<0, false> f{10, &log};
        auto c = strf::constrain<is_64>(f);
        auto c2 = strf::constrain<std::is_integral>(c);
        auto c3 = strf::constrain<std::is_signed>(c2);
        auto fp = pack(c3);

        BOOST_TEST_EQ(log.cp_count, 0);
        BOOST_TEST_EQ(log.mv_count, 0);
        BOOST_TEST_EQ(&f, (&strf::get_facet<fcategory<0>, std::int64_t>(fp)));
        BOOST_TEST_EQ(-1, (strf::get_facet<fcategory<0>, std::int32_t>(fp).value));
        BOOST_TEST_EQ(-1, (strf::get_facet<fcategory<0>, std::uint64_t>(fp).value));
        BOOST_TEST_EQ(-1, (strf::get_facet<fcategory<0>, double>(fp).value));
    }

    {   // constrain a facet reference
        ctor_log log;
        facet<0> f{10, &log};
        auto c = strf::constrain<is_64>(std::cref(f));
        auto c2 = strf::constrain<std::is_integral>(c);
        auto c3 = strf::constrain<std::is_signed>(c2);
        auto fp = pack(c3);

        BOOST_TEST_EQ(log.cp_count, 0);
        BOOST_TEST_EQ(log.mv_count, 0);
        BOOST_TEST_EQ(&f, (&strf::get_facet<fcategory<0>, std::int64_t>(fp)));
        BOOST_TEST_EQ(-1, (strf::get_facet<fcategory<0>, std::int32_t>(fp).value));
        BOOST_TEST_EQ(-1, (strf::get_facet<fcategory<0>, std::uint64_t>(fp).value));
        BOOST_TEST_EQ(-1, (strf::get_facet<fcategory<0>, double>(fp).value));
    }

    {   // constrain a facets_pack

        auto fp = strf::pack
            ( strf::constrain<std::is_signed>(facet<1>(201))
            , facet<2>(202)
            , facet<3>(203) );
        auto fp2 = strf::pack
            ( facet<1>(101)
            , strf::constrain<is_64>(fp)
            , strf::constrain<std::is_integral>(facet<1>(301)) );


        BOOST_TEST_EQ(101, (strf::get_facet<fcategory<1>, float>(fp2).value));
        BOOST_TEST_EQ(201, (strf::get_facet<fcategory<1>, double>(fp2).value));
        BOOST_TEST_EQ(301, (strf::get_facet<fcategory<1>, int>(fp2).value));
    }


    {   // constrain a facets_pack reference

        auto fp = strf::pack
            ( strf::constrain<std::is_signed>(facet<1>(201))
            , facet<2>(202)
            , facet<3>(203) );
        auto fp2 = strf::pack
            ( facet<1>(101)
            , strf::constrain<is_64>(std::ref(fp))
            , strf::constrain<std::is_integral>(facet<1>(301)) );

        BOOST_TEST_EQ(101, (strf::get_facet<fcategory<1>, float>(fp2).value));
        BOOST_TEST_EQ(201, (strf::get_facet<fcategory<1>, double>(fp2).value));
        BOOST_TEST_EQ(301, (strf::get_facet<fcategory<1>, int>(fp2).value));

        BOOST_TEST_EQ( &(strf::get_facet<fcategory<1>, double>(fp2))
                     , &(strf::get_facet<fcategory<1>, double>(fp)) );
    }

    return boost::report_errors();
}



