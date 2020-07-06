//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"
#include <vector>

struct fcategory;

struct facet_type
{
    using category = fcategory;
    int value = 0;
};

struct fcategory
{
    constexpr static bool constrainable = true;
    static auto STRF_HD get_default() noexcept
    {
        return facet_type {};
    }
};

template <int I> struct input_type
{
    static constexpr int type_n_id = I;
};

template<typename T, int N> struct filter_le
{
    static constexpr bool value = (T::type_n_id <= N);
};


template<typename T> using filter_le1 = filter_le<T, 1>;
template<typename T> using filter_le2 = filter_le<T, 2>;
template<typename T> using filter_le3 = filter_le<T, 3>;
template<typename T> using filter_le4 = filter_le<T, 4>;
template<typename T> using filter_le5 = filter_le<T, 5>;
template<typename T> using filter_le6 = filter_le<T, 6>;
template<typename T> using filter_le7 = filter_le<T, 7>;


facet_type f1 = {1};
facet_type f2 = {2};
facet_type f3 = {3};
facet_type f4 = {4};
facet_type f5 = {5};
facet_type f6 = {6};
facet_type f7 = {7};


auto x1 = strf::constrain<filter_le1>(f1);
auto x2 = strf::constrain<filter_le2>(f2);
auto x3 = strf::constrain<filter_le3>(f3);
auto x4 = strf::constrain<filter_le4>(f4);
auto x5 = strf::constrain<filter_le5>(f5);
auto x6 = strf::constrain<filter_le6>(f6);
auto x7 = strf::constrain<filter_le7>(f7);

template <typename FPack>
std::vector<int> digest(const FPack& fp)
{
    return std::vector<int>
    {
        strf::get_facet< fcategory, input_type<1> >(fp).value,
        strf::get_facet< fcategory, input_type<2> >(fp).value,
        strf::get_facet< fcategory, input_type<3> >(fp).value,
        strf::get_facet< fcategory, input_type<4> >(fp).value,
        strf::get_facet< fcategory, input_type<5> >(fp).value,
        strf::get_facet< fcategory, input_type<6> >(fp).value,
        strf::get_facet< fcategory, input_type<7> >(fp).value
    };
}

std::vector<int> expected = {1, 2, 3, 4, 5, 6, 7};


void test_facets_pack_merge()
{

    {
        auto fp = strf::pack(x7, x6, x5, x4, x3, x2, x1);
        TEST_TRUE(digest(fp) == expected);
    }

    {
        auto fp = strf::pack(x7, x6, x5, x5, x4, x5, x4, x3, x2, x1, x1);
        TEST_TRUE(digest(fp) == expected);
    }

    {
        auto fp = strf::pack
            (strf::pack(x7), strf::pack(x6, x5), x4, x3, x2, x1);
        TEST_TRUE(digest(fp) == expected);
    }

    {
        auto fp = strf::pack
            (x7, x6, x5, x4, x3, x2, strf::pack(x1));
        TEST_TRUE(digest(fp) == expected);
    }

    {
        auto fp = strf::pack
            (x7, x6, x5, strf::pack(x4, x3), x2, x1);
        TEST_TRUE(digest(fp) == expected);
    }

    {
        auto fp = strf::pack
            ( strf::pack(x7)
            , strf::pack(x6)
            , strf::pack(x5)
            , strf::pack(x4)
            , strf::pack(x3)
            , strf::pack(x2)
            , strf::pack(x1)
            );
        TEST_TRUE(digest(fp) == expected);
    }
    {
        auto fp = strf::pack
            ( strf::pack(strf::pack(x7))
            , strf::pack(strf::pack(x6))
            , strf::pack(strf::pack(x5))
            , strf::pack(strf::pack(x4))
            , strf::pack(strf::pack(x3))
            , strf::pack(strf::pack(x2))
            , strf::pack(strf::pack(x1))
            );
        TEST_TRUE(digest(fp) == expected);
    }
    {
        auto fp = strf::pack
            ( strf::pack(strf::pack(x7), strf::pack(x6))
            , strf::pack(strf::pack(x5), strf::pack(x4))
            , strf::pack(strf::pack(x3), strf::pack(x2))
            , strf::pack(strf::pack(x2), strf::pack(x1))
            );
        TEST_TRUE(digest(fp) == expected);
    }
    {
        auto fp = strf::pack
            ( strf::pack(strf::pack(x7, x6, x5))
            , strf::pack(x6, x5, x4)
            , x4, x3, x2
            , strf::pack(strf::pack(x2, x1))
            );
        TEST_TRUE(digest(fp) == expected);
    }
    {
        auto fp = strf::pack
            ( strf::pack()
            , strf::pack(strf::pack(x7, x6))
            , strf::pack(x5, x4, x5, x4)
            , strf::pack(x5, x4)
            , x3
            , x2
            , strf::pack(strf::pack(x2, x1))
            , strf::pack()
            );

        TEST_TRUE(digest(fp) == expected);
    }
}


