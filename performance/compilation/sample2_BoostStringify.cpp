#include <boost/stringify.hpp>

namespace strf = boost::stringify;

void sample1(std::FILE* out)
{
    strf::write_to(out) (20);
    strf::write_to(out) (20, 20u);
    strf::write_to(out) (20, 20u, 20l);
    strf::write_to(out) (20, 20u, 20l, 20ul, 20ll, 20ul);
    strf::write_to(out) ("aaa ", {123, {10, "<x"}}, "bbb");
    strf::write_to(out) ({123, "x"}, "abcdef", 456);
}

void sample2(std::streambuf& out)
{
    strf::write_to(out) (20);
    strf::write_to(out) (20, 20u);
    strf::write_to(out) (20, 20u, 20l);
    strf::write_to(out) (20, 20u, 20l, 20ul, 20ll, 20ul);
    strf::write_to(out) ("aaa ", {123, {10, "<x"}}, "bbb");
    strf::write_to(out) ({123, "x"}, "abcdef", 456);
}

void sample3(char* out)
{
    strf::write_to(out) (20);
    strf::write_to(out) (20, 20u);
    strf::write_to(out) (20, 20u, 20l);
    strf::write_to(out) (20, 20u, 20l, 20ul, 20ll, 20ul);
    strf::write_to(out) ("aaa ", {123, {10, "<x"}}, "bbb");
    strf::write_to(out) ({123, "x"}, "abcdef", 456);
}

void sample3(wchar_t* out)
{
    strf::write_to(out) (20);
    strf::write_to(out) (20, 20u);
    strf::write_to(out) (20, 20u, 20l);
    strf::write_to(out) (20, 20u, 20l, 20ul, 20ll, 20ul);
    strf::write_to(out) (L"aaa ", {123, {10, "<x"}}, L"bbb");
    strf::write_to(out) ({123, "x"}, L"abcdef", 456);
}

std::string sample4()
{
    return strf::make_string
        [{ 20, " ", 20u, " ", 20l, " ", 20ul, " ", 20ll, " ", 20ul
         , " aaa ", {123, {10, "<x"}}, " ", "abc"}];
}

std::wstring sample4w()
{
    return strf::make_wstring
        [{ 20, L" ", 20u, L" ", 20l, L" ", 20ul, L" ", 20ll, L" ", 20ul
         , L" aaa ", {123, {10, "<x"}}, L" ", L"abc"}];
}
