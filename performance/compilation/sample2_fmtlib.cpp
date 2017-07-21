#include "fmt/format.h"
#include "fmt/ostream.h"

void sample1(std::FILE* out)
{
    fmt::print(out, "{}", 20);
    fmt::print(out, "{}  {}", 20, 20u);
    fmt::print(out, "{}  {}  {}", 20, 20u, 20l);
    fmt::print(out, "{}  {}  {}  {}  {}  {}", 20, 20u, 20l, 20ul, 20ll, 20ul);
    fmt::print(out, "aaa {<10x} bbb", 123);
    fmt::print(out, "{x} {} {} ", 123, "abcdef", 456);
}

void sample2(std::ostream& out)
{
    fmt::print(out, "{}", 20);
    fmt::print(out, "{}  {}", 20, 20u);
    fmt::print(out, "{}  {}  {}", 20, 20u, 20l);
    fmt::print(out, "{}  {}  {}  {}  {}  {}", 20, 20u, 20l, 20ul, 20ll, 20ul);
    fmt::print(out, "aaa {<10x} bbb", 123);
    fmt::print(out, "{x} {} {} ", 123, "abcdef", 456);
}

void sample3(char* out, std::size_t len)
{
    fmt::BasicArrayWriter<char> writer(out, len);
    writer.write(out, "{}", 20);
    writer.write(out, "{}  {}", 20, 20u);
    writer.write(out, "{}  {}  {}", 20, 20u, 20l);
    writer.write(out, "{}  {}  {}  {}  {}  {}", 20, 20u, 20l, 20ul, 20ll, 20ul);
    writer.write(out, "aaa {<10x} bbb", 123);
    writer.write(out, "{x} {} {} ", 123, "abcdef", 456);
}

void sample3w(wchar_t* out, std::size_t len)
{
    fmt::BasicArrayWriter<wchar_t> writer(out, len);
    writer.write(out, L"{}", 20);
    writer.write(out, L"{}  {}", 20, 20u);
    writer.write(out, L"{}  {}  {}", 20, 20u, 20l);
    writer.write(out, L"{}  {}  {}  {}  {}  {}", 20, 20u, 20l, 20ul, 20ll, 20ul);
    writer.write(out, L"aaa {<10x} bbb", 123);
    writer.write(out, L"{x} {} {} ", 123, L"abcdef", 456);
}


std::string sample4()
{
    return fmt::format
        ( "{}  {}  {}  {}  {}  {} aaa {<10x} {}"
          , 20, 20u, 20l, 20ul, 20ll, 20ul, 123, "abc");
}


std::wstring sample4w()
{
    return fmt::format
        ( L"{}  {}  {}  {}  {}  {} aaa {<10x} {}"
          , 20, 20u, 20l, 20ul, 20ll, 20ul, 123, L"abc");
}



