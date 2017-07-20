#include <fmt/format.h>

void write_sample(std::FILE* out)
{
    fmt::print(out, "{}", 20);
    fmt::print(out, "{}  {}", 20, 20u);
    fmt::print(out, "{}  {}  {}", 20, 20u, 20l);
    fmt::print(out, "{}  {}  {}  {}  {}  {}", 20, 20u, 20l, 20ul, 20ll, 20ul);
    fmt::print(out, "aaa {<10x} bbb", 123);
    fmt::print(out, "{x} {} {} ", 123, "abcdef", 456);
}
