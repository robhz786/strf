#include <cstdio>

void write_sample(std::FILE* out)
{
    std::fprintf(out, "%d", 20);
    std::fprintf(out, "%d  %u", 20, 20u);
    std::fprintf(out, "%d  %u  %ld", 20, 20u, 20l);
    std::fprintf(out, "%d  %u  %ld  %lu  %lld  %llu", 20, 20u, 20l, 20ul, 20ll, 20ul);
    std::fprintf(out, "aaa %-10x bbb", 123);
    std::fprintf(out, "%x %s %d", 123, "abcdef", 456);
}
