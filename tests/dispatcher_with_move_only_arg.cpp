//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf.hpp>
#include "test_utils.hpp"

struct foo // note: foo is not copiable
{
    foo(char* begin, char* end)
        : dest(begin)
        , dest_end(end)
    {
    }

    foo(const foo&) = delete;

    foo(foo&& f)
        : dest(f.dest)
        , dest_end(f.dest_end)
    {
        f.dest = nullptr;
        f.dest_end = nullptr;
    }

    char* dest;
    char* dest_end;
};

class foo_writer: public strf::basic_outbuff<char>
{
public:

    foo_writer(foo&& f)
        : strf::basic_outbuff<char>{f.dest, f.dest_end - 1}
        , foo_(std::move(f))
    {
    }

    void recycle() override
    {
        this->set_pointer(foo_.dest);
    }

    void finish()
    {
        *this->pointer() = '\0';
    }

private:

    foo foo_;
};


auto write(foo&& f)
{
    return strf::destination<strf::facets_pack<>, foo_writer, foo>(std::move(f));
}



int main()
{
    char buff[500];

    write(foo(buff, buff + sizeof(buff))) ("abcd", 1234);

    TEST_CSTR_EQ("abcd1234", buff);

    write(foo(buff, buff + sizeof(buff))).tr("--{}--{}--", "abcd", 1234);

    TEST_CSTR_EQ("--abcd--1234--", buff);

    return test_finish();
}


