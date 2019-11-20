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

class foo_writer: public strf::basic_outbuf<char>
{
public:

    foo_writer(foo&& foo_)
        : strf::basic_outbuf<char>{foo_.dest, foo_.dest_end - 1}
        , _foo(std::move(foo_))
    {
    }

    void recycle() override
    {
        this->set_pos(_foo.dest);
    }

    void finish()
    {
        *this->pos() = '\0';
    }

private:

    foo _foo;
};


auto write(foo&& foo_)
{
    return strf::destination<strf::facets_pack<>, foo_writer, foo>(std::move(foo_));
}



int main()
{
    char buff[500];

    write(foo(buff, buff + sizeof(buff))) ("abcd", 1234);

    BOOST_TEST_CSTR_EQ("abcd1234", buff);

    write(foo(buff, buff + sizeof(buff))).tr("--{}--{}--", "abcd", 1234);

    BOOST_TEST_CSTR_EQ("--abcd--1234--", buff);

    return boost::report_errors();
}


