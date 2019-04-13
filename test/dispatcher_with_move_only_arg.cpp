//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify.hpp>
#include "test_utils.hpp"

namespace strf = boost::stringify::v0;

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

class foo_writer: public strf::output_buffer<char>
{
public:

    foo_writer(foo&& foo_)
        : _foo(std::move(foo_))
    {
        this->set_pos(_foo.dest);
        this->set_end(_foo.dest_end - 1);
    }

    bool recycle() override
    {
        return false;
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
    return strf::dispatcher<strf::facets_pack<>, foo_writer, foo>
        (strf::pack(), std::move(foo_));
}



int main()
{
    char buff[500];

    write(foo(buff, buff + sizeof(buff))) ("abcd", 1234);

    BOOST_TEST_CSTR_EQ("abcd1234", buff);

    write(foo(buff, buff + sizeof(buff))).as("--{}--{}--", "abcd", 1234);

    BOOST_TEST_CSTR_EQ("--abcd--1234--", buff);

    return boost::report_errors();
}


