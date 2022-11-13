//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/to_string.hpp>

class my_notifier: public strf::transcoding_error_notifier {
public:
    explicit my_notifier(strf::destination<char>& dest)
        : dest_(dest)
    {
    }

    void unsupported_codepoint(const char* charset, unsigned codepoint) override
    {
        strf::to(dest_).with(strf::uppercase)
            ( "The codepoint U+", strf::hex(codepoint).pad0(4)
            , " can not be encoded in ", charset, ".\n");
    }

    void invalid_sequence
        ( int code_unit_size
        , const char* charset_name
        , const void* sequence_ptr
        , std::ptrdiff_t count ) override
    {
        if (code_unit_size != 1 && code_unit_size != 2 && code_unit_size != 4) {
            strf::to(dest_) ( "Invalid sequence and invalid code unit size ("
                            , code_unit_size, ").\n" );
            return;
        }
        strf::to(dest_) ( "The sequence ");
        while (count--) {
            auto unit = extract_unit_and_advance(sequence_ptr, code_unit_size);
            strf::to(dest_)
                .with(strf::mixedcase)
                (*strf::hex(unit).pad0(static_cast<unsigned>(code_unit_size) * 2), ' ');
        }
        strf::to(dest_) ( "is not valid in ", charset_name, ".\n");
    }

private:

    static std::ptrdiff_t extract_unit_and_advance(const void*& ptr, std::ptrdiff_t size) {
        const auto *uptr = static_cast<const unsigned char*>(ptr);
        ptr = uptr + size;
        return extract_unit(uptr, size);
    }

    static std::ptrdiff_t extract_unit(const unsigned char* uptr, std::ptrdiff_t size) {
        switch (size) {
            case 1:
                return *uptr;
            case 2: {
                std::uint16_t tmp = 0;
                memcpy(&tmp, uptr, 2);
                return tmp;
            }
            default: {
                assert(size == 4);
                std::uint32_t tmp = 0;
                memcpy(&tmp, uptr, 4);
                return tmp;
            }
        }
    }

    strf::destination<char>& dest_;
};

#if ! defined(__cpp_char8_t)
using char8_t = char;
#endif


int main()
{
    strf::string_maker err_str_maker;
    my_notifier notifier(err_str_maker);
    strf::transcoding_error_notifier_ptr notifier_ptr{&notifier};

    auto output = strf::to_string.with(strf::iso_8859_1<char>, notifier_ptr)
        ( strf::transcode("---\xF0\x90\xBF---", strf::utf8<char>)    // invalid input
        , strf::transcode(u8"...\u20DF...",  strf::utf8<char8_t>) ); // unsupported codepoint

    auto err_msg = err_str_maker.finish();

    assert(output == "---?---...?...");
    assert(err_msg ==
           "The sequence 0xF0 0x90 0xBF is not valid in UTF-8.\n"
           "The codepoint U+20DF can not be encoded in ISO-8859-1.\n" );

    return 0;
}
