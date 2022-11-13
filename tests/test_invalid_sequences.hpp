#ifndef TEST_INVALID_SEQUENCES_HPP
#define TEST_INVALID_SEQUENCES_HPP

#include "test_utils.hpp"

namespace test_utils {

STRF_HD inline void memory_copy(void* to, const void* from, std::size_t count) {
#if defined(STRF_WITH_CSTRING)
    memcpy(to, from, count);
#else
    auto to_ = reinterpret_cast<char*>(to);
    auto from_ = reinterpret_cast<const char*>(from);
    for (; count; --count) {
        *to_++ = *from_ ++;
    }
#endif
}

template <typename CharT>
class invalid_sequences_tester: public strf::transcoding_error_notifier {
public:
    STRF_HD invalid_sequences_tester
        ( span<const strf::detail::simple_string_view<CharT>> expected_inv_seqs
        , test_failure_notifier& notifier )
        : expected_invalid_seqs_(expected_inv_seqs)
        , notifier_(notifier)
    {
    }

    STRF_HD ~invalid_sequences_tester() {
        if (counter_ < expected_invalid_seqs_.ssize()) {
            strf::to(notifier_) ("\n   Less invalid sequences than expected");
        }
    }

    STRF_HD void unsupported_codepoint
        ( const char* charset_name
        , unsigned codepoint ) override
    {
        (void) charset_name;
        (void) codepoint;
        strf::to(notifier_) ("\n   unsupported_codepoint was called. It was not supposed to.");
    }

    STRF_HD void invalid_sequence
        ( int code_unit_size
        , const char* charset_name
        , const void* sequence_ptr
        , std::ptrdiff_t code_units_count ) override
    {
        (void) charset_name;
        ++counter_;

        auto print = strf::to(notifier_).with(strf::lettercase::mixed);
        if (code_unit_size != sizeof(CharT)) {
            print("\n   code_units_count = ", code_unit_size, " (expected ", sizeof(CharT), ')');
            return;
        }
        if (counter_ > expected_invalid_seqs_.ssize()) {
            print("\n   More invalid sequences than expected");
            return;
        }
        auto expected_seq = expected_invalid_seqs_[counter_ - 1];
        auto const seq_mem = reinterpret_cast<const unsigned char*>(sequence_ptr);
        if (code_units_count == expected_seq.ssize()) {
            auto seq_mem_it = seq_mem;
            for (CharT expected_ch : expected_seq) {
                CharT ch;
                memory_copy(&ch, seq_mem_it, sizeof(ch));
                seq_mem_it += code_unit_size;
                if (ch != expected_ch) {
                    goto inform_difference;
                }
            }
        } else {
            inform_difference:
            switch(counter_) {
                case 1: print("\n  First"); break;
                case 2: print("\n  Second"); break;
                case 3: print("\n  Third"); break;
                default: print("\n  The ", counter_, "th");
            }
            print ( " invalid sequence is different than expected:"
                    "\n    Expected:");
            using UCharT = typename std::make_unsigned<CharT>::type;
            for (CharT expected_ch : expected_seq) {
                print (' ', *strf::hex((unsigned)(UCharT)expected_ch));
            }
            print ("\n    Obtained:");
            auto obtained_it = static_cast<const unsigned char*>(sequence_ptr);
            for (std::ptrdiff_t i = 0; i < code_units_count; ++i) {
                CharT ch;
                memory_copy(&ch, obtained_it, sizeof(ch));
                print (' ', *strf::hex((unsigned)(UCharT)ch));
                obtained_it += code_unit_size;
            }
        }
    }

private:
    std::ptrdiff_t counter_ = 0;
    span<const strf::detail::simple_string_view<CharT>> expected_invalid_seqs_;
    test_failure_notifier& notifier_;
};

template <typename SrcCharT, typename DestCharT>
STRF_HD void do_test_invalid_sequences
    ( const char* funcname
    , const char* srcfile
    , int srcline
    , strf::surrogate_policy policy
    , strf::transcode_f<SrcCharT, DestCharT> transcode_func
    , strf::detail::simple_string_view<SrcCharT> input
    , std::initializer_list<strf::detail::simple_string_view<SrcCharT>>
         expected_invalid_sequences )
{
    test_failure_notifier notifier{funcname, srcfile, srcline};

    invalid_sequences_tester<SrcCharT> inv_seq_tester
        { span<const strf::detail::simple_string_view<SrcCharT>>
            { expected_invalid_sequences.begin(), expected_invalid_sequences.size() }
        , notifier };

    DestCharT buff[200];
    strf::array_destination<DestCharT> result_dest{buff};
    transcode_func( result_dest, input.data(), input.size(), &inv_seq_tester, policy );
    if (result_dest.finish().truncated) {
        strf::to(notifier) ("In do_test_invalid_sequences, internal buffer is to small");
    }

    notifier.finish();
}

template < strf::charset_id SrcId, strf::charset_id DestId
         , typename SrcCharT, typename DestCharT, typename... T>
STRF_HD void test_invalid_sequences
    ( const char* function
    , const char* src_filename
    , int src_line
    , strf::surrogate_policy policy
    , const SrcCharT* input
    , T... expected_invalid_sequences )
{
    using transcoder_t = strf::static_transcoder<SrcCharT, DestCharT, SrcId, DestId>;
    using strview = strf::detail::simple_string_view<SrcCharT>;
    do_test_invalid_sequences<SrcCharT>
        ( function, src_filename, src_line, policy
        , transcoder_t::transcode_func(), input
        , {strview{expected_invalid_sequences}...} );
}

} // namespace test_utils


#endif // TEST_INVALID_SEQUENCES_HPP
