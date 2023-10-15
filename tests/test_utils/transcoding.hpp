#ifndef TEST_UTILS_TRANSCODING_HPP
#define TEST_UTILS_TRANSCODING_HPP

#include "../test_utils.hpp"

namespace test_utils {

class test_transcode_err_messenger {
public:

    test_transcode_err_messenger() = default;

    test_transcode_err_messenger(const test_transcode_err_messenger&) = delete;
    test_transcode_err_messenger& operator=(const test_transcode_err_messenger&) = delete;

    STRF_HD test_transcode_err_messenger(test_transcode_err_messenger&& other)
    {
        *this = (test_transcode_err_messenger&&) other;
    }

    STRF_HD test_transcode_err_messenger& operator=(test_transcode_err_messenger&& other)
    {
        function_name_ = other.function_name_;
        filename_ = other.filename_;
        line_ = other.line_;
        failed_ = other.failed_;

        other.filename_ = nullptr;
        other.function_name_ = nullptr;
        other.line_ = 0;
        other.failed_ = false;

        return *this;
    }

    STRF_HD ~test_transcode_err_messenger()
    {
        if (function_name_ != nullptr && failed_) {
            test_utils::print_test_message_end(function_name_);
        }
    }

    STRF_HD test_transcode_err_messenger
        ( const char* function_name
        , const char* filename
        , int line )
        : function_name_(function_name)
        , filename_(filename)
        , line_(line)
    {
        STRF_ASSERT((function_name == nullptr) == (filename == nullptr));
    }

    template <typename... Args>
    STRF_HD void line(Args&&... args)
    {
        start_line();
        continue_line((Args&&)args...);
    }

    template <typename... Args>
    STRF_HD void start_line(Args&&... args)
    {
        if (filename_ != nullptr) {
            if (!failed_) {
                ++ test_err_count();
                test_utils::print_test_message_header(filename_, line_);
                failed_ = true;
            }
            to(test_utils::test_messages_destination()) ("\n    ", (Args&&)args...);
        }
    }

    template <typename... Args>
    STRF_HD void continue_line(Args&&... args)
    {
        if (filename_ != nullptr) {
            to(test_utils::test_messages_destination()) ((Args&&)args...);
        }
    }

private:
    const char* function_name_ = nullptr;
    const char* filename_ = nullptr;
    int line_ = 0;
    bool failed_ = false;
};


template <typename SrcCharT>
class transcoding_error_tester: public strf::transcoding_error_notifier
{
    using inv_seq_list_t = std::initializer_list<std::initializer_list<SrcCharT>>;
    using unsupported_codepoints_list_t = std::initializer_list<char32_t>;

    test_transcode_err_messenger& err_msg_;
    strf::detail::simple_string_view<char> src_charset_name = nullptr;
    strf::detail::simple_string_view<char> dst_charset_name = nullptr;
    int src_code_unit_size = 0;
    std::size_t inv_seqs_count_ = 0;
    std::size_t unsupporded_codepoints_count_ = 0;

    inv_seq_list_t expected_inv_seqs_;
    unsupported_codepoints_list_t expected_unsupported_codepoints_;

public:

    template <typename SrcCharset, typename DstCharset>
    STRF_HD transcoding_error_tester
        ( test_transcode_err_messenger& err_msg
        , SrcCharset src_charset
        , DstCharset dst_charset
        , inv_seq_list_t expected_inv_seqs
        , unsupported_codepoints_list_t expected_unsupported_codepoints )
        : err_msg_(err_msg)
        , src_charset_name(src_charset.name())
        , dst_charset_name(dst_charset.name())
        , src_code_unit_size(sizeof(typename SrcCharset::code_unit))
        , expected_inv_seqs_(expected_inv_seqs)
        , expected_unsupported_codepoints_(expected_unsupported_codepoints)
    {
    }

    STRF_HD void finish()
    {
        if (inv_seqs_count_ != expected_inv_seqs_.size()) {
            err_msg_.line("`invalid_sequence` was called ", inv_seqs_count_, " time(s)");
            err_msg_.line("   It was expected to be called ", expected_inv_seqs_.size(), " time(s)");
        }

        if (unsupporded_codepoints_count_ != expected_unsupported_codepoints_.size()) {
            err_msg_.line("`unsupporded_codepoint` was called "
                         , unsupporded_codepoints_count_, " time(s)");
            err_msg_.line("   It was expected to be called "
                         , expected_unsupported_codepoints_.size(), " time(s)");
        }
    }

    STRF_HD void unsupported_codepoint
        ( const char* charset_name
        , unsigned codepoint ) override
    {
        ++unsupporded_codepoints_count_;
        if (dst_charset_name != charset_name) {
            err_msg_.line("unsupported_codepoint(\"", charset_name, "\", ...) called");
            err_msg_.line("   But expected charset name \"", dst_charset_name, '\"');
        }
        if (unsupporded_codepoints_count_ > expected_unsupported_codepoints_.size()) {
            //err_msg_.line("unsupported_codepoint() called more times than expected");
            return;
        }
        const unsigned expected_codepoint =
            expected_unsupported_codepoints_.begin()[unsupporded_codepoints_count_ - 1];

        if (expected_codepoint != codepoint) {
            err_msg_.line("unsupported_codepoint(..., ", *strf::hex(codepoint), ") called");
            err_msg_.line("   But expected codepoint ", *strf::hex(expected_codepoint));
        }
    }

    STRF_HD void invalid_sequence
        ( int code_unit_size
        , const char* charset_name
        , const void* sequence_ptr
        , std::ptrdiff_t code_units_count ) override
    {
        ++inv_seqs_count_;
        if (src_charset_name != charset_name) {
            err_msg_.line("invalid_sequence(\"", charset_name, "\", ...) called");
            err_msg_.line("   But expected charset name \"", src_charset_name, '\"');
        }
        if (code_units_count <= 0) {
            err_msg_.line("invalid_sequence(..., ", code_units_count, ") called");
            err_msg_.line("   Parameter code_units_count should be positive");
            return;
        }
        if (code_unit_size <= 0) {
            err_msg_.line("invalid_sequence(", code_unit_size, ", ...) called");
            err_msg_.line("   Parameter code_unit_size should be positive");
            return;
        }
        if (inv_seqs_count_ > expected_inv_seqs_.size()) {
            //err_msg_.line("invalid_sequence() called more times than expected");
            return;
        }
        auto expected_seq_as_il = expected_inv_seqs_.begin()[inv_seqs_count_ - 1];
        auto expected_seq = strf::detail::simple_string_view<SrcCharT>
            ( expected_seq_as_il.begin()
            , expected_seq_as_il.size() );

        auto obtained_seq =strf::detail::simple_string_view<SrcCharT>
            ( reinterpret_cast<const SrcCharT*>(sequence_ptr)
            , code_units_count );

        if (expected_seq != obtained_seq) {
            err_msg_.line("Mismatch invalid_sequence at index ", inv_seqs_count_ - 1);
            err_msg_.start_line( "    Expected: ");
            print_sequence(expected_seq);
            err_msg_.start_line( "    Obtained: ");
            print_sequence(obtained_seq);
        }
    }

private:

    STRF_HD void print_sequence(strf::detail::simple_string_view<SrcCharT> seq)
    {
        using src_uchar_t = typename std::make_unsigned<SrcCharT>::type;
        err_msg_.continue_line( "[");
        for (auto ch: seq) {
            unsigned uch = static_cast<src_uchar_t>(ch);
            err_msg_.continue_line(' ', strf::hex(uch));
        }
        err_msg_.continue_line( " ]");
    }
};


template <typename SrcCharset, typename DstCharset>
struct transcoding_test_data
{
    STRF_HD transcoding_test_data
        ( SrcCharset src_charset_
        , DstCharset dst_charset_
        , bool safe )
        : src_charset{src_charset_}
        , dst_charset{dst_charset_}
        , safe_{safe}
    {
    }

    SrcCharset src_charset;
    DstCharset dst_charset;
    const bool safe_;

    using src_char_t = typename SrcCharset::code_unit;
    using dst_char_t = typename DstCharset::code_unit;
    using inv_seq_list_t = std::initializer_list<std::initializer_list<src_char_t>>;
    using unsupported_codepoints_list_t = std::initializer_list<char32_t>;

    strf::detail::simple_string_view<src_char_t> src;
    strf::detail::simple_string_view<dst_char_t> expected;
    std::size_t dst_size = 200;
    inv_seq_list_t expected_inv_seqs;
    unsupported_codepoints_list_t expected_unsupported_codepoints;
    strf::transcode_stop_reason expected_stop_reason = strf::transcode_stop_reason::completed;
    strf::transcode_flags flags = strf::transcode_flags::none;
    src_char_t buff_input[80];
    dst_char_t buff_expected[80];

    STRF_HD strf::decode_encode_result<src_char_t, dst_char_t> transcode
        ( dst_char_t* dst, dst_char_t* dst_end
        , strf::transcoding_error_notifier* err_notifier )
    {
        if (safe_) {
            return strf::transcode
                ( src_charset, dst_charset, src.begin(), src.end()
                , dst, dst_end, err_notifier, flags );
        }
        return strf::unsafe_transcode
            ( src_charset, dst_charset, src.begin(), src.end()
            , dst, dst_end, err_notifier, flags );
    }

    STRF_HD strf::decode_encode_size_result<src_char_t> transcode_size(const src_char_t* src_end)
    {
        if (safe_) {
            return strf::transcode_size
                ( src_charset, dst_charset, src.begin(), src_end, dst_size, flags );
        }
        return strf::unsafe_transcode_size
            ( src_charset, dst_charset, src.begin(), src_end, dst_size, flags );
    }

};

template <typename SrcCharset, typename DstCharset>
struct transcoding_test_data_maker
{
    using data_t = transcoding_test_data<SrcCharset, DstCharset>;
    using inv_seq_list_t = typename data_t::inv_seq_list_t;
    using unsupported_codepoints_list_t = typename data_t::unsupported_codepoints_list_t;

    STRF_HD transcoding_test_data_maker
        ( SrcCharset src_charset
        , DstCharset dst_charset
        , bool safe )
        : data{src_charset, dst_charset, safe}
    {
    }

    data_t data;
    using src_char_t = typename SrcCharset::code_unit;
    using dst_char_t = typename DstCharset::code_unit;

    template <typename... Args>
    STRF_HD transcoding_test_data_maker&& input(Args&&... args) &&
    {
        auto r = strf::to_range(data.buff_input) (args...);
        STRF_ASSERT(!r.truncated);
        data.src = strf::detail::make_simple_string_view(data.buff_input, r.ptr);
        return (transcoding_test_data_maker&&) *this;
    }
    template <typename... Args>
    STRF_HD transcoding_test_data_maker&& expect(Args&&... args) &&
    {
        auto r = strf::to_range(data.buff_expected) (args...);
        STRF_ASSERT(!r.truncated);
        data.expected = strf::detail::make_simple_string_view(data.buff_expected, r.ptr);
        return (transcoding_test_data_maker&&) *this;
    }
    STRF_HD transcoding_test_data_maker&& flags(strf::transcode_flags f) &&
    {
        data.flags = f;
        return (transcoding_test_data_maker&&) *this;
    }
    STRF_HD transcoding_test_data_maker&& destination_size(std::size_t s) &&
    {
        data.dst_size = s;
        return (transcoding_test_data_maker&&) *this;
    }
    STRF_HD transcoding_test_data_maker&& expect_stop_reason
        ( strf::transcode_stop_reason reason) &&
    {
        data.expected_stop_reason = reason;
        return (transcoding_test_data_maker&&) *this;
    }
    STRF_HD transcoding_test_data_maker&& expect_invalid_sequences
        (inv_seq_list_t seqs) &&
    {
        data.expected_inv_seqs = seqs;
        return (transcoding_test_data_maker&&) *this;
    }
    STRF_HD transcoding_test_data_maker&& expect_unsupported_codepoints
        (unsupported_codepoints_list_t codepoints) &&
    {
        data.expected_unsupported_codepoints = codepoints;
        return (transcoding_test_data_maker&&) *this;
    }
};

template <typename SrcCharset, typename DstCharset>
STRF_HD transcoding_test_data_maker<SrcCharset, DstCharset>
transcoding_test_data_maker_type(SrcCharset src_charset, DstCharset dst_charset);


inline STRF_HD const char* stringify(strf::transcode_stop_reason reason)
{
    switch(reason) {
    case strf::transcode_stop_reason::completed: return "completed";
    case strf::transcode_stop_reason::insufficient_output_space: return "insufficient_output_space";
    case strf::transcode_stop_reason::unsupported_codepoint: return "unsupported_codepoint";
    case strf::transcode_stop_reason::invalid_sequence: return "invalid_sequence";
    }
    return "(Invalid stop reason)";
}

template <typename SrcCharset, typename DstCharset>
class transcode_tester
{
    transcoding_test_data<SrcCharset, DstCharset> data_;
    test_transcode_err_messenger err_msg_;

public:
    using src_char_t = typename SrcCharset::code_unit;
    using dst_char_t = typename DstCharset::code_unit;

    STRF_HD transcode_tester
        ( transcoding_test_data<SrcCharset, DstCharset> data
        , test_transcode_err_messenger&& err_msg )
        : data_(data)
        , err_msg_((test_transcode_err_messenger&&)err_msg)
    {
    }

    STRF_HD void run();

    STRF_HD bool adjust_stale_src_ptr
        ( strf::decode_encode_result<src_char_t, dst_char_t>& res)
    {
        return adjust_stale_src_ptr(res.stale_src_ptr, res.u32dist, "transcode");
    }

    STRF_HD bool adjust_stale_src_ptr
        ( strf::decode_encode_size_result<src_char_t>& res)
    {
        return adjust_stale_src_ptr(res.stale_src_ptr, res.u32dist, "transcode_size");
    }

    STRF_HD bool adjust_stale_src_ptr
        ( const src_char_t*& stale_src_ptr, std::int32_t u32dist, const char* funcname )
    {
        const auto flags = data_.flags & strf::transcode_flags::surrogate_policy;
        auto res = data_.src_charset.to_u32().transcode_size
            ( stale_src_ptr, data_.src.end(), u32dist, flags);

        if (res.ssize != u32dist) {
            err_msg_.line("Wrong u32dist returned by ", funcname);
            err_msg_.line("    src_charset.to_u32().transcode_size(...).ssize == ", res.ssize);
            err_msg_.line("    while u32dist == ", u32dist);
            err_msg_.line("    ( they should be equal )", u32dist);
            return false;
        }
        stale_src_ptr = res.src_ptr;
        return true;
    }
};


template <typename SrcCharset, typename DstCharset>
STRF_HD void transcode_tester<SrcCharset, DstCharset>::run()
{
    const std::size_t buff_size = 200;
    dst_char_t buff[buff_size];
    if (data_.dst_size > buff_size) {
        err_msg_.line( "WARNING, requested destination_size is larget than "
                       "internal buffer's size. SKIPPING TEST CASE");
        return;
    }

    transcoding_error_tester<src_char_t> notifier
        { err_msg_, data_.src_charset, data_.dst_charset
        , data_.expected_inv_seqs
        , data_.expected_unsupported_codepoints };

    auto res_tr = data_.transcode(buff, buff + data_.dst_size, &notifier);
    notifier.finish();

    if (!adjust_stale_src_ptr(res_tr)) {
        return;
    }

    strf::detail::simple_string_view<dst_char_t> result_string(buff, res_tr.dst_ptr);
    if (result_string != data_.expected) {
        err_msg_.line("Wrong output");
        err_msg_.line("    Expected: \"", strf::transcode(data_.expected), '\"');
        err_msg_.line("    Obtained: \"", strf::transcode(result_string), '\"');
    }
    if (res_tr.stop_reason != data_.expected_stop_reason) {
        err_msg_.line("`transcode` returned wrong stop_reason");
        err_msg_.line("    Expected: ", stringify(data_.expected_stop_reason));
        err_msg_.line("    Obtained: ", stringify(res_tr.stop_reason));
    }

    auto src_end = ( data_.expected_stop_reason == strf::transcode_stop_reason::insufficient_output_space
                   ? res_tr.stale_src_ptr
                   : data_.src.end() );

    auto res_tr_size = data_.transcode_size(src_end);
    if (!adjust_stale_src_ptr(res_tr_size)) {
        return;
    }
    auto transcode_stop_reason =
        static_cast<strf::transcode_stop_reason>(res_tr_size.stop_reason);

    if (result_string.ssize() != res_tr_size.ssize) {
        err_msg_.line("Size mismatch:");
        err_msg_.line("    `transcode_size` calulated : ", res_tr_size.ssize);
        err_msg_.line("    But the output's size was  : ", result_string.ssize());
    }
    if (data_.expected_stop_reason != strf::transcode_stop_reason::insufficient_output_space) {
        if (data_.expected_stop_reason != transcode_stop_reason ) {
            err_msg_.line("`transcode_size` returned wrong stop_reason");
            err_msg_.line("    Expected: ", stringify(data_.expected_stop_reason));
            err_msg_.line("    Obtained: ", stringify(transcode_stop_reason));
        }
        if (transcode_stop_reason != res_tr.stop_reason) {
            err_msg_.line("`stop_reason` mismatch:");
            err_msg_.line("    `transcode`      returned `"
                          , stringify(res_tr.stop_reason), '`');
            err_msg_.line("    `transcode_size` returned `"
                          , stringify(transcode_stop_reason), '`');
        }
    }
    if (res_tr.stale_src_ptr != res_tr_size.stale_src_ptr) {
        err_msg_.line("Adjusted `stale_src_ptr` mismatch:");
        err_msg_.line( "    `transcode`      : `input.data() + "
                       , res_tr.stale_src_ptr - data_.src.begin(), '`');
        err_msg_.line( "    `transcode_size` : `input.data() + "
                       , res_tr_size.stale_src_ptr - data_.src.begin(), '`');
    }
}

class transcode_tester_caller
{
public:
    STRF_HD transcode_tester_caller
        ( const char* function_name
        , const char* filename
        , int line )
        : messenger_(function_name, filename, line)
    {
    }

    template <typename SrcCharset, typename DstCharset>
    STRF_HD void operator <<
        (const transcoding_test_data_maker<SrcCharset, DstCharset>& data_maker) &&
    {
        transcode_tester<SrcCharset, DstCharset> tester
            { data_maker.data, (test_transcode_err_messenger&&)messenger_ };
        tester.run();
    }

private:
    test_transcode_err_messenger messenger_;
};

} // namespace test_utils

#define TEST_UTF_TRANSCODE(SRC_CHAR_T, DST_CHAR_T)                      \
    test_utils::transcode_tester_caller(BOOST_CURRENT_FUNCTION, __FILE__, __LINE__) \
    << test_utils::transcoding_test_data_maker<strf::utf_t<SRC_CHAR_T>, strf::utf_t<DST_CHAR_T>> \
    (strf::utf<SRC_CHAR_T>, strf::utf<DST_CHAR_T>, true)

#define TEST_UTF_UNSAFE_TRANSCODE(SRC_CHAR_T, DST_CHAR_T) \
    test_utils::transcode_tester_caller(BOOST_CURRENT_FUNCTION, __FILE__, __LINE__) \
    << test_utils::transcoding_test_data_maker<strf::utf_t<SRC_CHAR_T>, strf::utf_t<DST_CHAR_T>> \
    (strf::utf<SRC_CHAR_T>, strf::utf<DST_CHAR_T>, false)

#endif // TEST_UTILS_TRANSCODING_HPP
