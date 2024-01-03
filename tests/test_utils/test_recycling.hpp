#ifndef TEST_RECYCLING_HPP
#define TEST_RECYCLING_HPP

#include "./array_destination_with_sub_initial_space.hpp"

namespace test_utils {

template <typename CharT, typename... Printers>
STRF_HD void test_printers_recycling
    ( strf::destination<char>& failure_notifier
    , std::size_t expected_size
    , strf::detail::simple_string_view<CharT> expected_output
    , const Printers&... printers )
{
    const auto print_fail = to(failure_notifier);

    constexpr std::size_t buff_size = 500;
    if (expected_output.size() + strf::min_destination_buffer_size > buff_size) {
        print_fail("\nSorry, tester buffer too small for this test case\n"
                   "expected output is \"", strf::transcode(expected_output), "\")");
        return;
    }
    if (expected_size != expected_output.size()) {
        print_fail("\nCalculated size is different than expected");
    }

    CharT buff[buff_size];
    test_utils::array_destination_with_sub_initial_space<CharT> dst(buff, buff_size);

    for (std::size_t space=0; space <= expected_size; ++space) {
        dst.reset_with_initial_space(space);
        strf::detail::call_printers(dst, printers...);
        auto result = dst.finish();
        if (result != expected_output) {
            print_fail("\noutput different from expected");
        }
        if (expected_size > space && dst.recycle_calls_count() == 0) {
            print_fail("\nrecycle should have been called, but it hasn't");
        }
        if (expected_size <= space && dst.recycle_calls_count() > 0) {
            print_fail("\nrecycle should not have been called, but it has");
        }
        if (!dst.good()) {
            print_fail("\nrecycle was called more than once. This shouln'd have happened here");
        }
    }
}

template <typename CharT, typename... Printers>
inline STRF_HD void test_printers_recycling
    ( strf::destination<char>& failure_notifier
    , const strf::premeasurements<strf::size_presence::yes, strf::width_presence::no>& pre
    , strf::detail::simple_string_view<CharT> expected_output
    , const Printers&... printers )
{
    test_printers_recycling(failure_notifier, pre.accumulated_size(), expected_output, printers...);
}

template <typename CharT, typename FPack, typename... Printables>
STRF_HD void test_printables_recycling
    ( strf::destination<char>& failure_notifier
    , strf::detail::simple_string_view<CharT> expected_output
    , const FPack& fp
    , Printables&&... printables)
{
    // todo
    using pre_t = strf::premeasurements<strf::size_presence::yes, strf::width_presence::no>;
    pre_t pre;
    test_printers_recycling
        ( failure_notifier
        , pre
        , expected_output
        , strf::printer_type<CharT, pre_t, FPack, Printables>
            ( strf::make_printer<CharT>
                ( &pre, fp, (Printables&&)printables ) ) ...);
}

template <typename FpesList, typename PrintablesList>
struct deep_arg_tester;

template <typename... Fpes, typename... Printables>
struct deep_arg_tester
    < strf::detail::mp_type_list<Fpes...>
    , strf::detail::mp_type_list<Printables...> >
{
    template <typename CharT>
    static STRF_HD void test
        ( strf::destination<char>& failure_notifier
        , strf::detail::simple_string_view<CharT> expected
        , Fpes... fpes
        , Printables... printables )
    {
        test_printables_recycling<CharT>
            ( failure_notifier, expected
            , strf::pack((Fpes&&)fpes...), (Printables&&)printables... );
    }
};

template <typename CharT>
class full_recycler_tester
{
public:
    STRF_HD full_recycler_tester
        ( const char* funcname
        , const char* srcfile
        , int srcline
        , strf::detail::simple_string_view<CharT> expected )
        : notifier_(funcname, srcfile, srcline)
        , expected_(expected)
    {
    }

    template <typename... Args>
    STRF_HD void operator()(Args&&... args) &&
    {
        using separated_arg_types = strf::detail::args_without_tr::separate_args<Args...>;
        using fpes_type_list = typename separated_arg_types::fpes;
        using printables_type_list = typename separated_arg_types::printables;
        using impl = deep_arg_tester<fpes_type_list, printables_type_list>;

        impl::test(notifier_, expected_, (Args&&) args...);
    }

private:
    test_utils::test_failure_notifier notifier_;
    strf::detail::simple_string_view<CharT> expected_;
};


template <typename CharT>
constexpr STRF_HD CharT char_type_of(const CharT*) { return CharT{}; }

template <typename CharT>
constexpr STRF_HD CharT char_type_of(strf::detail::simple_string_view<CharT>)
{ return CharT{}; }

#if defined(STRF_HAS_STD_STRING_DECLARATION)

template <typename CharT, typename Traits, typename Allocator>
constexpr STRF_HD CharT char_type_of(std::basic_string<CharT, Traits, Allocator>)
{ return CharT{}; }

#endif // defined(STRF_HAS_STD_STRING_DECLARATION)

#if defined(STRF_HAS_STD_STRING_VIEW)

template <typename CharT, typename Traits>
constexpr STRF_HD CharT char_type_of(std::basic_string_view<CharT, Traits>)
{ return CharT{}; }

#endif // defined(STRF_HAS_STD_STRING_VIEW)

} // namespace test_utils


#define TEST_RECYCLING(EXPECTED)                                 \
    test_utils::full_recycler_tester                             \
    < decltype(test_utils::char_type_of(EXPECTED)) >             \
    ( BOOST_CURRENT_FUNCTION, __FILE__, __LINE__, (EXPECTED) )


#endif // TEST_RECYCLING_HPP
