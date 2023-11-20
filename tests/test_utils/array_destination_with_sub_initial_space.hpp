#ifndef ARRAY_DESTINATION_WITH_SUB_INITIAL_SPACE_HPP
#define ARRAY_DESTINATION_WITH_SUB_INITIAL_SPACE_HPP

#include "../test_utils.hpp"

namespace test_utils {

template <typename CharT>
class array_destination_with_sub_initial_space: public strf::destination<CharT>
{
public:
    STRF_HD array_destination_with_sub_initial_space(CharT* dst, std::size_t size)
        : strf::destination<CharT>(dst, size)
        , dst_(dst)
        , dst_end_(dst + size)
    {
    }

    STRF_HD void reset_with_initial_space(std::size_t space)
    {
        STRF_ASSERT(dst_ + space <= dst_end_);
        this->set_buffer_ptr(dst_);
        this->set_buffer_end(dst_ + space);
        this->set_good(true);
        recycle_count_ = 0;
        ptr_before_2nd_recycle_ = nullptr;
    }

    STRF_HD void recycle()
    {
        ++recycle_count_;
        if (recycle_count_ == 1) {
            this->set_buffer_end(dst_end_);
        } else {
            if (recycle_count_ == 2) {
                ptr_before_2nd_recycle_ = this->buffer_ptr();
            }
            this->set_good(false);
            this->set_buffer_ptr(strf::garbage_buff<CharT>());
            this->set_buffer_end(strf::garbage_buff_end<CharT>());
        }
    }
    STRF_HD int recycle_calls_count() const
    {
        return recycle_count_;
    }
    STRF_HD strf::detail::simple_string_view<CharT> finish()
    {
        return {dst_, recycle_count_ <= 1 ? this->buffer_ptr() : ptr_before_2nd_recycle_};
    }

private:

    int recycle_count_=0;
    CharT* dst_=nullptr;
    CharT* dst_end_=nullptr;
    CharT* ptr_before_2nd_recycle_=nullptr;
};

} // namespace test_utils

#endif // ARRAY_DESTINATION_WITH_SUB_INITIAL_SPACE_HPP

