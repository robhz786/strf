//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

//[QStringAppender_def

#include <QString>
#include <strf.hpp>
#include <climits>

class QStringAppender: public strf::basic_outbuf<char16_t>
{
public:

    QStringAppender(QString& str);

    explicit QStringAppender(QString& str, std::size_t size);

#if defined(STRF_NO_CXX17_COPY_ELISION)
    QStringAppender(QStringAppender&& str);
#else
    QStringAppender(QStringAppender&& str) = delete;
    QStringAppender(const QStringAppender& str) = delete;
#endif

    void recycle() override;

    std::size_t finish();

private:

    QString& str_;
    std::size_t count_ = 0;
    std::exception_ptr eptr_ = nullptr;

    constexpr static std::size_t buffer_size_ = strf::min_size_after_recycle<2>();
    char16_t buffer_[buffer_size_];
};

//]

//[QStringAppender_ctor
QStringAppender::QStringAppender(QString& str)
    : strf::basic_outbuf<char16_t>(buffer_, buffer_size_)
    , str_(str)
{
}

QStringAppender::QStringAppender(QString& str, std::size_t size)
    : strf::basic_outbuf<char16_t>(buffer_, buffer_size_)
    , str_(str)
{
    Q_ASSERT(str_.size() + size < static_cast<std::size_t>(INT_MAX));
    str_.reserve(str_.size() + static_cast<int>(size));
}


//]

//[QStringAppender_recycle
void QStringAppender::recycle()
{
    if (this->good()) {
        // Flush the content:
        std::size_t count = /*<<ouput_buffer::pointer() returns the immediate position
                              after the last character the library wrote in the buffer>>*/this->pointer() - buffer_;
        const QChar * qchar_buffer = reinterpret_cast<QChar*>(buffer_);

#if defined(__cpp_exceptions)

        try {
            str_.append(qchar_buffer, count);
            count_ += count;
        } catch(...) {
            eptr_ = std::current_exception();
            this->set_good(false);
        }

#else

        str_.append(qchar_buffer, count);
        count_ += count;

#endif // defined(__cpp_exceptions)

    }
    // Reset the buffer position:
    this->set_pointer(buffer_);

    // Not necessary to set the buffer's end since it's the same as before:
    // this->set_end(buffer_ + buffer_size_);
}
//]

//[QStringAppender_finish
std::size_t QStringAppender::finish()
{
    recycle();
    if (eptr_ != nullptr) {
        std::rethrow_exception(eptr_);
    }
    return count_;
}
//]

//[QStringAppenderFactory

class QStringAppenderFactory
{
public:

    using char_type = char16_t;
    using finish_type = std::size_t;

    QStringAppenderFactory(QString& str)
        : str_(str)
    {}

    QStringAppenderFactory(const QStringAppenderFactory& str) = default;

    template <typename ... Printers>
    finish_type write(const Printers& ... printers) const
    {
        QStringAppender ob(str_);
        strf::detail::write_args(ob, printers...);;
        return ob.finish();
    }

    template <typename ... Printers>
    finish_type sized_write( std::size_t size
                           , const Printers& ... printers ) const
    {
        str_.reserve(str_.size() + size);
        QStringAppender ob(str_);
        strf::detail::write_args(ob, printers...);;
        return ob.finish();
    }

    QStringAppender create() const
    {
        return QStringAppender{str_};
    }
    QStringAppender create(std::size_t size ) const
    {
        str_.reserve(str_.size() + size);
        return QStringAppender{str_};
    }

private:

    QString& str_;
};


//]

//[QStringAppender_append
inline auto append(QString& str)
{
    return strf::destination_no_reserve<QStringAppenderFactory> {str};
}
//]


//[QStringAppender_use
int main()
{
    QString str = "....";
    int initial_length = str.length();

    int x = 255;
    std::size_t append_count = append(str) (x, u" in hexadecimal is ", ~strf::hex(x));

    assert(str == "....255 in hexadecimal is 0xff");

    // append_count is equal to the value returned by QStringAppender::finish()
    assert(str.length() == (int)append_count + initial_length);
    (void)initial_length;
    (void)append_count;

    return 0;
}
//]
