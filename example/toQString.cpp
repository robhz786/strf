//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <QString>
#include <strf.hpp>
#include <climits>

class QStringCreator: public strf::basic_outbuf<char16_t>
{
public:

    QStringCreator()
        : strf::basic_outbuf<char16_t>(buffer_, buffer_size_)
    {
    }

#if defined(STRF_NO_CXX17_COPY_ELISION)
    QStringCreator(QStringCreator&& str);
#else
    QStringCreator(QStringCreator&&) = delete;
    QStringCreator(const QStringCreator&) = delete;
#endif

    explicit QStringCreator(std::size_t size)
        : strf::basic_outbuf<char16_t>(buffer_, buffer_size_)
    {
        Q_ASSERT(size < static_cast<std::size_t>(INT_MAX));
        str_.reserve(static_cast<int>(size));
    }

    void recycle() override;

    QString finish();

private:

    QString str_;
    std::exception_ptr eptr_ = nullptr;
    constexpr static std::size_t buffer_size_ = strf::min_size_after_recycle<2>();
    char16_t buffer_[buffer_size_];
};

void QStringCreator::recycle()
{
    if (this->good()) {
        const QChar * qchar_buffer = reinterpret_cast<QChar*>(buffer_);
        std::size_t count = this->pointer() - buffer_;

#if defined(__cpp_exceptions)

        try {
            str_.append(qchar_buffer, count);
        }
        catch(...) {
            eptr_ = std::current_exception();
            this->set_good(false);
        }
#else

        str_.append(qchar_buffer, count);

#endif // defined(__cpp_exceptions)

    }
    this->set_pointer(buffer_);
}

QString QStringCreator::finish()
{
    recycle();
    this->set_good(false);
    if (eptr_ != nullptr) {
        std::rethrow_exception(eptr_);
    }
    return std::move(str_);
}

class QStringCreatorFactory
{
public:
    using char_type = char16_t;
    using finish_type = QString;

    QStringCreator create() const
    {
        return QStringCreator();
    }
    QStringCreator create(std::size_t size) const
    {
        return QStringCreator(size);
    }
};

constexpr strf::destination_no_reserve<QStringCreatorFactory> toQString{};

int main()
{
    int x = 255;
    QString str = toQString(x, u" in hexadecimal is ", ~strf::hex(x));
    assert(str == "255 in hexadecimal is 0xff");

    return 0;
}
