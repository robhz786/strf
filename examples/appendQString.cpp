//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <QString>
#include <strf.hpp>
#include <climits>

class QStringAppender: public strf::basic_outbuff<char16_t>
{
public:

    QStringAppender(QString& str);

    explicit QStringAppender(QString& str, std::size_t size);

    QStringAppender(QStringAppender&& str) = delete;
    QStringAppender(const QStringAppender& str) = delete;

    void recycle() override;

    std::size_t finish();

private:

    QString& str_;
    std::size_t count_ = 0;
    constexpr static std::size_t buffer_size_ = strf::min_space_after_recycle<char16_t>();
    char16_t buffer_[buffer_size_];
};

QStringAppender::QStringAppender(QString& str)
    : strf::basic_outbuff<char16_t>(buffer_, buffer_size_)
    , str_(str)
{
}

QStringAppender::QStringAppender(QString& str, std::size_t size)
    : strf::basic_outbuff<char16_t>(buffer_, buffer_size_)
    , str_(str)
{
    Q_ASSERT(str_.size() + size < static_cast<std::size_t>(INT_MAX));
    str_.reserve(str_.size() + static_cast<int>(size));
}

void QStringAppender::recycle()
{
    std::size_t count = this->pointer() - buffer_;
    this->set_pointer(buffer_);
    if (this->good()) {
        this->set_good(false);
        QChar qchar_buffer[buffer_size_];
        std::copy_n(buffer_, count, qchar_buffer);
        str_.append(qchar_buffer, count);
        count_ += count;
        this->set_good(true);
    }
}

std::size_t QStringAppender::finish()
{
    recycle();
    return count_;
}

class QStringAppenderFactory
{
public:

    using char_type = char16_t;
    using finish_type = std::size_t;
    using outbuff_type = QStringAppender;
    using sized_outbuff_type = QStringAppender;

    QStringAppenderFactory(QString& str)
        : str_(str)
    {}

    QStringAppenderFactory(const QStringAppenderFactory& str) = default;

    QString& create() const
    {
        return str_;
    }
    QString& create(std::size_t size ) const
    {
        str_.reserve(str_.size() + size);
        return str_;
    }

private:

    QString& str_;
};

inline auto append(QString& str)
{
    return strf::destination_no_reserve<QStringAppenderFactory> {str};
}

int main()
{
    QString str = "....";
    int initial_length = str.length();

    int x = 255;
    std::size_t append_count = append(str) (x, u" in hexadecimal is ", *strf::hex(x));

    assert(str == "....255 in hexadecimal is 0xff");

    // append_count is equal to the value returned by QStringAppender::finish()
    assert(str.length() == (int)append_count + initial_length);
    (void)initial_length;
    (void)append_count;

    return 0;
}
