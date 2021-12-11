//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <QString>
#include <strf.hpp>
#include <climits>
#include <algorithm>

class QStringCreator: public strf::destination<char16_t>
{
public:

    QStringCreator()
        : strf::destination<char16_t>(buffer_, buffer_size_)
    {
    }

    explicit QStringCreator(strf::tag<>)
        : QStringCreator()
    {
    }

    QStringCreator(QStringCreator&&) = delete;
    QStringCreator(const QStringCreator&) = delete;

    explicit QStringCreator(std::size_t size)
        : strf::destination<char16_t>(buffer_, buffer_size_)
    {
        Q_ASSERT(size < static_cast<std::size_t>(INT_MAX));
        str_.reserve(static_cast<int>(size));
    }

    void recycle_buffer() override;

    QString finish();

private:

    QString str_;
    constexpr static std::size_t buffer_size_ = strf::destination_space_after_flush;
    char16_t buffer_[buffer_size_];
};

void QStringCreator::recycle_buffer()
{
    std::size_t count = this->buffer_ptr() - buffer_;
    this->set_buffer_ptr(buffer_);
    if (this->good()) {
        this->set_good(false);
        QChar qchar_buffer[buffer_size_];
        std::copy_n(buffer_, count, qchar_buffer);
        str_.append(qchar_buffer, count);
        this->set_good(true);
    }
}

QString QStringCreator::finish()
{
    flush();
    this->set_good(false);
    return std::move(str_);
}

class QStringCreatorFactory
{
public:
    using char_type = char16_t;
    using finish_type = QString;
    using destination_type = QStringCreator;
    using sized_destination_type = QStringCreator;

    strf::tag<> create() const
    {
        return strf::tag<>{};
    }
    std::size_t create(std::size_t size) const
    {
        return size;
    }
};

constexpr strf::printer_no_reserve<QStringCreatorFactory> toQString{};

int main()
{
    int x = 255;
    QString str = toQString(x, u" in hexadecimal is ", *strf::hex(x));
    assert(str == "255 in hexadecimal is 0xff");

    return 0;
}
