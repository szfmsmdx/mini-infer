#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <vector>

namespace mtensor {

inline constexpr int kMaxDim = 8;

// 只借用数据。原 Tensor 或容器的生命周期必须长于此视图。
template <class T>
class LayoutView {
public:
    LayoutView() = default;
    LayoutView(const T* data, std::size_t size) : data_(size ? data : &empty_), size_(size) {}
    LayoutView(const std::vector<T>& values) : LayoutView(values.data(), values.size()) {}

    const T* data() const { return data_; }
    std::size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    const T* begin() const { return data_; }
    const T* end() const { return data_ + size_; }
    auto rbegin() const { return std::make_reverse_iterator(end()); }
    auto rend() const { return std::make_reverse_iterator(begin()); }
    const T& operator[](std::size_t i) const { return data_[i]; }
    const T& front() const { return *begin(); }
    const T& back() const { return *(end() - 1); }
    std::vector<T> to_vector() const { return {begin(), end()}; }

    friend bool operator==(LayoutView lhs, LayoutView rhs) {
        return std::equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
    }
    friend bool operator!=(LayoutView lhs, LayoutView rhs) { return !(lhs == rhs); }

private:
    inline static constexpr T empty_{};
    const T* data_ = &empty_;
    std::size_t size_ = 0;
};

// layoutview 存储shape和stride
using ShapeView = LayoutView<int>;
using StrideView = LayoutView<std::size_t>;

}  // namespace mtensor
