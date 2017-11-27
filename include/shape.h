#include<tuple>
#include<assert>

namespace TinyML {

// shape has at most three dim at present
class shape {
 public:
  shape(size_t d0, size_t d1, size_t d2):
    shape_(make_tuple(d0, d1, d2)), ndim_(3), total_(d0 * d1 * d2)
  {}

  shape(size_t d0, size_t d1):
    shape_(make_tuple(d0, d1, 0)), ndim_(2), total_(d0 * d1)
  {}

  shape(size_t d0):
    shape_(make_tuple(d0, 0, 0)), ndim_(1), total_(d0)
  {}

  shape(const shape& s):
    shape_(s.getShape()), ndim_(s.getDimNum()), total_(s.getTotal())
  {}

  size_t getDim(int k) {
    assert(k >= 1 && k <= ndim_);
    switch (k) {
      case 1:
        return std::get<0>(shape_);
      case 2:
        return std::get<1>(shape_);
      case 3:
        return std::get<2>(shape_);
      default:
        throw "Show not hit here!!"
    }
    return -1;
  }

  size_t getTotal() {
    return total_;
  }

  std::tuple<size_t, size_t, size_t> getShape() {
    return shape_;
  }

  size_t getDimNum() {
    return ndim_;
  }

  bool operator==(const shape& a, const shape& b) {
    return (a.getDimNum() == b.getDimNum()) &&
      (a.getDimNum() >= 1 || std::get<0>(a)==std::get<0>(b)) &&
      (a.getDimNum() >= 2 || std::get<1>(a)==std::get<1>(b)) &&
      (a.getDimNum() >= 3 || std::get<2>(a)==std::get<2>(b));
  }
  
 private:
  std::tuple<size_t, size_t, size_t> shape_;  
  size_t ndim_;
  size_t total_;
}

}
