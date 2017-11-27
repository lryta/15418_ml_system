#include<tuple>
#include<cassert>



namespace TinyML {

// shape has at most three dim at present
class shape {
 public:
  shape(size_t d0, size_t d1, size_t d2):
    shape_(std::make_tuple(d0, d1, d2)), ndim_(3), total_(d0 * d1 * d2)
  {}

  shape(size_t d0, size_t d1):
    shape_(std::make_tuple(d0, d1, 0)), ndim_(2), total_(d0 * d1)
  {}

  shape(size_t d0):
    shape_(std::make_tuple(d0, 0, 0)), ndim_(1), total_(d0)
  {}

  shape(const shape& s):
    shape_(s.getShape()), ndim_(s.getDimNum()), total_(s.getTotal())
  {}

  size_t getDim(size_t k) {
    assert(k >= 1 && k <= ndim_);
    switch (k) {
      case 1:
        return std::get<0>(shape_);
      case 2:
        return std::get<1>(shape_);
      case 3:
        return std::get<2>(shape_);
      default:
        throw "Show not hit here!!";
    }
    return -1;
  }

  size_t getTotal() const {
    return total_;
  }

  std::tuple<size_t, size_t, size_t> getShape() const {
    return shape_;
  }

  size_t getDimNum() const {
    return ndim_;
  }

  bool operator==(const shape& b);
  
 private:
  std::tuple<size_t, size_t, size_t> shape_;  
  size_t ndim_;
  size_t total_;
};

}
