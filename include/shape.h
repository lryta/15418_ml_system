#include<tuple>
#include<assert>

namespace MLLib {

// shape has at most three dim at present
class Shape {
 public:
  Shape(size_t d0, size_t d1, size_t d2):
    shape_(make_tuple(d0, d1, d2)), total_(d0 * d1 * d2) {
      ndim_ = 3;
  }

  Shape(size_t d0, size_t d1):
    shape_(make_tuple(d0, d1, 1)), total_(d0 * d1) {
      ndim = 2;
  }

  Shape(size_t d0):
    shape_(make_tuple(d0, 1, 1)), total_(d0) {
      ndim = 1;
  }

  size_t getDim(int k) {
    assert(k <= 3);
    return std::get<k-1>(shape_);
  }

  size_t getTotal() {
    return total_;
  }

  size_t ndim() {
    return ndim_;
  }
  
 private:
  std::tuple<size_t, size_t, size_t> shape_;  
  size_t ndim_;
  size_t total_;
}

}
