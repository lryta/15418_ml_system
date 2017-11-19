#include<tuple>
#include<assert>

namespace MLLib {

// shape has at most three dim at present
class shape {
 public:
  shape(size_t d0, size_t d1, size_t d2):
    shape_(make_tuple(d0, d1, d2)), total_(d0 * d1 * d2) {
  }

  size_t getDim(int k) {
    assert(k <= 3);
    return std::get<k-1>(shape_);
  }

  size_t getTotal() {
    return total_;
  }
  
 private:
  std::tuple<size_t, size_t, size_t> shape_;  
  size_t total_;
}

}
