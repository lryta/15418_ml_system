#include "shape.h"

namespace TinyML {

bool shape::operator==(const shape& b) {
  return (getDimNum() == b.getDimNum()) &&
    (getDimNum() >= 1 || std::get<0>(this->shape_)==std::get<0>(b.shape_)) &&
    (getDimNum() >= 2 || std::get<1>(this->shape_)==std::get<1>(b.shape_)) &&
    (getDimNum() >= 3 || std::get<2>(this->shape_)==std::get<2>(b.shape_));
}

}
