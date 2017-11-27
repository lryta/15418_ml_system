#include "shape.h"

namespace TinyML {

bool shape::operator==(const shape& b) {
  return (getDimNum() == b.getDimNum()) &&
    (getDimNum() >= 1 || std::get<0>(*this)==std::get<0>(b)) &&
    (getDimNum() >= 2 || std::get<1>(*this)==std::get<1>(b)) &&
    (getDimNum() >= 3 || std::get<2>(*this)==std::get<2>(b));
}

}
