#ifndef _TINYML_WEIGHTINIT_H
#define _TINYML_WEIGHTINIT_H

#include <random>

#include "tensor.h"

namespace TinyML {

class weightInit {
 public:
  void uniformInit(tensor *t, float low_thresh, float high_thresh);

  void normalDistInit(tensor *t, float mean, float stdv);
};

}

#endif
