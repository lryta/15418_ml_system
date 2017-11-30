#ifndef _TINYML_ENV_H
#define _TINYML_ENV_H

#include "weightInit.h"

namespace TinyML {

class env {
 public:
  env():weight_init_(NULL) {
    weight_init_ = new weightInit();
  }

  ~env() {
    delete weight_init_;
  }

  weightInit *getWeightInit() {
    assert(weight_init_ != NULL);
    return weight_init_;
  }

  static env* getInstance() {
    if (env_ == NULL)
      env_ = new env();
    return env_;
  }

  static void clearEnv() {
    delete env_;
  }

 private:
  weightInit *weight_init_;
  static env* env_;
};

}

#endif
