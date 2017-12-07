#ifndef _TINYML_ENV_H
#define _TINYML_ENV_H

#include <cassert>
#include "weightInit.h"

namespace TinyML {

class env {
 public:
  env(bool use_ispc = false):weight_init_(NULL), use_ispc_(use_ispc) {
#ifndef COMPILE_ISPC
    assert(!use_ispc);
#endif // ifndef COMPILE_ISPC

    weight_init_ = new weightInit();
  }

  ~env() {
    delete weight_init_;
  }

  weightInit *getWeightInit() {
    assert(weight_init_ != NULL);
    return weight_init_;
  }

  bool useISPC() {
    return use_ispc_;
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
  bool use_ispc_;
};

}

#endif
