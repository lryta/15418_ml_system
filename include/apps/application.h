#ifndef _TINYML_APPS_APPLICATION_H
#define _TINYML_APPS_APPLICATION_H

#include "modelTrainer.h"

namespace TinyML {

class application {
 public:
  application();
  virtual void run();
};

class runMLP : application {
 public:
  runMLP();
  ~runMLP();
  void run();
 private:
  trainerConfig *train_config_;
  modelTrainer *trainer_;

  optimizerConfig *opt_config_;
};

}

#endif
