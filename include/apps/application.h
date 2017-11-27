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
  trainerConfig *trainer_conf_;
  modelTrainer *trainer_;

  optimizerConfig *opt_conf_;
};

}
