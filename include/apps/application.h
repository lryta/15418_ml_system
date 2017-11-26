#include "include/modelTrainer.h"

namespace TinyML {

class application {
 public:
  application();
  virtual void run();
};

class runMLP : application {
 public:
  runMLP();
  virtual void run();
 private:
  modelTrainer trainer_;
};

}
