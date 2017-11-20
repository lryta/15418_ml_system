#include "data.h"
#include "optim.h"

namespace MLLib {

class Solver {
 public:
  Solver(Net* net,
      Optimizer* ioptim, 
      DataIterator* itrainIter,
      DataIterator* ivalidIter,
      DataIterator* itestIter=NULL);
  void train(int epoch);
  void valid();
  void test();
  void reset();
  void setBest();

 private:
  Optimizer* optim;
  DataIterator* trainIter;
  DataIterator* validIter;
  DataIterator* testIter;
};

}
