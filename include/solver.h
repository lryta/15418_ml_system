#include "data.h"
#include "optim.h"


class Solver {
  public:
    Solver(Optimizer* ioptim, 
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
}
