#include "solver.h"

namespace MLLib {

Solver::Solver(Optimizer* ioptim, DataIterator itrainIter,
  DataIterator ivalidIter, DataIterator itestIter)
  :optim(ioptim), trainIter(itrainIter), validIter(ivalidIter), testIter(itestIter){
}

Solver::train(int epoch) {
  for (int i = 0; i < epoch; ++i) {
    float totalLoss = 0;
    float totalAcc = 0;
    while (trainIter.nextEnd()) {
      Tensor* X = std::get<0>(trainIter);
      Tensor* Y = std::get<1>(trainIter);
    }

  }
}

}
