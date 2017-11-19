#include "optim.h"


SGD::SGD(Settings setting) {
  settings = setting;
}

SGD::registerParams(std::vector<Tensor*> params) {
  for (auto &param : params) {
    weights.push_back(param);
    velocity.push_back(zerosLike(param));
  }
}

SGD::update() {
  for (int i = 0; i < weights.size(); ++i) {
    v = velocity[i];
    v = settings.momentum * v + settings.lr * weights[i]->grad;
    velocity[i] = v;
    if (settings.weightDecay != 0) {
      Tensor decay = settings.weightDecay * weights[i]->data;
      weights[i]->data -= decay;
    }
    weights[i]->data -= weights[i]->grad;
  }
}

SGD::reset() {
  for (auto &i : velocity) {
    i.setZeros();
  }
}
