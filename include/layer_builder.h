#include<vector>

namespace MLLib {
using std::vector;

enum layerType {
  FullyConnected,
  Sigmoid,
  L2Loss,
};

class layerBuilder {
 public:
  static layerBuilder* get() {
    if (instance_ == NULL)
      instance_ = new layerBuilder();
    return instance_;
  }

  layer* makeLayer(layerBuildHelper, vector<shape>,
    vector<shape>);

 private:
  layerBuilder();
  static layerBuilder *instance_;
};

}
