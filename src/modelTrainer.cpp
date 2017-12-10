#include <chrono>
#include "modelTrainer.h"

namespace TinyML {

modelTrainer::modelTrainer(trainerConfig *config)
  :weight_updater_(NULL), data_iter_(NULL), config_(*config) {
  data_iter_ = new MNISTIterator(config_.dataset_dir_, config_.batch_num_);
}

modelTrainer::~modelTrainer() {
  delete data_iter_;
  delete weight_updater_;
  delete net_;
}

void modelTrainer::setModel(ModelType type, std::vector<size_t> inter_dims) {
  if (weight_updater_ != NULL)
    throw "setModel() should be called before setOptimizer()";

  switch (type) {
    case ModelType::MLPnetType:
      {
        assert(inter_dims.size() > 0);
        auto total_dims = inter_dims;
        total_dims.push_back(data_iter_->getTargetDim());
        net_ = new MLPnet(data_iter_->getDataShape(), data_iter_->getTargetShape(), total_dims);
      }
      break;
    default:
      throw "network Type not recognized";
  }
}

void modelTrainer::setOptimizer(optimizerConfig config) {
  if (net_ == NULL)
    throw "setModel() should be called before setOptimizer()";

  weight_updater_ = new SGDOptimizer(&config);
  weight_updater_->registerParams(net_->getLayers(), net_->getParams(), net_->getParamIdToLayerIdMap());
  weight_updater_->randomizeParams();
}

void modelTrainer::train() {
  if (weight_updater_ == NULL)
    throw "setOptimizer() should be called before train()";


  for (int i = 0; i < config_.epoch_num_; ++i) {
    printf("Run iter %d\n", i);
    // Reset iterator at very beginning
    data_iter_->reset();

    labelled_num_ = 0;
    correctly_labelled_num_ = 0;

    std::chrono::duration<double> total_time;
    int batch_cnt = 0;

    while (data_iter_->hasNext()) {
      ++batch_cnt;
      auto data = data_iter_->next();
      auto input = std::get<0>(data);
      auto target = std::get<1>(data);
      auto start = std::chrono::system_clock::now();
 

      // forward
      net_->forward({input}, {target});
      correctly_labelled_num_ += net_->correctlyRecognizedDataNum();
      labelled_num_ += config_.batch_num_;
      printf("Total Loss is %f\n", net_->getLoss());

      // back prop and update weight
      net_->backward({input}, {target});
      weight_updater_->update();
      net_->printTimeStat();


      auto end = std::chrono::system_clock::now();
      total_time += end-start;
      
    }

    printf("avg ms per batch is %f\n", total_time.count()/batch_cnt);

    printf("Total Accuracy in epoch %d is %f\n",
        i + 1, (float)correctly_labelled_num_/labelled_num_);
  }
}

}
