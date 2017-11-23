
namespace MLLib {

class appConfig {
 public:
  appConfig(int e, int b): epoch_(e), batch_(b) {}
  int epoch_num_, batch_num_;
};

class application {
 public:
  application();
  virtual void run();
};

}
