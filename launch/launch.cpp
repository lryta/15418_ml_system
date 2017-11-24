#include "include/apps/application.h"

int main() {
  auto app = new runMLP();
  app->run();
  delete app;
  return 0;
}
