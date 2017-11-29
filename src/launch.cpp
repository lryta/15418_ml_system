#include <iostream>

#include "apps/application.h"

int main() {
  try {
    auto app = new TinyML::runMLP();
    app->run();
    delete app;
  } catch (const char* str) {
    std::cout << str << "\n";
  }
  return 0;
}
