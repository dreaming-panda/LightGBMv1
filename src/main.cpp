/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <LightGBM/application.h>

#include <iostream>
#include <string>

#include "network/linkers.h"

int main(int /*argc*/, char** /*argv*/) {
  bool success = false;
  try {
    std::string config = std::string("config=src/train.conf");
    char* config_ptr = new char[config.size() + 1];
    for (size_t i = 0; i < config.size(); ++i) {
      config_ptr[i] = config[i];
    }
    config_ptr[config.size()] = '\0';
    LightGBM::Application app(2, &config_ptr - 1);
    app.Run();

#ifdef USE_MPI
    LightGBM::Linkers::MpiFinalizeIfIsParallel();
#endif

    success = true;
  }
  catch (const std::exception& ex) {
    std::cerr << "Met Exceptions:" << std::endl;
    std::cerr << ex.what() << std::endl;
  }
  catch (const std::string& ex) {
    std::cerr << "Met Exceptions:" << std::endl;
    std::cerr << ex << std::endl;
  }
  catch (...) {
    std::cerr << "Unknown Exceptions" << std::endl;
  }

  if (!success) {
#ifdef USE_MPI
    LightGBM::Linkers::MpiAbortIfIsParallel();
#endif

    exit(-1);
  }
}
