// Copyright 2023 AMD, Inc
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or Implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <string>
#include <sstream>
#include <iostream>
//#include <fstream>
#include <mutex>

namespace amd {
namespace maize {

class TracerSingleton {
public:
  static TracerSingleton& GetInstance();

  TracerSingleton();
  
  ~TracerSingleton();

  bool Enabled();

protected:
  bool logging_enabled_; 

};

class TracerEvent {
public:
  TracerEvent(const std::string &name);

  ~TracerEvent();

protected:
  std::string name_;
  unsigned int m_event_id_;
  static unsigned int event_id_;
  static std::mutex m_;
};

}
}

#define TRACE(EVENT_NAME) \
  amd::maize::TracerEvent(( EVENT_NAME ))

#define TRACE_BLOCK(EVENT_NAME) \
  amd::maize::TracerEvent _VAR(( EVENT_NAME ))
