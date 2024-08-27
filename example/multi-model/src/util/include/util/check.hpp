#pragma once
#include <string>
class MyExceptoin {
 public:
  MyExceptoin(const std::string &info) : info_{info} {}
  MyExceptoin(std::string &&info) : info_{std::move(info)} {}
  std::string &what() { return info_; }

 private:
  std::string info_;
};

#define CHECK(bool_exp)                                                 \
  do {                                                                  \
    if (!(bool_exp)) {                                                  \
      throw MyExceptoin{std::string{__FILE__} + ":" +                   \
                        std::to_string(__LINE__) + " => Fail: " + #bool_exp}; \
    }                                                                   \
  } while (0);

#define CHECK_WITH_INFO(bool_exp, info)                                       \
  do {                                                                        \
    if (!(bool_exp)) {                                                        \
      throw MyExceptoin{std::string{__FILE__} + ":" +                         \
                        std::to_string(__LINE__) + " => Fail: " + #bool_exp + " " + \
                        info};                                                \
    }                                                                         \
  } while (0);

#define THIS_LINE (std::string(__FILE__) + ":" + std::to_string(__LINE__))

#define PRINT(text)                                                            \
  std::cout << text << "\n";

#define PRINT_THIS_LINE() \
  std::cout << (std::string(__FILE__) + ":" + std::to_string(__LINE__)) << "\n";