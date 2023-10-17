
set(SOURCE src/getenv.c src/getenv.cpp src/getopt.c)
add_library(my_util STATIC ${SOURCE})
target_include_directories(my_util PUBLIC include)

