cmake_minimum_required(VERSION 2.8)
project( Hello_Nanoflann )

#find_package(nanoflann)
find_package(Eigen3)

add_executable(HelloWorld src/main.cpp)
target_include_directories(HelloWorld PRIVATE ${EIGEN3_INCLUDE_DIR})
#target_link_libraries(HelloWorld nanoflann::nanoflann)

# READ MORE:
#   https://github.com/jlblancoc/nanoflann#11-obtaining-the-code
#   https://github.com/jlblancoc/nanoflann#17-use-in-your-project-via-cmake