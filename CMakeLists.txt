cmake_minimum_required(VERSION 3.5)
project(point_lio_unilidar)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set build type to Debug if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

# Add compile options
add_compile_options(-pthread -fexceptions)

# Define ROOT_DIR macro
add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")
message("DROOT_DIR = ${CMAKE_CURRENT_SOURCE_DIR}/")

# Detect CPU architecture and set MP_PROC_NUM
message("Current CPU architecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)" )
  include(ProcessorCount)
  ProcessorCount(N)
  message("Processor number:  ${N}")
  if(N GREATER 5)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=4)
    message("core for MP:  3")
  elseif(N GREATER 3)
    math(EXPR PROC_NUM "${N} - 2")
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM="${PROC_NUM}")
    message("core for MP:  ${PROC_NUM}")
  else()
    add_definitions(-DMP_PROC_NUM=1)
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

# Find OpenMP
find_package(OpenMP QUIET)
if (OPENMP_FOUND)
  message(STATUS "OpenMP found")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Find required packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pcl_conversions REQUIRED)

# Include directories
include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${Python3_INCLUDE_DIRS}
)

# Message output for Eigen
message("Eigen include directory: ${EIGEN3_INCLUDE_DIRS}")

# Source files
set(SOURCES
  src/laserMapping.cpp 
  include/ikd-Tree/ikd_Tree.cpp 
  src/parameters.cpp 
  src/preprocess.cpp 
  src/Estimator.cpp
)

# Add executable
add_executable(pointlio_mapping ${SOURCES})

# Link libraries and dependencies
ament_target_dependencies(pointlio_mapping
  rclcpp
  sensor_msgs
  geometry_msgs
  nav_msgs
  std_msgs
  tf2_ros
  tf2_eigen
  Eigen3
  PCL
  pcl_conversions
)

target_link_libraries(pointlio_mapping
  ${PCL_LIBRARIES}
  ${Python3_LIBRARIES}
)

# Set include directories for the target
target_include_directories(pointlio_mapping
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${EIGEN3_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)

# Install rules
install(TARGETS pointlio_mapping
  DESTINATION lib/${PROJECT_NAME}
)

# Ament package
ament_package()
