cmake_minimum_required(VERSION 3.8)
project(eevee_shared_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

# --- core library (Eigen + OpenCV + nav_msgs only, no rclcpp) ---
add_library(afs_core
  src/eevee_shared_control/distance_field.cpp
  src/eevee_shared_control/afs_model.cpp
  src/eevee_shared_control/afs_mpc.cpp)

target_include_directories(afs_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_compile_features(afs_core PUBLIC cxx_std_17)

target_link_libraries(afs_core PUBLIC
  Eigen3::Eigen
  opencv_core
  opencv_imgproc)

ament_target_dependencies(afs_core PUBLIC nav_msgs)

# --- ROS node ---
add_executable(mpc_controller_node src/mpc_controller_node.cpp)

target_include_directories(mpc_controller_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_features(mpc_controller_node PRIVATE cxx_std_17)

target_link_libraries(mpc_controller_node PRIVATE afs_core)

ament_target_dependencies(mpc_controller_node
  "rclcpp"
  "nav_msgs"
  "std_msgs"
  "sensor_msgs"
  "geometry_msgs"
  "visualization_msgs"
  "tf2_ros"
)

# --- install ---
install(TARGETS afs_core
  EXPORT export_afs_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

install(TARGETS mpc_controller_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/
  DESTINATION include)

ament_export_include_directories(include)
ament_export_targets(export_afs_core HAS_LIBRARY_TARGET)
ament_export_dependencies(nav_msgs Eigen3 OpenCV)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  # ament_add_gtest(test_distance_field test/test_distance_field.cpp)
  # target_link_libraries(test_distance_field afs_core)
endif()

ament_package()