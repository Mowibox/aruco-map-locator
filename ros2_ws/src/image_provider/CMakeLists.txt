cmake_minimum_required(VERSION 3.8)
project(image_provider)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(yaml-cpp REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(std_msgs REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

add_executable(image_publisher src/image_publisher.cpp)
add_executable(image_subscriber src/image_subscriber.cpp)

ament_target_dependencies(image_publisher rclcpp sensor_msgs OpenCV std_msgs)
ament_target_dependencies(image_subscriber rclcpp sensor_msgs OpenCV yaml-cpp)

target_link_libraries(image_subscriber ${YAML_CPP_LIBRARIES})

install(TARGETS image_subscriber DESTINATION lib/${PROJECT_NAME})
install(TARGETS image_publisher DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
