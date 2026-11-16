cmake_minimum_required(VERSION 3.16)
project(pat_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PatImu.msg"
  "msg/PatCamera.msg"
  "msg/PatEstimate.msg"
  "msg/PatFsmCommand.msg"
  "msg/PatFsmState.msg"
  "msg/PatTruth.msg"
  DEPENDENCIES builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()