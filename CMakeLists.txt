cmake_minimum_required(VERSION 3.8)
project(manipulator_skill_acquisition)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(manipulator REQUIRED)
find_package(mplibrary REQUIRED)
find_package(mplearn REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python menus
install(DIRECTORY
  ${PROJECT_NAME}/SetupMenu
  ${PROJECT_NAME}/EvalnRunMenu
  DESTINATION lib/${PROJECT_NAME}
  PATTERN "*.py"
  PATTERN "__pycache__" EXCLUDE
)

install(DIRECTORY
  manipulator_skill_acquisition/rl
  DESTINATION lib/${PROJECT_NAME}
  PATTERN "*.py"
  PATTERN "__pycache__" EXCLUDE
)

# Install resource files
install(DIRECTORY
  resources config_rl config_dmp
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package() 