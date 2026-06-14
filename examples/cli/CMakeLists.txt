add_executable(da3-cli main.cpp)
target_link_libraries(da3-cli PRIVATE depthanything)
target_include_directories(da3-cli PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include)
