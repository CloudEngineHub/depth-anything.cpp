function(da_add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE depthanything)
  target_include_directories(${name} PRIVATE
      ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include
      ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/third_party)
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES
      SKIP_RETURN_CODE 77
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      ENVIRONMENT "DA_TEST_GGUF=${CMAKE_SOURCE_DIR}/models/depth-anything-base-f32.gguf;DA_TEST_BASELINE=${CMAKE_SOURCE_DIR}/dumps/reference.gguf")
endfunction()

da_add_test(test_model_loader)
da_add_test(test_backend)
