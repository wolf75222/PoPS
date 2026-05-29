# Module Python pybind11 de la facade compilee. Le .so est nomme `adc` et lie
# adc::solver (libadc). Build : cmake -B build -DADC_BUILD_PYTHON=ON puis
# cmake --build build --target adc_py.
pybind11_add_module(adc_py bindings.cpp)
target_link_libraries(adc_py PRIVATE adc::solver)
target_compile_features(adc_py PRIVATE cxx_std_23)
set_target_properties(adc_py PROPERTIES OUTPUT_NAME adc)

# Test : importe le module et verifie les invariants (masse, forme du champ).
if(ADC_BUILD_TESTS)
  add_test(NAME test_bindings
           COMMAND ${Python_EXECUTABLE}
                   ${CMAKE_CURRENT_SOURCE_DIR}/test_bindings.py)
  set_tests_properties(test_bindings PROPERTIES
    ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:adc_py>")
endif()
