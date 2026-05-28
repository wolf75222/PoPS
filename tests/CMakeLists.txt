# Tests sans dependance externe pour ce premier socle (executable + CTest).
# Catch2 sera cable quand la surface de test le justifiera.

add_executable(test_diocotron_model test_diocotron_model.cpp)
target_link_libraries(test_diocotron_model PRIVATE adc::adc)
add_test(NAME test_diocotron_model COMMAND test_diocotron_model)
