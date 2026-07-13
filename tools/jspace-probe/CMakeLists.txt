set(TARGET llama-jspace-probe)
add_executable(${TARGET} jspace-probe.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

if(LLAMA_BUILD_TESTS)
    add_test(NAME test-jspace-probe-self-test COMMAND $<TARGET_FILE:${TARGET}> --self-test)
    set_tests_properties(test-jspace-probe-self-test PROPERTIES PASS_REGULAR_EXPRESSION "\"ok\": true")
endif()
