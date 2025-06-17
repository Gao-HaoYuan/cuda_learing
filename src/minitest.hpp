#ifndef MINI_TEST_HPP
#define MINI_TEST_HPP

#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <sstream>

namespace minitest {

struct TestCase {
    std::string suite;
    std::string name;
    std::function<void()> func;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

inline void register_test(const std::string& suite, const std::string& name, std::function<void()> func) {
    registry().push_back({suite, name, func});
}

inline void run_all(const std::string& filter = "") {
    int total = 0, passed = 0;
    for (const auto& t : registry()) {
        std::string full_name = t.suite + "." + t.name;
        if (!filter.empty()) {
            // 支持 suite 或 suite.name 过滤
            if (filter != t.suite && filter != full_name)
                continue;
        }

        ++total;
        try {
            t.func();
            std::cout << "[  PASSED  ] " << full_name << std::endl;
            ++passed;
        } catch (const std::exception& e) {
            std::cout << "[  FAILED  ] " << full_name << " - " << e.what() << std::endl;
        }
    }

    std::cout << "\n========== SUMMARY ==========" << std::endl;
    std::cout << "Total: " << total << ", Passed: " << passed << ", Failed: " << (total - passed) << std::endl;
}

#define TEST(suite, name) \
    void suite##_##name(); \
    struct suite##_##name##_registrar { \
        suite##_##name##_registrar() { \
            ::minitest::register_test(#suite, #name, suite##_##name); \
        } \
    } suite##_##name##_registrar_instance; \
    void suite##_##name()

#define EXPECT_TRUE(cond) \
    if (!(cond)) throw std::runtime_error("EXPECT_TRUE failed: " #cond);

#define EXPECT_FALSE(cond) \
    if (cond) throw std::runtime_error("EXPECT_FALSE failed: " #cond);

#define EXPECT_EQ(a, b) \
    if (!((a) == (b))) { \
        std::ostringstream oss; \
        oss << "EXPECT_EQ failed: " << #a << " != " << #b << " (" << (a) << " vs " << (b) << ")"; \
        throw std::runtime_error(oss.str()); \
    }

#define EXPECT_NE(a, b) \
    if (!((a) != (b))) { \
        std::ostringstream oss; \
        oss << "EXPECT_NE failed: " << #a << " == " << #b << " (" << (a) << " vs " << (b) << ")"; \
        throw std::runtime_error(oss.str()); \
    }
} // namespace minitest

#endif // MINI_TEST_HPP
