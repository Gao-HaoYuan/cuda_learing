#include "minitest.hpp"
using namespace minitest;

TEST(Math, Add) {
    EXPECT_EQ(1 + 1, 2);
}

TEST(Math, Subtract) {
    EXPECT_EQ(5 - 3, 2);
}

TEST(Strings, Hello) {
    EXPECT_EQ(std::string("he") + "llo", "hello");
}

int main(int argc, char **argv) {
    std::string filter;
    if (argc > 1 && std::string(argv[1]) == "--filter" && argc > 2) {
        filter = argv[2];
    }
    run_all(filter);
    return 0;
}
