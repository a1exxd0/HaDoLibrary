#include <gtest/gtest.h>
#include <HaDo/base/EndLayer.hpp>

TEST(empty, emp) {
    int a = 0, b = 0;
    ASSERT_EQ(a, b);
}

int main(int argc, char **argv) 
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}