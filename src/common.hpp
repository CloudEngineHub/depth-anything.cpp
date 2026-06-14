#pragma once
#include <cstdio>
#define DA_LOG(...) do { std::fprintf(stderr, "[da3] " __VA_ARGS__); std::fprintf(stderr, "\n"); } while (0)
