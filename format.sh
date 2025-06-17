#!/bin/bash

find . \( -path './pytorch' -o -path './cuda-samples' -o -path './Benchmark' \) -prune -o \
  -regex '.*\.\(cpp\|hpp\|cu\|c\|h\)' -exec clang-format -style=file -i {} \;
