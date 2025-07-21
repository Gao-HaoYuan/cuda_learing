#!/bin/bash

python setup.py clean
find . -name "*.so" -delete

# python setup.py install # generate the file: xxxx.egg
python setup.py build_ext --inplace

rm -rf build/ *.egg-info/ dist/ lib/
find . -name "*.o" -delete
find . -name "*.a" -delete