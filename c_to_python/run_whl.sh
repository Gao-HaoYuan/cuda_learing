python3 setup.py bdist_wheel
pip install ./dist/*.whl
echo "-----------------start---------------------"
python test_whl.py
echo "------------------end-------------------"
pip uninstall hello-test -y

rm -rf build
rm -rf dist
rm -rf hello_test.egg-info
