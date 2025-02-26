# cuda-ml
Currently only a ridge regression solver.

## Building
```bash
mkdir build && cd build
cmake ..
cmake --build .
```

This currently builds two example programs:
* examples/online_regression/device_online_regression
* examples/online_regression/host_online_regression
