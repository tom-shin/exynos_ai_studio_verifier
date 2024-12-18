# [Tools](/tools) > [Data Preparation](/tools/data_prep) > Quantizer

Change data type for raw data.

## [FP32 to UInt8](./fp32_2_uint8.py)
Usage:
```bash
$ ./fp32_2_uint8.py --help
usage: fp32_2_uint8.py [-h] -i INPUT -o OUTPUT [-r RANGE RANGE] [-f {bin,npy} [{bin,npy} ...]]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input file path
  -o OUTPUT, --output OUTPUT
                        Output file name
  -r RANGE RANGE, --range RANGE RANGE
                        Range of input data [min, max]
  -f {bin,npy} [{bin,npy} ...], --format {bin,npy} [{bin,npy} ...]
                        Output format
```