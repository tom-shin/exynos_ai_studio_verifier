# [Tools](/tools) > Model Modifier

Modify models for certain purpose.

## ONNX
### [`onnx_mod`](./onnx_mod.py)
Removes unused input layers, `Dropout` nodes, and change OP set.

Usage:
```bash
./onnx_mod.py --help
usage: onnx_mod.py [-h] -m MODEL [-o OUTPUT] [--debug]

options:
  -h, --help           show this help message and exit
  -m, --model MODEL    Model to create data for
  -o, --output OUTPUT  Output file name
  --debug
```

### [`onnx_rm_node`](./onnx_rm_node.py)
Usage:
```bash
./onnx_rm_node.py -h
usage: onnx_rm_node.py [-h] -m MODEL [-o OUTPUT] [--nodes NODES [NODES ...]] [--layer LAYER [LAYER ...]]

options:
  -h, --help            show this help message and exit
  -m, --model MODEL     Model to create data for
  -o, --output OUTPUT   Output file name
  --nodes NODES [NODES ...]
                        List of nodes to remove
  --layer LAYER [LAYER ...]
                        List of output layers to remove
```

### [`onnx_dim_set`](./onnx_dim_set.py)