# Hotspy: Performance Hotspot Identification with GNN-based Static Analysis

## Overview
**Hotspy** is a performance analysis tool designed to identify hotspot functions in parallel programs using Graph Neural Networks (GNN). By analyzing Control-Flow Graphs (CFGs) and Call Graphs (CGs) derived from LLVM Intermediate Representation (IR), Hotspy predicts potential hotspots **without requiring program pre-execution**, significantly reducing profiling overhead.

## Key Features
- **Static Analysis**: Eliminates the need for pre-execution by leveraging LLVM IR for CFG/CG extraction.
- **GNN-Based Model**: Combines CFG and CG features with GNN and ResNet architectures for accurate predictions.
- **Low Overhead**: Reduces profiling time by **98%** compared to traditional tools like Scalasca.
- **Multi-Language Support**: Compatible with C/C++ and Fortran programs via LLVM/Clang/Flang.

## Installation
### Dependencies
- **LLVM 11.8** (for IR and CFG/CG extraction)
- **Python 3.8+** with PyTorch 2.1.2
- **OpenMPI 5.0.1** (for parallel program support)
- Additional packages: `networkx`, `scipy`, `gensim`

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/hd9568/hotspy-tool.git
   cd hotspy-tool