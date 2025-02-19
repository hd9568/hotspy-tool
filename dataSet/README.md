# Directory Structure Explanation

This directory illustrates the organization of the dataset, where both the training set and the testing set can be constructed as follows:

├── data2json.py
├── README.md
├── spec648exchange2_data
│   ├── cfg_dot
│   ├── cg_dot
│   │   ├── exchange2.fppized.ll.callgraph.dot
│   │   └── exchange2.ll.callgraph.dot
│   └── spec648exchange2_data.txt
└── spec999randir_data
├── cfg_dot
├── cg_dot
│   ├── main.ll.callgraph.dot
│   └── specrand.ll.callgraph.dot
└── spec999randir_data.txt


### Details

- `*_data.txt` files contain function names along with their hotness information.
- `*.dot` files are graph files representing call graphs.
- `data2json.py` is a script used to convert the data into JSON format for easier processing.
  
Please refer to each subdirectory for specific datasets. The structure is designed to facilitate easy access and processing of the data according to your needs.
If you want the entire dataset, please contact huoda@buaa.edu.cn.