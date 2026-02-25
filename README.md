# Attending to the Core: Core-Task Attention for Recommendation
## Setup the Environment

We recommend using Python 3.8+.Run the following commands to install dependencies:

```
pip install -r requirements.txt
```
Our approach is based on the DeepCTR framework. Run
   ```
   git clone https://github.com/shenweichen/DeepCTR.git
   ```

to obtain the DeepCTR source code, and organize the code as follows:

```
DeepCTR/
├── deepctr/
│   └── models/
│       └── multitask/
│           └── coreatt_kuaisar.py
├── data/
│   └── KuaiSar/
├── coreatt_kuaisar.py
└── README.md
```


## Run Example

Set the environment variables by running:

```
export PATH="/opt/conda/bin:$PATH" 
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

Then, run training and testing with:

```
python train_and_test_kuaisar.py output.csv
```

