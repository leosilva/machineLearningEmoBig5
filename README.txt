Running Machine Learning Algorithms
-----------------------------------
Runs machine learning algorithms on prepared datasets to evaluate their performance.

Key Operations:
1. Configuring output paths and dataset weighting in the 'utils.py', 'init_config.py', and 'main.py' scripts.
2. Running the algorithms with command-line instructions, allowing flexibility in model selection.

Examples:
“python3 main.py -t True --models knn,svc,random-forest,decision-tree,mlp-classifier,logistic-regression -b False”
“python3 main.py -t True --models knn -b False”