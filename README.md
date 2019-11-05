# Combine Graph Attention and Graph2Vec

## Usage

Datasets available:

```
['ENZYMES', 'PROTEINS', 'PROTEINS_full', 'MUTAG', 'PTC_FM', 'NCI1', 'PTC_FR', 'FRANKENSTEIN']
```

Usage:

```
python3 launch.py ENZYMES
```

It will create a folder 'ENZYMES_weights' with:

+ the weights of the node embedder, with h5 format
+ the weights of the graph embeddings, with h5 format
+ the weights of the graph embedding, in CSV format

To test the quality of the embeddings go to https://projector.tensorflow.org/  
You need to upload 'ENZYMES\_weights/graph\_embeddings.csv' and 'ENZYMES/ENZYMES\_graph\_labels.txt'

To test the quality of the embeddings just use the SVM:

```
python3 svm.py ENZYMES
```

It will print the accuracy.
