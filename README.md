# Combine Skip Gram and Convolutional Neural Networks

Datasets available:

```
['ENZYMES', 'PROTEINS', 'PROTEINS_full', 'MUTAG',
 'PTC_FM', 'NCI1', 'PTC_FR', 'DD',
 'Letter-high', 'Letter-med', 'Letter-low',
 'REDDIT_BINARY', 'COLLAB', 'MCF-7', 'MCF-7H']
```

Note: 'ENZYMES' contains 32 unconnected graphs.

## Usage of the embedding method

Usage:

```
python3 launch_embeddings.py --task=ENZYMES
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

This algorithm aims to produce graph embeddings with the use convolutional networks (GCN or GAT) to extract vocabulary.  
Then, using Skip Gram with graph embeddings as context embeddings, we generate embeddings for the graph that contain information on the vocabulary inside.  

## Usage of classification method

Instead of training raw embeddings, the embeddings are now function of the node features, and the fixed size representation is obtained via coarsening of the initial graph. Kron reduction is used thanks to its capacity to preserve the spectral properties of the graph.

```
python3 launch_classification.py --task=ENZYMES
```

# Thanks

The `loukas_coarsening` folder is copied from https://github.com/loukasa/graph-coarsening with only slight modifications.  
My work is redistributed under the same license.    
