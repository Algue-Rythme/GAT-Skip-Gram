# Combine Skip Gram and Convolutional Neural Networks

## Requirements

+ Python 3.6
+ Tensorflow 2
+ Pygsp
+ Scipy
+ Scikit-learn

## Dataset

Datasets available:

```
['ENZYMES', 'PROTEINS', 'PROTEINS_full', 'MUTAG',
 'PTC_FM', 'NCI1', 'PTC_FR', 'DD',
 'DLA', 'IMDB-BINARY', 'MNIST_test']
```

Note: 'ENZYMES' contains 32 unconnected graphs.

## Usage of the embedding method

Usage:

```
python3 hierarchical_skip_gram.py --task=PROTEINS --loss_type=negative_sampling --batch_size=32 --max_depth=3 --num_features=128 --num_epochs=30 --gnn_type=krylov-4 --num_tests=10 --device=1
```

It will:

+ train over PROTEINS dataset
+ train the model over 30 epochs with batches of size 32
+ the dimension of the embedding will be equal to 128
+ it will use GPU 1
+ the graph will be coarsened to maximum depth 3
+ the vocabulary will be extracted with Truncated Krylov layer, of depth 4
+ using negative sampling loss
+ repeat this procedure 10 times in order to generate statistics to assess the average performance

It will create a folder 'PROTEINS_weights' with:

+ the weights of the node embedder, with h5 format
+ the weights of the graph embeddings, with h5 format
+ the weights of the graph embedding, in CSV format

To test the quality of the embeddings go to https://projector.tensorflow.org/  
You need to upload 'PROTEINS\_weights/graph\_embeddings.csv' and 'PROTEINS/PROTEINS\_graph\_labels.txt'

To test the quality of the embeddings just use the SVM:

```
python3 baselines.py --task=PROTEINS
```

It will print the accuracy.

This algorithm aims to produce graph embeddings with the use convolutional networks to extract vocabulary.  
Then, using Skip Gram with graph embeddings as context embeddings, we generate embeddings for the graph that contain information on the vocabulary inside.  
The nodes are pooled with Loukas method.

# Thanks

The `loukas_coarsening` folder is copied from https://github.com/loukasa/graph-coarsening with only slight modifications.  
My work is redistributed under the same license.    
