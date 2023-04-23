# PatchGraphNet: Medical Image Classification using Graph Neural Networks

Welcome to the PatchGraphNet repository! This project aims to classify medical images using graph neural networks (GNNs) by following a pipeline that involves clustering, graph construction, graph learning, and graph classification. The idea behind this approach is to create a graph representation of each medical image and then classify it into the appropriate medical labels specified in the dataset.

## Pipeline Overview

1. **Clustering:** We first cluster patches of the medical images to learn a higher space representation for the medical image set. This is done by grouping the patches together based on the cluster they belong to.
2. **Graph Construction:** Next, we construct a graph where each group of patches forms a node. This graph representation is essential for the subsequent learning and classification stages.
3. **Graph Learning:** In this phase, we start learning the graph structure and the relationships between nodes using graph neural networks.
4. **Graph Classification:** Finally, we classify the graph that represents each image into the medical labels specified in the dataset.

## Usage

To start using the PatchGraphNet pipeline, follow these steps:

1. Clone the repository:
Clone the repository:
```bash
git clone https://github.com/usama13o/GCNs.git
```
cd patchgraphnet
Install the required dependencies:
```bash
pip install -r requirements.txt
```
Run the main script with the following command:
```
python main.py --data_path /path/to/your/dataset
```
## MedMNIST Dataset

For this project, we use the [MedMNIST](https://medmnist.com/) dataset, which is a collection of 10 preprocessed medical image datasets. Each dataset is tailored for a specific medical imaging problem, ranging from organ segmentation to disease classification. MedMNIST is a great resource for evaluating and benchmarking medical image analysis models, and it serves as a solid foundation for our project.

