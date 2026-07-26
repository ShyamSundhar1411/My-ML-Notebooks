<div align="center">

# My ML Notebooks

**A working notebook library: fundamentals, paper replications, and end-to-end projects.**

![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3670A0?logo=python&logoColor=white)

</div>

---

## What this is

A collection of Jupyter notebooks written while working through machine learning from the ground
up — from NumPy and pandas basics to replicating architectures from their original papers. Each
notebook is self-contained and runnable, with the reasoning written out rather than just the code.

## Highlights

### Paper replications
Implementations built from the original papers rather than from a library call:

- **[Attention Is All You Need](Paper%20Replications/Attention_Is_All_You_Need.ipynb)** — the Transformer architecture from scratch
- **[Vision Transformer](Paper%20Replications/Vision_Transformer.ipynb)** — ViT, patch embeddings through classification head

### Milestone projects
Longer, end-to-end builds:

- **Food Vision** — image classification with transfer learning, in two parts: feature extraction and fine-tuning
- **SkimLit** — NLP model for sequential sentence classification of medical abstracts
- **BitPredict** — time-series forecasting

### Foundations
- **NumPy** — introduction and exercises
- **pandas** — introduction and exercises
- **TensorFlow** — fundamentals and exercises
- **Optimizers** — gradient descent implemented and visualised

### Classification and regression
- Computer vision with TensorFlow, neural network classification
- Titanic, Spaceship Titanic, Iris, employee attrition
- Linear regression and neural network regression on the medical cost dataset

### NLP and agents
- Introduction to NLP in TensorFlow
- **LangGraph** — graph-building and agent-building exercises

## Running the notebooks

```bash
git clone https://github.com/ShyamSundhar1411/My-ML-Notebooks.git
cd My-ML-Notebooks
pip install jupyter tensorflow numpy pandas matplotlib scikit-learn
jupyter notebook
```

Most notebooks were written to run on Google Colab, so they'll open there without setup too.

## Related work

Peer-reviewed publications building on this foundation:

- **LeafNet** — weight initialisation and residual connections for groundnut disease detection · *IEEE Access, 2024*
- **GAT-GCN hybrid model** for leaf disease classification · *Frontiers in Plant Science, 2025*
- **Attention-based groundwater quality forecasting** · *Environmental Science Europe, 2025*

## License

See repository license.
