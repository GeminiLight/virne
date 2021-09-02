# Easy SFC

A python framework for Service Function Chain (SFC) deployment, which is capable of implementing relevant algorithms easily.

The characteristics are as followed:

- The first library for SFC deployment based on Python.
- Flexible to adapt to customized algorithms (heuristic-based and RL-based).
- Convenient to call existing methods (matrix operations and graph algorithms).

## File Structure

```
easy-sfc
│
├─algo
│
├─data
│
├─dataset
│
├─records
│
├─config.py
└─main.py
```

## Document

The structure and implements of this framework are still optimized steadily. We will construct the first version of the document as soon as possible until stability is ensured.

## Requirements

### Simple installation

```powershell
pip install -r requirements.txt
```
### Selective installation

If you only need to run heuristic-based algorithms, that is to say without the DL/RL requirements, please install partly dependencies.

```powershell
pip install numpy pandas matplotlib scipy networkx
```