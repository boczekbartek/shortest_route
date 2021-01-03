# Rectangular city problem
We consider a rectangular (NY-like) city of 50×50 crossings.There are thus 2500 states and 4 actions. 

## Dependencies
* python > 3.6
* numpy
* numba
* matplotlib
* seaborn

## Running
```bash
python model-based.py && python q_learning.py
```

## Notes
Q-learning is written entirely in *numpy* so it can be optimized using *numba* library. Using numba gives ~9times learning process speedup.
