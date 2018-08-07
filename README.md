# Installation

Using the terminal, `cd` in the directory in which you want to store this package, and execute the following commands:

```
git clone https://github.com/lucainnocenti/python-manybosonstates
cd python-manybosonstates
pip install -e .
```

After this, you should be able to use the package by simply importing it.
To test it, try running the following code:
```
import manybosonstates.manybosonstates as mb
mb.ManyBosonFockState(4, (0, 1)).get_mol()
```
If the output is `[1, 1, 0, 0]`, then you are good to go.
See [manybosonstates_examples.ipynb](../notebooks/manybosonstates_examples.ipynb) for usage examples.
