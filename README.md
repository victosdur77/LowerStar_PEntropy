This repository contains the code for compute lower stair filtration and persistence entropy as is done in the article: https://www.sciencedirect.com/science/article/abs/pii/S0165168416303486

- functions.py contains the functions necessary to compute it.
- Example.ipynb is an example of how to use it. It is based on the first example of the article.

You have to create a virtual enviroment of Python with version 3.10 to execute correctly the jupyter notebooks: 

```bash
python3 -m venv entorno python=3.10
```

Then, activate it (we use the command source because we are working in the WSL):

```bash
source entorno/bin/activate
```

Finally, we install the necessary libraries:

```bash
pip install numpy matplotlib gudhi scipy ripser
```

