# SI-AIEP-paper



### Running python code

The python code was developed using `poetry` as a package manager. After installing poetry the repository can be initialized by running:

```bash
$ poetry install
```



The virtual environment containing the installed packages and the python binary is included in the repository to make it easier to use this repository. This does not require you to actually have poetry installed. The python binary can be accessed by running:

```bash
$ .venv/bin/python
```

The model code can be run from inside this REPL. Alternatively a file can be directly run by calling:

```bash
.venv/bin/python $file
```

where `$file` for example equals `plot_concentration_predictions.py`

### Installing pyearth

Since the pyearth package is not available to install directly from pip (it might be from conda if you have it), we have to jump through some hoops to install it. Please follow the installation instructions [here](https://github.com/scikit-learn-contrib/py-earth).

If you are not getting it to work, I've had many problems with it as well. In order to still be able to run the code, please comment the lines relating to the evaluation of the pyearth model in `plot_covariate_selection.py`.



### Running julia code

Make sure you have julia installed, and then run:

```bash
$ julia --project=.
julia> ]
(SI-AIEP-paper) pkg> instantiate
```

Now you can either run the julia code from the REPL or call:

```bash
$ julia $file --project=. 
```

