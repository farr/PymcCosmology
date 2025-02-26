# Cosmology with pymc

Examples of using [pymc](https://www.pymc.io/welcome.html) to fit cosmology.

This repository contains a `conda` enviroment file that should install all the
necessary dependencies to install the package and run the example notebooks in
the `notebooks` directory.  To install the prerequesites:

```shell
conda env create -f environment.yml
conda activate pymc_cosmology
python -m pip install .  # To install this package in the env.  If you want, you can add the `-e` option to install in development mode
```

Now you can use the code and run the example notebooks.

Installing from `pip` (e.g. in a virtual environment) *should* work, but if you
want to run the notebooks you will have to install the `jupyterlab` and
`ipykernel` packages by hand (they are not listed as requirements in the
`setup.cfg` file, but *are* installed automatically in the `conda` environment).
