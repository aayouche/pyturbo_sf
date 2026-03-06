Installation
============

PyTurbo_SF can be installed in several ways. Choose the method that best fits your needs.

Requirements
------------

Python Version
~~~~~~~~~~~~~~

PyTurbo_SF requires Python 3.12 or higher.

Dependencies
~~~~~~~~~~~~

The following packages are required:

* **numpy** (≥1.20.0) - Numerical computing
* **xarray** (≥0.19.0) - Labeled multi-dimensional arrays
* **scipy** (≥1.7.0) - Scientific computing
* **pandas** (≥1.3.0) - Data manipulation
* **joblib** (≥1.0.0) - Parallel computing

Optional dependencies for enhanced functionality:

* **matplotlib** (≥3.5.0) - Plotting and visualization
* **dask** (≥2021.10.0) - Parallel computing for large datasets
* **zarr** (≥2.10.0) - Chunked, compressed array storage
* **netcdf4** (≥1.5.0) - NetCDF file support

Installation Methods
--------------------

Method 1: PyPI (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to install PyTurbo_SF is using pip:

.. code-block:: bash

   pip install pyturbo_sf

To install with optional dependencies:

.. code-block:: bash

   pip install "pyturbo_sf[complete]"



Method 2: Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For development or to get the latest features:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/aayouche/pyturbo_sf.git
   cd pyturbo_sf

   # Install in development mode
   pip install -e .

   # Or with optional dependencies
   pip install -e .[dev,complete]

Method 3: From Source
~~~~~~~~~~~~~~~~~~~~~

To install from source without cloning:

.. code-block:: bash

   pip install git+https://github.com/aayouche/pyturbo_sf.git


Method 4: Virtual Environment Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We strongly recommend using a virtual environment to avoid dependency conflicts.

**Option A: Using venv (built-in Python)**

.. code-block:: bash

   # Create virtual environment
   python -m venv pyturbo_env
   
   # Activate (Linux/Mac)
   source pyturbo_env/bin/activate
   
   # Activate (Windows)
   pyturbo_env\Scripts\activate
   
   # Install with all optional dependencies (recommended)
   pip install "pyturbo_sf[complete]"
   
   # Or install base package only
   pip install pyturbo_sf

**Option B: Using Conda (Recommended for New Users)**

If you don't have Python installed or are new to Python, we recommend using Miniconda, 
a lightweight distribution that includes conda, Python, and essential packages.

1. **Install Miniconda** (Linux/Mac):

.. code-block:: bash

   # Download Miniconda installer
   curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   
   # Run installer
   bash Miniconda3-latest-Linux-x86_64.sh
   
   # Follow the prompts, then restart your terminal or run:
   source ~/.bashrc

For Mac (Apple Silicon):

.. code-block:: bash

   curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
   bash Miniconda3-latest-MacOSX-arm64.sh

For Windows, download the installer from https://docs.conda.io/en/latest/miniconda.html

2. **Create and activate a conda environment**:

.. code-block:: bash

   # Create a new environment with Python 3.12
   conda create -n pyturbo_env python=3.12
   
   # Activate the environment
   conda activate pyturbo_env

3. **Install PyTurbo_SF with all dependencies**:

.. code-block:: bash

   # Install with all optional dependencies (recommended)
   pip install "pyturbo_sf[complete]"
   
   # Or install base package only
   pip install pyturbo_sf

4. **Verify installation**:

.. code-block:: bash

   python -c "import pyturbo_sf; print(pyturbo_sf.__version__)"

.. note::
   
   To deactivate the environment when you're done:
   
   .. code-block:: bash
   
      conda deactivate




Verification
------------

To verify your installation, run the following in Python:

.. code-block:: python

   import pyturbo_sf
   print(f"PyTurbo_SF version: {pyturbo_sf.__version__}")
   
   # Test basic functionality
   import numpy as np
   import xarray as xr
   
   # Create simple test data
   x = np.linspace(0, 10, 100)
   data = np.sin(x) + 0.1 * np.random.randn(100)
   ds = xr.Dataset(
       data_vars={"signal": ("x", data)},
       coords={"x": x}
   )
   
   # Test structure function calculation
   bins = {'x': np.logspace(-1, 1, 10)}
   result = pyturbo_sf.bin_sf_1d(
       ds=ds,
       variables_names=["signal"],
       order=2,
       bins=bins,
       fun='scalar',
       bootsize={'x':10},
       initial_nbootstrap=5,
       max_nbootstrap=10
   )
   
   print("Installation successful!")



Common Issues
~~~~~~~~~~~~~

**ImportError: No module named 'pyturbo_sf'**
   - Ensure you've activated the correct environment
   - Verify installation with ``pip list | grep pyturbo``

**Memory errors with large datasets**
   - Use appropriate ``bootsize`` parameters
   - Consider using Dask for larger-than-memory datasets
   - Increase system swap space if needed

**Slow performance**
   - Use appropriate ``backend`` parameter ('loky', 'threading', 'multiprocessing')


**Convergence issues**
   - Increase ``max_nbootstrap`` parameter
   - Adjust ``convergence_eps`` threshold
   - Check data quality and structure


Getting Help
------------

If you encounter installation issues:

1. Check the `GitHub Issues <https://github.com/aayouche/pyturbo_sf/issues>`_

2. Create a new issue with:

   * Your operating system and Python version
   * Complete error message
   * Installation method used

3. Join our community discussions

Upgrading
---------

To upgrade PyTurbo_SF to the latest version:

.. code-block:: bash

   # Using pip
   pip install --upgrade pyturbo_sf
   

To upgrade to a specific version:

.. code-block:: bash

   pip install pyturbo_sf==1.0.7

Uninstallation
--------------

To remove PyTurbo_SF:

.. code-block:: bash

   # Using pip
   pip uninstall pyturbo_sf
   


Next Steps
----------

Once installed, check out the :doc:`quickstart` guide to begin using PyTurbo_SF, or explore the :doc:`examples` for detailed tutorials.
