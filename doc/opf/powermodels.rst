.. _powermodels:

Optimization with PowerModels.jl
=================================

Installation
--------------

If you are not yet using Julia, install it. Note that you need a version that is supported PowerModels, PyCall and pyjulia for the interface to work. Currently, `Julia 1.1 <https://julialang.org/downloads/>`_ is the most recent version of Julia that supports all these packages.

.. note:: You don't necessarily need a Julia IDE if you are using PowerModels through pandapower, but it might help for debugging to install an IDE such as `Juno <http://docs.junolab.org/latest/man/installation>`_. Also, PyCharm has a Julia Plugin.

1. Add the Julia binary folder (e.g. /Julia-1.1.0/bin) to the `system variable PATH <https://www.computerhope.com/issues/ch000549.htm>`_. Providing the path is correct, you can now enter the julia prompt by executing :code:`julia` in your shell.

2. The library `PyCall <https://github.com/JuliaPy/PyCall.jl#installation>`_ allows to use Python from inside julia. By default, PyCall uses the Conda.jl package to install a Miniconda distribution private to Julia. To use an already installed Python distribution (e.g. Anaconda), set the :literal:`PYTHON` environment variable inside the Julia prompt to e.g.: :code:`ENV["PYTHON"]="C:\\Anaconda3\\python.exe"`.


3. Access the package mode by typing :kbd:`]`. Now install the packages: :code:`add Ipopt PowerModels PyCall`.

4. Test your `PowerModels <https://lanl-ansi.github.io/PowerModels.jl/stable/#Installation-1>`_ installation by executing :code:`test PowerModels`. Then, test if calling Python from Julia works as described `here <https://github.com/JuliaPy/PyCall.jl#usage>`_.

.. note:: If you cannot plot using PyCall and PyPlot in Julia, see the workarounds offered `here <https://github.com/JuliaPy/PyCall.jl/issues/665>`_.

5. To call Julia from Python, install the pyjulia package with :code:`pip install julia`. Afterwards, test if everything works by importing PowerModels from Python with: :code:`from julia.PowerModels import run_ac_opf`. This takes some time, since Python starts a julia instance in the background, but it if the import completes without error everything is configured correctly and you can now use PowerModels to optimize pandapower networks.

.. warning:: A bug currently makes it necessary to modify a file of the pyjulia package. The exact line which needs to be modified depends on how you installed pyjulia. To find the exact line, have a look at your traceback after you execute the command :code:`from julia import Main`, e.g. in Pycharm (Corresponding `Github Issue <https://github.com/JuliaPy/pyjulia/issues/67#issuecomment-508407747>`_):

    .. code-block::

        File "C:\python\Anaconda3\lib\site-packages\julia\core.py", line 686, in __init__
        self.libjulia = ctypes.PyDLL(libjulia_path, ctypes.RTLD_GLOBAL)
        File "C:\python\Anaconda3\lib\ctypes\__init__.py", line 356, in __init__
        self._handle = _dlopen(self._name, mode)
        OSError: [WinError 127] The specified procedure could not be found

    In this case, it is line no. 686. Replace

    .. code-block::

        self.libjulia = ctypes.PyDLL(libjulia_path, ctypes.RTLD_GLOBAL)

    with

    .. code-block::

        cwd = os.getcwd()
        os.chdir(os.path.dirname(libjulia_path))
        self.libjulia = ctypes.PyDLL(libjulia_path, ctypes.RTLD_GLOBAL)
        os.chdir(cwd)

    Now Julia should work in your Python shell, Anaconda prompt and in Spyder, but currently not in Pycharm.

Usage
------

The usage is explained in the `PowerModels tutorial <https://github.com/e2nIEE/pandapower/blob/develop/tutorials/opf_powermodels.ipynb>`_.

.. autofunction:: pandapower.runpm_ac_opf

.. autofunction:: pandapower.runpm_dc_opf

.. autofunction:: pandapower.runpm

The TNEP optimization is explained in the `PowerModels TNEP tutorial <https://github.com/e2nIEE/pandapower/blob/develop/tutorials/tnep_powermodels.ipynb>`_.

.. autofunction:: pandapower.runpm_tnep