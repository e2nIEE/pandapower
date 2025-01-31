.. _powermodels:

Optimization with PowerModels.jl
=================================

Installation
--------------

If you are not yet using Julia, install it. Note that you need a version that is supported PowerModels, PyCall and pyjulia for the interface to work. Currently, `Julia 1.5 <https://julialang.org/downloads/>`_ is the most recent version of Julia that supports all these packages.

.. note:: You don't necessarily need a Julia IDE if you are using PowerModels through pandapower, but it might help for debugging to install an IDE such as `Juno <http://docs.junolab.org/latest/man/installation>`_. Also, PyCharm has a Julia Plugin.

1. Add the Julia binary folder (e.g. /Julia-1.5.0/bin) to the `system variable PATH <https://www.computerhope.com/issues/ch000549.htm>`_. Providing the path is correct, you can now enter the julia prompt by executing :code:`julia` in your shell.

2. The library `PyCall <https://github.com/JuliaPy/PyCall.jl#installation>`_ allows to use Python from inside julia. By default, PyCall uses the Conda.jl package to install a Miniconda distribution private to Julia. To use an already installed Python distribution (e.g. Anaconda), set the :literal:`PYTHON` environment variable inside the Julia prompt to e.g.: :code:`ENV["PYTHON"]="C:\\Anaconda3\\python.exe"`.


3. Access the package mode by typing :kbd:`]`. Now install the packages: :code:`add Ipopt PowerModels PyCall`.

4. Test your `PowerModels <https://lanl-ansi.github.io/PowerModels.jl/stable/#Installation-1>`_ installation by executing :code:`test PowerModels`. Alternatively, you can call :code:`using Pkg` and then :code:`Pkg.test("PowerModels")` Then, test if calling Python from Julia works as described `here <https://github.com/JuliaPy/PyCall.jl#usage>`_.

.. note:: If you cannot plot using PyCall and PyPlot in Julia, see the workarounds offered `here <https://github.com/JuliaPy/PyCall.jl/issues/665>`_.

5. To call Julia from Python, install the pyjulia package with :code:`pip install julia`. Afterwards, test if everything works by importing PowerModels from Python with: :code:`from julia.PowerModels import run_ac_opf`. This takes some time, since Python starts a julia instance in the background, but it if the import completes without error everything is configured correctly and you can now use PowerModels to optimize pandapower networks.

6. Additional packages are required to use the pandapower - PowerModels.jl interface with all features like TNEP or OTS. Install the "JSON" and "JuMP" packages with, e.g., :code:`julia -e 'import Pkg; Pkg.add("JSON"); Pkg.add("JuMP");` and maybe also `julia -e 'import Pkg; Pkg.add("Cbc"); Pkg.add("Juniper")'` to get the TNEP and OTS libraries. Alternatively, install these packages by entering :code:`]` inside the julia console and calling :code:`add JSON` :code:`add JuMP`

7. You can then finally test if the pandapower - PowerModels.jl interface works by navigating to your local pandapower test folder :code:`pandapower/pandapower/test/opf` folder and run :code:`python-jl test_powermodels.py`. If everything works there should be no error.

Additional Solvers
--------------------

Optional additional solvers, such as `Gurobi <https://www.gurobi.com/>`_ are compatible to PowerModels.jl. To use these solvers, you first have to install the solver itself on your system and then the julia interface. Gurobi is very fast for linear problems such as the DC model and free for academic usage. Let's do this step by step for Gurobi:

1. Download and install from `Gurobi download <https://www.gurobi.com/downloads/>`_  (you'll need an account for this)

2. Get your Gurobi license at `Gurobi license <https://www.gurobi.com/downloads/licenses/>`_ and download it (remember where you stored it).

3. Activate the license by calling :code:`grbgetkey YOUR_KEY` as described on the Gurobi license page.

4. Add some Gurobi paths and the license path to your local PATH environment variables. In linux you can just open your `.bashrc` file with, e.g., :code:`nano .bashrc` in your home folder and add:

::

    # gurobi
    export GUROBI_HOME="/opt/gurobi_VERSION/linux64"
    export PATH="${PATH}:${GUROBI_HOME}/bin"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
    export GRB_LICENSE_FILE="/PATH_TO_YOUR_LICENSE_DIR/gurobi.lic"

5. Install the julia - `Gurobi interface <https://github.com/jump-dev/Gurobi.jl>`_ with :code:`julia -e 'import Pkg; Pkg.add("Gurobi");'`

6. Build and test your Gurobi installation by calling :code:`julia` and then :code:`import Pkg; Pkg.build("Gurobi")`. This should compile without an error. 

7. Now, you can use Gurobi to solve your linear problems, e.g., the DC OPF, with :code:`runpm_dc_opf(net, pm_model="DCPPowerModel", pm_solver="gurobi")`


Usage
------

The usage is explained in the `PowerModels tutorial <https://github.com/e2nIEE/pandapower/blob/develop/tutorials/opf_powermodels.ipynb>`_.

.. autofunction:: pandapower.runpm_ac_opf

.. autofunction:: pandapower.runpm_dc_opf

.. autofunction:: pandapower.runpm

The TNEP optimization is explained in the `PowerModels TNEP tutorial <https://github.com/e2nIEE/pandapower/blob/develop/tutorials/tnep_powermodels.ipynb>`_. Additional packages including "juniper" 

.. autofunction:: pandapower.runpm_tnep