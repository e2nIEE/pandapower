.. _pandamodels:

####################################
Optimization with PandaModels.jl
####################################


Introduction
--------------------

`PandaModels.jl <https://github.com/e2nIEE/PandaModels.jl>`_ (pandapower + PowerModels.jl) is an interface (Julia package) enabling the connection of pandapower and PowerModels in a stable and functional way. Except for calling the implemented optimization models in PowerModels, users can create custom optimization models with PandaModels. Presently, users can solve some reactive power optimization problems with PandaModels.


Installation
--------------

If you are not yet using `Julia <https://julialang.org/downloads/>`_, install it. For the interface to work, note that you need a version that is supported by PowerModels, PyCall and pyjulia. Currently, Former julia versions are available `here <https://julialang.org/downloads/oldreleases/>`__.

.. note:: You don't necessarily need a Julia IDE if you are using PandaModels through pandapower, but it might help for debugging to install an IDE such as `Juno <http://docs.junolab.org/latest/man/installation>`_. Also, PyCharm has a Julia Plugin.

1. Add the Julia binary folder (e.g. `C:\Users\username\AppData\Local\Programs\Julia-1.8.0\bin\` on Windows or `/Applications/Julia-1.5.app/Contents/Resources/julia/bin` on MacOS) to the `system variable PATH <https://www.computerhope.com/issues/ch000549.htm>`_. Providing the path is correct, you can now enter the julia prompt by executing :code:`julia` in your shell (on Windows, rebooting the system is needed to take advantage of changes to the :code:`PATH`.

2. The library `PyCall <https://github.com/JuliaPy/PyCall.jl#installation>`__ allows to use Python from inside julia. By default, PyCall uses the Conda.jl package to install a Miniconda distribution private to Julia. To use an already installed Python distribution (e.g. Anaconda), set the :literal:`PYTHON` environment variable inside the Julia prompt.

   On Windows:

    :code:`ENV["PYTHON"]=raw"C:\\Anaconda3\\python.exe"`

   On MacOS:

    :code:`ENV["PYTHON"]="/Users/%Username/opt/anaconda3/bin/python"`


3. Access the package manager by typing :code:`]`. Now install the packages: :code:`add Ipopt PowerModels PyCall`. To pass the python environment variable, running :code:`build PyCall` inside the julia package manager may be neccessary.

4. Inside package manager, test your `PowerModels <https://lanl-ansi.github.io/PowerModels.jl/stable/#Installation-1>`_ installation by executing :code:`test PowerModels`. Alternatively, you can call :code:`using Pkg` and then :code:`Pkg.test("PowerModels")` outside the package manager directly as julia expression. Then, test wether calling Python from Julia works, as described `here <https://github.com/JuliaPy/PyCall.jl#usage>`__.

.. note:: If you cannot plot using PyCall and PyPlot in Julia, see the workarounds offered `here <https://github.com/JuliaPy/PyCall.jl/issues/665>`__.

5. To call Julia from Python, install the pyjulia package with :code:`pip install julia`. Afterwards, test if everything works by importing PowerModels from Python with: :code:`from julia.PowerModels import run_ac_opf`. This takes some time, since Python starts a julia instance in the background, but it if the import completes without error everything is configured correctly and you can now use PowerModels to optimize pandapower networks.

6. Additional packages are required to use the pandapower - PowerModels.jl interface with all features like TNEP or OTS. Install the "JSON" and "JuMP" packages with, e.g., :code:`julia -e 'import Pkg; Pkg.add("JSON"); Pkg.add("JuMP");` and maybe also `julia -e 'import Pkg; Pkg.add("Cbc"); Pkg.add("Juniper")'` to get the TNEP and OTS libraries. Alternatively, install these packages by entering :code:`]` inside the julia console and calling :code:`add JSON` :code:`add JuMP`

7. Now install our interface `PandaModels.jl` by type :code:`add PandaModels` inside Julia package manager.

8. Finally, you can then test whether the PandaModels.jl interface works: Navigate to your local pandapower test folder :code:`pandapower/pandapower/test/opf` folder and run :code:`python-jl test_pandamodels_runpm.py` or :code:`pytest test_pandamodels_runpm.py` if pytest is intalled. If everything works there should be no error.


Additional Solvers
--------------------

Optional additional solvers, such as `Gurobi <https://www.gurobi.com/>`_ are compatible to PowerModels.jl. To use these solvers, you first have to install the solver itself on your system and then the julia interface. Gurobi is very fast for linear problems such as the DC model and free for academic usage. Let's do this step by step for Gurobi:

1. Download and install from `Gurobi download <https://www.gurobi.com/downloads/>`_ (you'll need an account for this)

2. Run the file to get the gurobi folder, e.g., in linux you need to run :code:`tar -xzf gurobi<version>_linux64.tar.gz`

3. Get your Gurobi license at `Gurobi license <https://www.gurobi.com/downloads/licenses/>`_ and download it (remember where you stored it).

4. Activate the license by calling :code:`grbgetkey YOUR_KEY` as described on the Gurobi license page.

5. Add some Gurobi paths and the license path to your local PATH environment variables. In linux you can just open your `.bashrc` file with, e.g., :code:`nano .bashrc` in your home folder and add:

::

    # Linux

    # gurobi
    export GUROBI_HOME="/opt/gurobi_VERSION/linux64"
    export PATH="${PATH}:${GUROBI_HOME}/bin"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
    export GRB_LICENSE_FILE="/PATH_TO_YOUR_LICENSE_DIR/gurobi.lic"


::

    # MacOS

    # gurobi
    export GUROBI_HOME="/Library/gurobiVERSION/mac64"
    export PATH="$PATH:$GUROBI_HOME/bin"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$GUROBI_HOME/lib"
    export GRB_LICENSE_FILE="/PATH_TO_YOUR_LICENSE_DIR/gurobi.lic"


7. Install the  `julia - Gurobi interface <https://github.com/jump-dev/Gurobi.jl>`_ and set the GUROBI_HOME environment with

    :code:`julia -e 'import Pkg; Pkg.add("Gurobi");'`

   or type

    :code:`add Gurobi`

   inside Julia package mode.

8. Build and test your Gurobi installation by entering :code:`julia` prompt and then :code:`import Pkg; Pkg.build("Gurobi")`. This should compile without an error.

9. Now, you can use Gurobi to solve your linear problems, e.g., the DC OPF, with :code:`runpm_dc_opf(net, pm_model="DCPPowerModel", pm_solver="gurobi")`


Usage
------

The usage is explained in the `PandaModels tutorial <https://github.com/e2nIEE/pandapower/blob/develop/tutorials/pandamodels_opf.ipynb>`_.

.. autofunction:: pandapower.runpm_ac_opf

.. autofunction:: pandapower.runpm_dc_opf

.. autofunction:: pandapower.runpm

The TNEP optimization is explained in the `PandaModels TNEP tutorial <https://github.com/e2nIEE/pandapower/blob/develop/tutorials/pandamodels_tnep.ipynb>`_. Additional packages including "juniper"

.. autofunction:: pandapower.runpm_tnep
