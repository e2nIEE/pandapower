.. _pandamodels:

####################################
Optimization with PandaModels.jl
####################################


Introduction
--------------------

`PandaModels.jl <https://github.com/e2nIEE/PandaModels.jl>`__ (pandapower + PowerModels.jl) is an interface
(Julia package) enabling the connection of pandapower and PowerModels in a stable and functional way. Except for calling
the implemented optimization models in PowerModels, users can create custom optimization models with PandaModels.
Presently, users can solve some reactive power optimization problems with PandaModels.


Installation
--------------

pandapower calls Julia through `juliacall <https://github.com/JuliaPy/PythonCall.jl>`__. Install it with the
``pandamodels`` extra:

::

    pip install pandapower[pandamodels]

Julia and PandaModels.jl do not need to be installed by hand. pandapower declares PandaModels.jl in
``pandapower/juliapkg.json``, and `juliapkg <https://github.com/JuliaPy/pyjuliapkg>`__ installs a suitable Julia
version and all Julia packages into its own environment the first time ``juliacall`` is imported. This first import
takes a few minutes; later imports only take a few seconds. To check the installation, run the PandaModels tests:

::

    pytest pandapower/test/opf/test_pandamodels_runpm.py

.. note:: The first optimization in a Python process additionally compiles the Julia code (about 10 s). Run all
    optimizations of a study in one Python process to pay this only once.

**Using a local custom PandaModels.jl checkout:** register it once in the juliapkg project. juliapkg then uses the checkout
in development mode instead of the released package:

::

    import juliapkg
    juliapkg.add("PandaModels", "2dbab86a-7cbf-476f-9afe-75ffd3079e7c", path="path/to/PandaModels.jl", dev=True)

Remove it again with ``juliapkg.rm("PandaModels")``. The environment variable ``PYTHON_JULIAPKG_PROJECT`` selects a
separate Julia project, which keeps such a development setup apart from the default one.


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


Redispatch
------------

A simple redispatch optimization is available via :code:`runpm_redispatch`. Starting from a base
dispatch (the generator setpoints of a previously computed power flow), it adjusts the participating
generators as little as possible - or at minimal cost - so that all network constraints (branch
loading, bus voltage limits, generator limits) are satisfied.

Only controllable ``gen`` and ``sgen`` participate. An element is selected for redispatch if it is
``controllable`` **and** has a ``poly_cost`` entry with (non-NaN) ``redispatch_up_eur_per_mw`` and
``redispatch_down_eur_per_mw``. These two coefficients are added to
:code:`create_poly_cost`/:code:`create_poly_costs`:

::

    pp.create_poly_cost(net, gen_idx, "gen", cp1_eur_per_mw=cp1,
                        redispatch_up_eur_per_mw=cost_up,
                        redispatch_down_eur_per_mw=cost_down)

Two objective modes are available:

- ``redispatch_cost=False`` (default): minimize the squared deviation from the base dispatch
  (``sum((pg - pg0)^2)``) - the "least redispatch" that satisfies the constraints.
- ``redispatch_cost=True``: minimize the total redispatch cost, splitting each generator's
  adjustment into an upward and downward part weighted by ``redispatch_up_eur_per_mw`` and
  ``redispatch_down_eur_per_mw``.

The base dispatch is read from the result tables, so a power flow result must be present. By default
(``init_pq="results"``) the generator setpoints are taken from a previously run power flow. Both
``ACPPowerModel`` (default) and ``DCPPowerModel`` are supported.

.. autofunction:: pandapower.runpm_redispatch


The TNEP optimization is explained in the `PandaModels TNEP tutorial <https://github.com/e2nIEE/pandapower/blob/develop/tutorials/pandamodels_tnep.ipynb>`_. Additional packages including "juniper"

.. autofunction:: pandapower.runpm_tnep
