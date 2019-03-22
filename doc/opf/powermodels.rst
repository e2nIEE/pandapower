.. _powermodels:

Optimization with PowerModels.jl
=================================

Installation
--------------

#. If you are not yet using Julia, install it.

    .. note::
        You need a version that is supported PowerModels, PyCall and pyjulia for the interface to work. Currently, the most recent version of Julia that supports all these packages is `Julia 0.64 <https://julialang.org/downloads/oldreleases.html>`_.
        
    You don't necessarily need a Julia IDE if you are using PowerModels through pandapower, but it might help for debugging to install an IDE such as `Juno <http://docs.junolab.org/latest/man/installation.html>`_. 

#. `Install PowerModels.jl <https://lanl-ansi.github.io/PowerModels.jl/stable/#Installation-1>`_

#. Configure Julia to be able to call Python

    - open the Julia console
    - set ENV["PYTHON"] to your Python executable (e.g. ENV["PYTHON"]="C:\\Anaconda\\python.exe")  
    - Install PyCall with Pkg.add("PyCall")
    - test if calling Python from Julia works as described `here <https://github.com/JuliaPy/PyCall.jl#usage>`_

    .. note::    
        PyCall is only tested with Python 3.6 and 3.7, so make sure to use one of those versions 


#. Configure Python to be able to call Julia 

    - Add the Julia binary folder (e.g. /Julia-0.6.4/bin) to the `system variable PATH <https://www.computerhope.com/issues/ch000549.htm>`_
    - Install pyjulia with :code:`pip install julia`
    - test if everything works by importing PowerModels from Python with: :code:`from julia.PowerModels import run_ac_opf`. This takes some time, since Python starts a julia instance in the background, but it if the import completes without error everything is configured correctly and you can now use PowerModels to optimize pandapower networks.
      
Usage
------

The usage is explained in the `PowerModels tutorial <https://github.com/e2nIEE/pandapower/blob/develop/tutorials/opf_powermodels.ipynb>`_.

.. autofunction:: pandapower.runpm_ac_opf

.. autofunction:: pandapower.runpm_dc_opf

.. autofunction:: pandapower.runpm
