======================================================================
Installing with other Distributions than Anaconda
======================================================================

pandapower can of course also be used with other distributions besides Anaconda. It is however important that the following packages are included:

- numpy
- scipy
- numba
- matplotlib

since these packages depend on C-libraries and cannot be easily installed through pip. If you use a distribution that does not include one of these
packages, your only option is to either build the libraries yourself or to switch to a different distribution.

If these packages are however included in your distribution, installing pandapower is as simple as opening a command prompt (e.g. start-->cmd on windows systems) and running ::

    pip install pandapower