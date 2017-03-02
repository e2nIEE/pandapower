===================================================
Installing development version
===================================================

The pandapower development version is hosted on github: https://github.com/lthurner/pandapower
This guide assumes that there is already a working python environment (preferably with anaconda distribution) availble on your computer.

1. Download and install git from https://git-scm.com

2. Open a git shell and navigate to the directory where you want to keep your pandapower files.

3. Run the following command ::

    git clone https://github.com/lthurner/pandapower develop
       
3. Set your python path to the outer pandapower folder (/pandapower, NOT pandapower/pandapower). 

4. Install dependencies if missing via pip install: ::

    pip install pypower