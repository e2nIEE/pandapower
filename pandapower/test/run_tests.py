# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import argparse
import os
import shutil
import tempfile
from multiprocessing import cpu_count

import pytest

from pandapower.test import test_path, tutorials_path

import pandapower as pp

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging
test_dir = os.path.abspath(os.path.join(pp.pp_dir, "test"))

logger = logging.getLogger()


def _remove_logger():
    for handler in logger.handlers:
        logger.removeHandler(handler)
        logger.setLevel(logging.CRITICAL)


def _get_cpus():
    # returns of a string of all available CPUs - 1 or 1 if you only have one CPU
    return str(cpu_count() - 1) if cpu_count() > 1 else str(1)


def run_all_tests(parallel=False, n_cpu=None):
    """ function executing all tests

    Inputs:
    parallel (bool, False) - If true and pytest-xdist is installed, tests are run in parallel

    n_cpu (int, None) - number of CPUs to run the tests on in parallel. Only relevant fo
        parallel runs.
    """

    if parallel:
        if n_cpu is None:
            n_cpu = _get_cpus()
        err = pytest.main([test_dir, "-xs", "-n", str(n_cpu), "-log_cli=false"])
        if err == 4:
            raise ModuleNotFoundError("Parallel testing not possible. "
                                      "Please make sure that pytest-xdist is installed correctly.")
        elif err > 2:
            logger.error("Testing not successfully finished.")
    else:
        pytest.main([test_dir, "-xs"])
    logger.setLevel(logging.INFO)


def run_fast_tests(parallel=False, n_cpu=None):
    """ function executing fast tests
    Only executes the tests which are **not** marked as slow with pytest.mark.slow

    parallel (bool, False) - If true and pytest-xdist is installed, tests are run in parallel

    n_cpu (int, None) - number of CPUs to run the tests on in parallel. Only relevant for
        parallel runs.

    """

    if parallel:
        if n_cpu is None:
            n_cpu = _get_cpus()
        err = pytest.main([test_dir, "-xs", "-m", "not slow", "-n", str(n_cpu)])
        if err == 4:
            raise ModuleNotFoundError("Parallel testing not possible. "
                                      "Please make sure that pytest-xdist is installed correctly.")
        elif err > 2:
            logger.error("Testing not successfully finished.")
    else:
        pytest.main([test_dir, "-xs", "-m", "not slow"])


def run_slow_tests(parallel=False, n_cpu=None):
    """ function executing slow tests
    Only executes the tests which are marked as slow with pytest.mark.slow

    parallel (bool, False) - If true and pytest-xdist is installed, tests are run in parallel

    n_cpu (int, None) - number of CPUs to run the tests on in parallel. Only relevant for
        parallel runs.
    """

    if parallel:
        if n_cpu is None:
            n_cpu = _get_cpus()
        err = pytest.main([test_dir, "-xs", "-m", "slow", "-n", str(n_cpu)])
        if err == 4:
            raise ModuleNotFoundError("Parallel testing not possible. "
                                      "Please make sure that pytest-xdist is installed correctly.")
        elif err > 2:
            logger.error("Testing not successfully finished.")
    else:
        pytest.main([test_dir, "-xs", "-m", "slow"])


def get_command_line_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-which', type=str, default="all", help="run 'fast' or 'all' tests")
    parser.add_argument('-n_cpu', type=int, default=1,
                        help="runs the tests in parallel if n_cpu > 1")

    args = parser.parse_args()
    # return as dict
    return vars(args)


def start_tests(**settings):
    n_cpu = settings["n_cpu"]
    parallel = False if n_cpu < 2 else True
    # run either fast or all tests
    _remove_logger()
    if settings["which"] == "fast":
        run_fast_tests(parallel=parallel, n_cpu=n_cpu)
    elif settings["which"] == "slow":
        run_slow_tests(parallel=parallel, n_cpu=n_cpu)
    else:
        run_all_tests(parallel=parallel, n_cpu=n_cpu)


def run_tutorials(parallel=False, n_cpu=None):
    """
    Function to execute all tutorials / jupyter notebooks.

    Copies the whole "tutorials" folder to a temporary folder which is removed after all
    notebooks have been executed. Errors in the notebooks show up as Failures.
    For futher options on nbmake, visit
    https://semaphoreci.com/blog/test-jupyter-notebooks-with-pytest-and-nbmake

    :param parallel : If true and pytest-xdist is nistalled, jupyter notebooks are running in parallel
    :type parallel : bool, default False
    :param n_cpu : number of CPUs to run the files on in parallel. Only relevant for parallel runs.
    :type n_cpu : int, default None
    :return : No Output.

    """
    try:
        import nbmake
    except ImportError:
        raise ModuleNotFoundError('Testing of jupyter notebooks requires the pytest extension '
                                  '"nbmake". Please make sure that nbmake is installed correctly.')

    # run notebooks in tempdir to safely remove output files
    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.copytree(tutorials_path, os.path.join(tmpdir, 'tmp'))
        test_dir = tmpdir

        if parallel:
            if n_cpu is None:
                n_cpu = 'auto'
            err = pytest.main(["--nbmake", f"-n={n_cpu}", test_dir])
            if err == 4:
                raise ModuleNotFoundError("Parallel testing not possible. Please make sure "
                                          "that pytest-xdist is installed correctly.")
            elif err > 2:
                logger.error("Testing not successfully finished.")
        else:
            pytest.main(["--nbmake", test_dir])


if __name__ == "__main__":
    # get some command line options
    # settings = get_command_line_args()
    # start_tests(**settings)
    # run_tutorials()
    run_all_tests(parallel=True, n_cpu=4)
