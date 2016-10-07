# -*- coding: utf-8 -*-

import pandapower as pp

import pytest

def test_generation_only():
    """ testfile provided by erika kaempf in a bug report (Issue 161)
    """
    net = pp.from_pickle(filename="dumnet.p")
    runpp_with_consistency_checks(net)
    
if __name__ == "__main__":
    pytest.main(["openissues.py"])