from pandapower.networks.create_examples import example_simple
from pandapower import diagnostic

net = example_simple()
net.load.at[0, "p_mw"] = 1000

report = diagnostic(net)
print(report["test_continuous_bus_indices"])