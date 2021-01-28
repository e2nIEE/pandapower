import Pkg
Pkg.activate(joinpath(homedir(), "pandapower", "pandapower", "opf","PpPmInterface"))

# Pkg.activate("test")
# Pkg.dev(".")

using PpPmInterface
using Test

json_path = joinpath(pwd(), "test", "test_net.json");

@testset "PpPmInterface.jl" begin
    pm = load_pm_from_json(json_path)
    # result = run_powermodels(json_path)
end
