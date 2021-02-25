import Pkg
Pkg.activate(joinpath(homedir(),"GitHub", "pandapower", "pandapower", "opf","PpPmInterface"))

using PpPmInterface
using Test

test_path=joinpath(homedir(),"GitHub", "pandapower", "pandapower", "opf", "PpPmInterface", "test")
test_net=joinpath(test_path, "pm_test.json")
test_ipopt = joinpath(test_path, "test_ipopt.json")
test_Gurobi = joinpath(test_path, "test_Gurobi.json") #use gurobi to solve
@testset "PpPmInterface.jl" begin
        # for i in ["","_ots","_powerflow","_tnep"]
        #         comm="run_powermodels"*i*"(test_ipopt)"
        #         result=Meta.parse($comm)
        #         @test typeof(result)==Dict{String,Any}
        #         @test result["solve_time"]>=0
        # end
        result=run_powermodels(test_ipopt)
        @test typeof(result)==Dict{String,Any}
        @test result["solve_time"]>=0
        # result=run_powermodels_mn_storage(test_Gurobi)
        # @test typeof(result)==Dict{String,Any}
        # @test result["solve_time"]>=0
        result=run_powermodels_ots(test_net) #use Gurobi
        @test typeof(result)==Dict{String,Any}
        @test result["solve_time"]>=0
        result=run_powermodels_powerflow(test_ipopt)
        @test typeof(result)==Dict{String,Any}
        @test result["solve_time"]>=0
        result=run_powermodels_tnep(test_ipopt)
        @test typeof(result)==Dict{String,Any}
        @test result["solve_time"]>=0
end

# json_path = joinpath(pwd(), "pandapower","opf","PpPmInterface","test", "test_net.json");
# @testset "PpPmInterface.jl" begin
#     pm = load_pm_from_json(json_path)
#     # result = run_powermodels(json_path)
# end
