import Pkg
# Pkg.activate(joinpath(homedir(),"GitHub", "pandapower", "pandapower", "opf","PpPmInterface")) # for lyu
Pkg.activate(".") #for maryam
# Pkg.update()
# Pkg.build()
# Pkg.resolve()

using PpPmInterface
using Test

test_path = joinpath(pwd(), "test") # for maryam
# test_path=joinpath(homedir(),"pandapower", "pandapower", "opf", "PpPmInterface", "test") # for lyu

# test_net=joinpath(test_path, "pm_test.json")
# test_ipopt = joinpath(test_path, "test_ipopt.json")
# test_Gurobi = joinpath(test_path, "test_Gurobi.json") #use gurobi to solve

@testset "test converting net from pandapower to pm" begin

    @testset " simbench grid 1-HV-urban--0-sw with ipopt solver for DCOPF" begin
        
        json_path = joinpath(test_path, "test_ipopt.json")
        pm = PpPmInterface.load_pm_from_json(json_path)

        @test length(pm["bus"]) == 82
        @test length(pm["gen"]) == 1
        @test length(pm["branch"]) == 116
        @test length(pm["load"]) == 177

    end

    # TODO: you can copy the test above and change some conditions, such as solver or model or even the grid and run test for it
    # @testset " simbench grid 1-HV-urban--0-sw with Gurobi solver for ACOPF" begin
    #     json_path = joinpath(test_path, "test_ipopt.json")
    #     pm = PpPmInterface.load_pm_from_json(json_path)
    #
    #     @test length(pm["bus"]) == 82
    #     @test length(pm["gen"]) == 1
    #     @test length(pm["branch"]) == 116
    #     @test length(pm["load"]) == 177
    #
    # end

    end
end

# @testset "PpPmInterface.jl" begin
#         # for i in ["","_ots","_powerflow","_tnep"]
#         #         comm="run_powermodels"*i*"(test_ipopt)"
#         #         result=Meta.parse($comm)
#         #         @test typeof(result)==Dict{String,Any}
#         #         @test result["solve_time"]>=0
#         # end
#         result=run_powermodels(test_ipopt)
#         @test typeof(result)==Dict{String,Any}
#         @test result["solve_time"]>=0
#         # result=run_powermodels_mn_storage(test_Gurobi)
#         # @test typeof(result)==Dict{String,Any}
#         # @test result["solve_time"]>=0
#         result=run_powermodels_ots(test_net) #use Gurobi
#         @test typeof(result)==Dict{String,Any}
#         @test result["solve_time"]>=0
#         result=run_powermodels_powerflow(test_ipopt)
#         @test typeof(result)==Dict{String,Any}
#         @test result["solve_time"]>=0
#         result=run_powermodels_tnep(test_ipopt)
#         @test typeof(result)==Dict{String,Any}
#         @test result["solve_time"]>=0
# end

# json_path = joinpath(pwd(), "pandapower","opf","PpPmInterface","test", "test_net.json");
# @testset "PpPmInterface.jl" begin
#     pm = load_pm_from_json(json_path)
#     # result = run_powermodels(json_path)
# end
