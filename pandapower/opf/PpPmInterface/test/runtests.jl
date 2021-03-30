# using Logging
# NullLogger()

import Pkg
Pkg.activate(joinpath(pwd(), "pandapower", "opf","PpPmInterface")) # for lyu
# Pkg.activate(".") #for maryam

using PpPmInterface
using Test

# test_path = joinpath(pwd(), "test", "data") # for maryam
test_path=joinpath(pwd(), "pandapower", "opf", "PpPmInterface", "test", "data") # for lyu

test_net=joinpath(test_path, "pm_test.json")
test_ipopt = joinpath(test_path, "test_ipopt.json")
test_Gurobi = joinpath(test_path, "test_Gurobi.json") #use gurobi to solve
test_ots = joinpath(test_path, "test_ots.json")
test_tnep = joinpath(test_path, "test_tnep.json")

@testset "test converting net from pandapower to pm" begin

# simbench grid 1-HV-urban--0-sw with ipopt solver
        pm = PpPmInterface.load_pm_from_json(test_ipopt)

        @test length(pm["bus"]) == 82
        @test length(pm["gen"]) == 1
        @test length(pm["branch"]) == 116
        @test length(pm["load"]) == 177

        model = PpPmInterface.get_model(pm["pm_model"])
        # FIXME: when we register the package we need to change it to "DCPPowerModel"
        @test string(model) == "PowerModels.DCPPowerModel"

        solver = PpPmInterface.get_solver(pm["pm_solver"], pm["pm_nl_solver"], pm["pm_mip_solver"],
        pm["pm_log_level"], pm["pm_time_limit"], pm["pm_nl_time_limit"], pm["pm_mip_time_limit"])

        @test string(solver.optimizer_constructor) == "Ipopt.Optimizer"

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

@testset "test run_powermodels" begin
# simbench grid 1-HV-urban--0-sw with ipopt solver

        result=run_powermodels(test_ipopt)

        @test typeof(result) == Dict{String,Any}
        @test string(result["termination_status"]) == "LOCALLY_SOLVED"
        @test isapprox(result["objective"], -96.1; atol = 1e0)
        @test result["solve_time"] >= 0


    # TODO: you can copy the test above and change some conditions, such as solver or model or even the grid and run test for it
end



@testset "PpPmInterface.jl" begin
        # for i in ["","_ots","_powerflow","_tnep"]
        #         comm="result=run_powermodels"*i*"(test_ipopt)"
        #         eval(Meta.parse(comm))
        #         @test isa(result, Dict{String,Any})
        #         @test result["solve_time"]>=0
        # end
        result=run_powermodels(test_ipopt)
        @test isa(result, Dict{String,Any})
        string(result["termination_status"]) == "LOCALLY_SOLVED"
        @test result["solve_time"]>=0
        # result=run_powermodels_mn_storage(test_Gurobi)
        # @test isa(result, Dict{String,Any})
        # @test result["solve_time"]>=0
        result=run_powermodels_ots(test_ots)
        @test isa(result, Dict{String,Any})
        string(result["termination_status"]) == "LOCALLY_SOLVED"
        @test result["solve_time"]>=0
        result=run_powermodels_powerflow(test_Gurobi)
        @test isa(result, Dict{String,Any})
        string(result["termination_status"]) == "LOCALLY_SOLVED"
        @test result["solve_time"]>=0
        result=run_powermodels_tnep(test_tnep)
        @test isa(result, Dict{String,Any})
        string(result["termination_status"]) == "LOCALLY_SOLVED"
        @test result["solve_time"]>=0
end
