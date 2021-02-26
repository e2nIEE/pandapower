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

@testset "test converting net from pandapower to pm" begin

    @testset " simbench grid 1-HV-urban--0-sw with ipopt solver for DCOPF" begin

        json_path = joinpath(test_path, "test_ipopt.json")

        pm = PpPmInterface.load_pm_from_json(json_path)

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

@testset "test run_powermodels" begin
    @testset " simbench grid 1-HV-urban--0-sw with ipopt solver for DCOPF" begin

        json_path = joinpath(test_path, "test_ipopt.json")

        result=run_powermodels(json_path)

        @test typeof(result) == Dict{String,Any}
        @test string(result["termination_status"]) == "LOCALLY_SOLVED"
        @test isapprox(result["objective"], -96.1; atol = 1e0)
        @test result["solve_time"] >= 0

    end

    # TODO: you can copy the test above and change some conditions, such as solver or model or even the grid and run test for it
end



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
