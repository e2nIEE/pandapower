
function get_model(model_type)
    """
    gets the model function
    model_type (str) - examples: "ACPPowerModel", "DCPPowerModel", "SOCWRPowerModel"...
    see: https://lanl-ansi.github.io/PowerModels.jl/stable/formulation-details/
    """

    s = Symbol(model_type)
    # FIXME: when we register the package as global package with sepeate repo, we must change it to getfield(Main, s)
    return getfield(PpPmInterface, s)
end

function get_solver(optimizer::String, nl::String="ipopt", mip::String="cbc",
    log_level::Int=0, time_limit::Float64=Inf, nl_time_limit::Float64=Inf,
    mip_time_limit::Float64=Inf, ipopt_tol::Float64=1e-8)

    if optimizer == "gurobi"
            solver = JuMP.optimizer_with_attributes(Gurobi.Optimizer, "TimeLimit" => time_limit, "OutputFlag" => log_level)
    end

    if optimizer == "ipopt"
                solver = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "print_level" => log_level, "max_cpu_time" => time_limit,
                "tol" => ipopt_tol)
    end

    if optimizer == "juniper" && nl == "ipopt" && mip == "cbc"
        mip_solver = JuMP.optimizer_with_attributes(Cbc.Optimizer, "logLevel" => log_level, "seconds" => mip_time_limit)
        nl_solver = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "print_level" => log_level, "max_cpu_time" => nl_time_limit)
        solver = JuMP.optimizer_with_attributes(Juniper.Optimizer,
                     "nl_solver" => nl_solver,
                     "mip_solver" => mip_solver,
                     "log_levels" => [],
                     "time_limit" => time_limit)
    end

    if optimizer == "juniper" && nl == "gurobi" && mip == "cbc"
        mip_solver = JuMP.optimizer_with_attributes(Cbc.Optimizer, "logLevel" => log_level, "seconds" => mip_time_limit)
        nl_solver = JuMP.optimizer_with_attributes(Gurobi.Optimizer, "TimeLimit" => nl_time_limit)
        solver = JuMP.optimizer_with_attributes(Juniper.Optimizer,
                     "nl_solver" => nl_solver,
                     "mip_solver" => mip_solver,
                     "log_levels" => [],
                     "time_limit" => time_limit)
    end

    if optimizer == "juniper" && nl == "gurobi" && mip == "gurobi"
        mip_solver = JuMP.optimizer_with_attributes(Gurobi.Optimizer, "TimeLimit" => mip_time_limit)
        nl_solver = JuMP.optimizer_with_attributes(Gurobi.Optimizer, "TimeLimit" => nl_time_limit)
        solver = JuMP.optimizer_with_attributes(Juniper.Optimizer,
                     "nl_solver" => nl_solver,
                     "mip_solver" => mip_solver,
                     "log_levels" => [],
                     "time_limit" => time_limit)
    end

    if optimizer == "knitro"
        solver = JuMP.optimizer_with_attributes(KNITRO.Optimizer)
    end

    if optimizer == "cbc"
        solver = JuMP.optimizer_with_attributes(Cbc.Optimizer, "seconds" => time_limit)
    end

    if optimizer == "scip"
        solver = JuMP.optimizer_with_attributes(SCIP.Optimizer)
    end

    return solver

end

function load_pm_from_json(json_path)
    pm = Dict()
    open(json_path, "r") do f
        pm = JSON.parse(f)  # parse and transform data
    end

    for (idx, gen) in pm["gen"]
        if gen["model"] == 1
            pm["gen"][idx]["cost"] = convert(Array{Float64,1}, gen["cost"])
        end
    end
    if pm["correct_pm_network_data"]
        PowerModels.correct_network_data!(pm)
    end
    return pm
end
