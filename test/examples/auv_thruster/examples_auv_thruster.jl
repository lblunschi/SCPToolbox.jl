#= Minimal examples loader for AUVThruster only. =#

module ExamplesAUVThruster

include("../../../src/SCPToolbox.jl")

module AUVThruster
include("parameters.jl")
include("definition.jl")
include("plots.jl")
include("tests.jl")
include("export_to_csv.jl")
end

end