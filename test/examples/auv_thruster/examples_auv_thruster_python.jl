module ExamplesAUVThrusterPython

include("../../../src/SCPToolbox.jl")

module AUVThrusterPython

include("parameters.jl")
include("definition.jl")
include("plots.jl")
include("auv_scvx_opt.jl")
include("export_to_csv.jl")
end

end