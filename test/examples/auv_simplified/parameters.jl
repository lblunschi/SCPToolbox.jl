#= AUVSimplified landing flip maneuver data structures and custom methods.

Disclaimer: the data in this example is obtained entirely from publicly
available information, e.g. on reddit.com/r/spacex, nasaspaceflight.com, and
spaceflight101.com. No SpaceX engineers were involved in the creation of this
code.

Sequential convex programming algorithms for trajectory optimization.
Copyright (C) 2021 Autonomous Controls Laboratory (University of Washington)

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <https://www.gnu.org/licenses/>. =#

using LinearAlgebra
using ..SCPToolbox

# ..:: Data structures ::..

#= AUVSimplified vehicle parameters. =#
struct AUVSimplifiedParameters
    id_r::IntRange # Position indices of the state vector
    id_v::IntRange # Velocity indices of the state vector
    id_e::Int      # Energy indice of state vector
    id_u::IntRange # Indices of the thrust input vector
    u_max::Real    # [N] Maximum thrust
    u_min::Real    # [N] Minimum thrust
    v_max::Real    # [m/s] Maximum velocity
    v_min::Real    # [m/s] Minimum velocity
    v_min_u::Real  # [m/s] Minimum forward velocity (no reverse)
    thruster_allocation_matrix::RealMatrix
end

#= AUVSimplified hydrodynamics parameters. =#
struct AUVSimplifiedHydrodynamicsParameters
    mass::Real # [kg] Mass of the vehicle
    inertia_z::Real # [kg*m^2] Yaw moment of inertia
    added_mass_x::Real # [kg] Added mass in surge
    added_mass_y::Real # [kg] Added mass in sway
    added_mass_z::Real # [kg] Added mass in heave
    added_mass_yaw::Real # [kg*m^2] Added mass in
    linear_drag_x::Real # [kg/s] Linear drag in surge
    linear_drag_y::Real # [kg/s] Linear drag in sway
    linear_drag_z::Real # [kg/s] Linear drag in heave
    linear_drag_yaw::Real # [kg*m^2/s] Linear drag in yaw
    quadratic_drag_x::Real # [kg/m] Quadratic drag in surge
    quadratic_drag_y::Real # [kg/m] Quadratic drag in sway
    quadratic_drag_z::Real # [kg/m] Quadratic drag in heave
    quadratic_drag_yaw::Real # [kg*m] Quadratic drag in yaw
    buoyancy::Real # [N] buoyancy force
    weight::Real # [N] Weight force
end

##=Keyword constructor (outer constructor) =#
function AUVSimplifiedHydrodynamicsParameters(; mass, inertia_z,
    added_mass_x, added_mass_y, added_mass_z, added_mass_yaw,
    linear_drag_x, linear_drag_y, linear_drag_z, linear_drag_yaw,
    quadratic_drag_x, quadratic_drag_y, quadratic_drag_z, quadratic_drag_yaw,
    buoyancy, weight)

    return AUVSimplifiedHydrodynamicsParameters(
        mass, inertia_z,
        added_mass_x, added_mass_y, added_mass_z, added_mass_yaw,
        linear_drag_x, linear_drag_y, linear_drag_z, linear_drag_yaw,
        quadratic_drag_x, quadratic_drag_y, quadratic_drag_z, quadratic_drag_yaw,
        buoyancy, weight
    )
end

#= AUVSimplified flight environment. =#
struct AUVSimplifiedEnvironmentParameters
    g::RealVector          # [m/s^2] Gravity vector
    obs::Vector{Ellipsoid} # Obstacles (ellipsoids)
    n_obs::Int             # Number of obstacles
end

#= Trajectory parameters. =#
mutable struct AUVSimplifiedTrajectoryParameters
    r0::RealVector # Initial position
    rf::RealVector # Terminal position
    v0::RealVector # Initial velocity
    vf::RealVector # Terminal velocity
    tf::Real       # Simplified trajectory finish time
end

#= AUVSimplified trajectory optimization problem parameters all in one. =#
mutable struct AUVSimplifiedProblem
    vehicle::AUVSimplifiedParameters        # The ego-vehicle
    env::AUVSimplifiedEnvironmentParameters # The environment
    traj::AUVSimplifiedTrajectoryParameters # The trajectory
    hydroparams::AUVSimplifiedHydrodynamicsParameters # The hydrodynamics parameters
end

# ..:: Methods ::..

#= Constructor for the environment.

Arguments
    gnrm: gravity vector norm.
    obs: array of obstacles (ellipsoids).

Returns
    env: the environment struct.
=#
function AUVSimplifiedEnvironmentParameters(
    gnrm::Real,
    obs::Vector{Ellipsoid})::AUVSimplifiedEnvironmentParameters

    # Derived values
    g = zeros(3)
    g[end] = -gnrm
    n_obs = length(obs)

    env = AUVSimplifiedEnvironmentParameters(g, obs, n_obs)

    return env
end

#= Constructor for the AUVSimplified landing flip maneuver problem.

Returns:
    mdl: the problem definition object. =#
function AUVSimplifiedProblem()::AUVSimplifiedProblem

    # >> Environment <<
    g = 9.81
    obs = Ellipsoid[] # No obstacles
    env = AUVSimplifiedEnvironmentParameters(g, obs)


    # >> AUVSimplified <<
    # Indices
    id_r = 1:4 # Position indices of the state vector (x, y, z, yaw)
    id_v = 5:8 # Velocity indices of the state vector (vx, vy, vz, yaw_rate)
    id_u = 1:4 # Indices of the thrust input vector (taux, tauy, tauz, tauyaw)
    id_e = 9 # Index of energy state
    # Mechanical parameters
    u_max = 50      # [N] Maximum thrust)
    u_min = -40     # [N] Minimum thrust
    v_max = 1.0     # [m/s] Maximum velocity
    v_min = -1.0    # [m/s] Minimum velocity
    v_min_u = -0.01 # [m/s] Minimum forward velocity (no reverse)
    thruster_allocation_matrix = [
        -0.707  -0.707   0.707   0.707   0.0   0.0;
         0.707  -0.707   0.707  -0.707   0.0   0.0;
         0.0     0.0     0.0     0.0     1.0   1.0;
         0.1888 -0.1888 -0.1888  0.1888  0.0   0.0
    ]

    auv_simpl = AUVSimplifiedParameters(id_r, 
                        id_v, 
                        id_e, 
                        id_u, 
                        u_max, 
                        u_min, 
                        v_max, 
                        v_min, 
                        v_min_u, 
                        thruster_allocation_matrix)

    # >> Trajectory <<
    # Initial values
    r0 = zeros(4)   # Initial position (x, y, z, ψ) [m,m,m,rad]
    r0[1] = -1.0    # x init
    r0[2] = 0.0     # y init
    r0[3] = -0.5    # z init
    r0[4] = 0.0     # ψ init
    v0 = zeros(4) # Initial velocity (u, v, w, ψ_rate) [m/s, m/s, m/s, rad/s]
    
    # Terminal values
    rf = zeros(4)   # Terminal position (x, y, z, ψ) [m,m,m,rad]
    rf[1] = 2.5     # x final
    rf[2] = 6.0     # y final
    rf[3] = -5.0    # z final
    rf[4] = 0.0     # ψ final
    vf = zeros(4) # Terminal velocity (u, v, w, ψ_rate) [m/s, m/s, m/s, rad/s]
    tf = 50.0

    traj = AUVSimplifiedTrajectoryParameters(r0, 
                                    rf, 
                                    v0, 
                                    vf,
                                    tf)

    # >> Hydrodynamics << from https://flex.flinders.edu.au/file/27aa0064-9de2-441c-8a17-655405d5fc2e/1/ThesisWu2018.pdf
    hydro = AUVSimplifiedHydrodynamicsParameters(
        mass=11.5, # [kg] Mass of the vehicle
        inertia_z=0.16, # [kg*m^2] Yaw moment of inertia
        added_mass_x=-5.5, # [kg] Added mass in surge
        added_mass_y=-12.7, # [kg] Added mass in sway
        added_mass_z=-14.57, # [kg] Added mass in heave
        added_mass_yaw=-0.12, # [kg*m^2/rad] Added mass in yaw
        linear_drag_x=-4.03, # [Ns/m] Linear drag in surge
        linear_drag_y=-6.22, # [Ns/m] Linear drag in sway
        linear_drag_z=-5.18, # [Ns/m] Linear drag in heave
        linear_drag_yaw=-0.07, # [Ns/rad] Linear drag in yaw
        quadratic_drag_x=-18.18, # [Ns²/m²] Quadratic drag in surge
        quadratic_drag_y=-21.66, # [Ns²/m²] Quadratic drag in sway
        quadratic_drag_z=-36.99, # [Ns²/m²] Quadratic drag in heave
        quadratic_drag_yaw=-1.55, # [Ns²/rad²] Quadratic drag in yaw
        buoyancy=114.8, # [N] buoyancy force (buoyancy = mass*gravity)
        weight=112.8)  # [N] Weight force (weight = mass*gravity)

    mdl = AUVSimplifiedProblem(auv_simpl, env, traj, hydro)

    return mdl
end
