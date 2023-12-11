using CSV
using DataFrames
using RData
using Turing
using Distributions
# using Optimization
# using OptimizationOptimJL, OptimizationOptimisers
using Optim
using Memoization
using Zygote
using ReverseDiff
using FiniteDiff
using FileIO
using StatsPlots

#Gompertz model
@model function gompertz(y, pr_type, ::Type{T} = Float64) where {T}
    if pr_type == 0
        #these priors work
        alpha ~ Uniform(-5,5)
        beta ~ Uniform(-1,1)
        sigma ~ Uniform(0.01,1)
        tau ~ Uniform(0.01,1)
    else
        alpha ~ Normal(0,5)
        beta ~ Normal(0,1)
        sigma ~ InverseGamma(0.01, 0.01)
        tau ~ InverseGamma(0.01, 0.01)
    end
    #Unifrom(-10,10) prior works with pr_type=0
    u_init ~ Uniform(-10,10) #not the ideal way to specify an improper prior?
    n = size(y)[1]
    umed = Vector{T}(undef, n)
    u = Vector{T}(undef, n)
    umed[1] = u_init
    u[1] ~ Normal(umed[1], sigma)
    y[1] ~ Normal(u[1], tau)
    for t in 2:n
        umed[t] = alpha + beta*u[t-1]
        u[t] ~ Normal(umed[t], sigma)
        y[t] ~ Normal(u[t], tau)
    end
end


#Logistic growth model
@model function logistic(y, pr_type, ::Type{T} = Float64) where {T}
    if pr_type == 0
        r ~ Uniform(0.01,5)
        K ~ Uniform(0.01,1000)
        sigma ~ Uniform(0.0001,1)
        tau ~ Uniform(0.0001,1)
    else
        r ~ LogNormal(-1,4)
        K ~ LogNormal(5,4)
        sigma ~ Exponential(0.1)
        tau ~ Exponential(0.1)
    end
    u_init ~ Uniform(0.01,100)
    n = size(y)[1]
    umed = Vector{T}(undef, n)
    u = Vector{T}(undef, n)
    umed[1] = u_init
    u[1] ~ LogNormal(umed[1], sigma)
    y[1] ~ LogNormal(log(u[1]), tau)
    for t in 2:n
        umed[t] = u[t-1] + r*u[t-1]*(1-u[t-1]/K)
        u[t] ~ LogNormal(log(umed[t]), sigma)
        y[t] ~ LogNormal(log(u[t]), tau)
    end
end


#read in csv file: requires CSV and DataFrames
datadir = joinpath(@__DIR__, "..", "..", "data")
df = DataFrame(CSV.File(joinpath(datadir, "gompertz", "gompertz_n128.csv")))
# . syntax extracts column from dataframe and casts as vector
gompertzDat = df.y
gompertzInits = load(joinpath(datadir, "gompertz", "gompertzInits_n128.RData"))
plot(gompertzDat)

#read in csv file: requires CSV and DataFrames
df = DataFrame(CSV.File(joinpath(datadir, "logistic", "logistic_n128.csv")))
# . syntax extracts column from dataframe and casts as vector
logisticDat = df.y
logisticInits = load(joinpath(datadir, "logistic", "logisticInits_n128.RData"))


model = gompertz(gompertzDat, 0)
# Check model for type-stability: see tips at
# https://turing.ml/dev/docs/using-turing/performancetips#make-your-model-type-stable
@code_warntype model.f(
    model,
    Turing.VarInfo(model),
    Turing.DefaultContext(),
    model.args...,
)


#Default Forward mode, not timed but takes long time (~1hr)
#TrackerAD: 1791 seconds - now 754 sec using 1000 burn-in and 0.8 adaptive sampling
#Zygote: didn't work: mutating arrays not supported
#ReverseDiff: 505 sec using 1000 burn-in and 0.8 adaptive sampling
#NUTS() working but not same as default STAN/tmbstan settings (2000,0.8)
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)
mle_estimate = optimize(model, MLE(), BFGS())
map_estimate = optimize(model, MAP(), BFGS())
H = FiniteDiff.finite_difference_hessian(u -> mle_estimate.f(u), vec(mle_estimate.values))


@time sampNUTSgompertzP0 = sample(model, NUTS(), MCMCThreads(), 2000, 4,
    init_theta = gompertzInits)
# 102 s first time, 73 s after compilation
describe(sampNUTSgompertzP0)

using MonteCarloMeasurements
ribbonplot([Particles(collect(col)) for col in  eachcol(Array(group(sampNUTSgompertzP0, :u)))],
    label="Gompertz P0")

CSV.write(joinpath(@__DIR__, "MCMCgompertzP0.csv"), sampNUTSgompertzP0)

model1 = gompertz(gompertzDat, 1)
@time sampNUTSgompertzP1 = sample(model1, NUTS(1000,0.8), 2000, nchains = 4,
    init_theta = gompertzInits)
# 90 s
describe(sampNUTSgompertzP1)
ribbonplot!([Particles(collect(col)) for col in  eachcol(Array(group(sampNUTSgompertzP1, :u)))],
    label="Gompertz P1")
plot!(mle_estimate.values[6:end], label="MLE")
plot!(map_estimate.values[6:end], label="MAP")

alpha = mean(sampNUTSgompertzP0[:alpha])
beta = mean(sampNUTSgompertzP0[:beta])
sigma = mean(sampNUTSgompertzP0[:sigma])
tau = mean(sampNUTSgompertzP0[:tau])
CSV.write(joinpath(@__DIR__, "MCMCgompertzP1.csv"), sampNUTSgompertzP1)
params = summarystats(sampNUTSgompertzP0)


sampNUTSlogistic = sample(logistic(logisticDat, 0), NUTS(1000, 0.8), 1000)
