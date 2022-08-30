import Zygote
import ShapML
import Plots
import Flux
import CSV

using Statistics
using DataFrames
using DeepCompartmentModels

population = load("./data/warfarin.csv", [:AGE, :SEX], DVID=1)
train_population, test_population = create_split(population; ratio=0.75)

# DEEP COMPARTMENT MODEL
function two_comp_abs!(dA, A, p, t)
    kₐ, k₂₀, k₂₃, k₃₂, D = p

    dA[1] = D - kₐ * A[1]
    dA[2] = kₐ * A[1] + A[3] * k₃₂ - A[2] * (k₂₀ + k₂₃)
    dA[3] = A[2] * k₂₃ - A[3] * k₃₂
end

ann = Flux.Chain(
    Flux.Dense(size(population.x, 2), 16, Flux.swish),
    Flux.Dense(16, 4, Flux.softplus)
)

model = DCM(two_comp_abs!, ann, 3; measurement_compartment=2)
optimizer = Flux.ADAM(1e-2)

my_callback(loss, epoch) = println("Epoch $epoch, training set rmse: $(sqrt(loss)), test set rmse: $(sqrt(mse(model, test_population)))")

fit!(model, train_population, optimizer; iterations=500, callback=my_callback)

Plots.plot(predict(model, test_population[1]; interpolating=true), linewidth=1.6)
Plots.scatter!(test_population[1].t, test_population[1].y, color=:black, markershape=:star6)


# RUN SHAP
i = 1
pred(model, data) = DataFrame(y_pred=model(Matrix(data))[i, :])
shap = ShapML.shap(explain=DataFrame(population.x, [:Age, :Sex]), model=model, predict_function=pred)

age_on_cl = shap[shap.feature_name .== "Age", [:feature_value, :shap_effect]]
sex_on_cl = shap[shap.feature_name .== "Sex", [:feature_value, :shap_effect]]
age_on_cl[!, :sex] = Int.(sex_on_cl.feature_value .== 1.)
age_on_cl[!, :feature_value] = normalize_inv(age_on_cl[:, :feature_value], (21, 63))


Plots.plot(normalize_inv(-0.5:0.01:1.5, (21, 63)), model(hcat(collect(-0.5:0.01:1.5), zeros(length(-0.5:0.01:1.5))))[i, :] .- 0.17, linewidth=1.6, color=:pink, label="Neural network prediction for females")
Plots.plot!(normalize_inv(-0.5:0.01:1.5, (21, 63)), model(hcat(collect(-0.5:0.01:1.5), ones(length(-0.5:0.01:1.5))))[i, :] .- 0.14, linewidth=1.6, color=:dodgerblue, label="Neural network prediction for males")
Plots.scatter!(normalize_inv(age_on_cl.feature_value[sex_on_cl.feature_value.==0], (21, 63)), age_on_cl.shap_effect[sex_on_cl.feature_value.==0], color=:pink, label="SHAP values females")
Plots.scatter!(normalize_inv(age_on_cl.feature_value[sex_on_cl.feature_value.==1], (21, 63)), age_on_cl.shap_effect[sex_on_cl.feature_value.==1], color=:dodgerblue, label="SHAP values males")
Plots.plot!(xlabel="Age in years", ylabel="Absorption rate (mg/h)")

sim_age = -0.5:0.01:1.5
sim_age_norm = normalize_inv(-0.5:0.01:1.5, (21, 63))

res = hcat(sim_age_norm, model(hcat(collect(-0.5:0.01:1.5), ones(length(-0.5:0.01:1.5))))[i, :], model(hcat(collect(-0.5:0.01:1.5), zeros(length(-0.5:0.01:1.5))))[i, :])
CSV.write("data/dcm_relationships.csv", DataFrame(res, [:Age, :True_Male, :True_Female]))
CSV.write("data/shap_relationships.csv", age_on_cl)

