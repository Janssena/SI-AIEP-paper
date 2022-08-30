import Zygote
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

Plots.scatter(test_population.y, map(sol -> sol.u, predict(model, test_population)), color=:lightblue, legend=false)
Plots.plot!([0., 15.], [0., 15.], color=:black)

sol = predict(model, test_population[1]; interpolating=true)
time_dv = zeros((length(sol.t), 2))
time_dv[begin:length(test_population[1].t), :] = hcat(test_population[1].t, test_population[1].y)
res = hcat(sol.t, sol.u, zeros(length(sol.u)), time_dv) 
CSV.write("./data/result_dcm_test_rsme_$(sqrt(mse(model, test_population))).csv", DataFrame(res, [:TIME, :PRED, :PRED_NO_DOSE, :TIME2, :DV]))


# NAIVE NEURAL NETWORK
D_train = map(cb -> first(cb.affect!.rates) / 120., train_population.callbacks)
D_test = map(cb -> first(cb.affect!.rates) / 120., test_population.callbacks)

min = minimum(D_train)
max = maximum(D_train)
D_train = (D_train .- min) ./ (max - min)
D_test = (D_test .- min) ./ (max - min)

x_train = []
for (i, time) in enumerate(train_population.t)
    push!(x_train, hcat(transpose(repeat(vcat(train_population[i].x, D_train[i]), outer=(1, length(train_population[i].t)))), train_population[i].t))
end
x_train = vcat(x_train...)

x_test = []
for (i, time) in enumerate(test_population.t)
    push!(x_test, hcat(transpose(repeat(vcat(test_population[i].x, D_test[i]), outer=(1, length(test_population[i].t)))), test_population[i].t))
end
x_test = vcat(x_test...)

ann2 = Flux.Chain(
    Flux.Dense(4, 16, Flux.swish),
    Flux.Dense(16, 4, Flux.swish),
    Flux.Dense(4, 1, Flux.relu)
)

opt = Flux.ADAM(1e-3)
parameters = Flux.params(ann2)
losss(mod, x, pop) = mean((vcat(pop.y...) - mod(x')[1, :]).^2)

for epoch in 1:2500
    lss, back = Zygote.pullback(() -> losss(ann2, x_train, train_population), parameters)
    println("Epoch $epoch: train set rmse: $(sqrt(lss)), test set rmse: $(sqrt(losss(ann2, x_test, test_population)))")
    grad = back(1.0)
    Flux.update!(opt, parameters, grad)
end

Plots.scatter(vcat(test_population.y...), ann2(x_test')[1, :], color=:lightblue, legend=false)
Plots.plot!([0., 15.], [0., 15.], color=:black)


# Test patient
i = 1
t_plotting = 0:0.5:120
x_plotting = vcat(repeat(x_test[i, 1:3], outer=(1, length(t_plotting))), t_plotting')
x_plotting_no_dose = copy(x_plotting)
x_plotting_no_dose[3, :] .= 0.

Plots.plot(t_plotting, ann2(x_plotting)[1, :], linewidth=1.6, label="Regular dose")
Plots.plot!(t_plotting, ann2(x_plotting_no_dose)[1, :], linewidth=1.6, label="No dose")
Plots.scatter!(test_population[i].t, test_population[i].y, color=:black, markershape=:star6)

CSV.write("data/result_naive_test_rmse_$(sqrt(losss(ann2, x_test, test_population))).csv", DataFrame(hcat(t_plotting, ann2(x_plotting)[1, :], ann2(x_plotting_no_dose)[1, :]), [:TIME, :PRED, :PRED_NO_DOSE]))
# Train patient
i = 5
x_plotting = vcat(repeat(x_train[i, 1:3], outer=(1, length(t_plotting))), t_plotting')

Plots.plot(t_plotting, ann2(x_plotting)[1, :], linewidth=1.6)
Plots.scatter!(train_population[i].t, train_population[i].y, color=:black, markershape=:star6)

