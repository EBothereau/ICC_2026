#Code proposed By Emma Bothereau

using Serialization
using GLMakie
using PyCall
using Random
using MultivariateStats
using HDF5
using DSP, FFTW, Statistics
using NearestNeighbors
using GaussianMixtures
using Flux


# =============================================================================
# Configuration
# =============================================================================

Base.@kwdef struct Config
    seed::Int = 42
    n_features::Int = 400            #f = 400 default
    lda_dimension::Int = 5           #p = 5 default
    n_estimators::Int = 50
    n_enrollment::Int = 100
    n_test::Int = 100
    knn_neighbors::Int = 15
    device_threshold::Int = 40
    classifier::String = "GMM"        #GMM or kNN 
    method::String = "RogueDetection" #RogueDetection or Authentication
    train_features::Bool = false
    training_path::String = "/Users/ebothere/Documents/Code/These_Emma/Dataset/LoRa_RFFI/dataset/Train/dataset_training_no_aug.h5"
    enrollment_path::String = "/Users/ebothere/Documents/Code/These_Emma/Dataset/LoRa_RFFI/dataset/Test/dataset_residential.h5"
    authentication_path::String = "/Users/ebothere/Documents/Code/These_Emma/Dataset/LoRa_RFFI/dataset/Test/channel_problem/A.h5"
    rogue_path::String = "/Users/ebothere/Documents/Code/These_Emma/Dataset/LoRa_RFFI/dataset/Test/dataset_rogue.h5"
    feature_file::String = "selected_indexes.bin"
    lda_file::String = "lda_proj.bin"
    roc_file::String = "ROCcurve.png"
end

function initialize(config)
    Random.seed!(config.seed)
    np = pyimport("numpy")
    np.random.seed(config.seed)
end


# =============================================================================
# Data loading
# =============================================================================



function loadBDD(Path)
    ratio = 1
    raw_hdf_data = h5open(Path, "r")  
    obj = raw_hdf_data["data"]
    Xtemp = HDF5.read(obj)
    obj = raw_hdf_data["label"]
    Y = Int.(HDF5.read(obj))
    X = zeros(Int(size(Xtemp)[1]/2),2,size(Xtemp)[end])
    X[:,1,:] = Xtemp[1:8192,:]
    X[:,2,:] = Xtemp[8193:end,:]
    return X[:,1,:]+1*im*X[:,2,:], Y
end


function compute_auc(fpr, tpr)
    sort_idx = sortperm(fpr)
    fpr_sorted = fpr[sort_idx]
    tpr_sorted = tpr[sort_idx]
    auc = sum(diff(fpr_sorted) .* (tpr_sorted[1:end-1] + tpr_sorted[2:end]) / 2)
    return auc
end

# EER : trouver l'endroit où FPR ≈ 1 - TPR
function compute_eer(fpr, tpr)
    fnr = 1 .- tpr
    differences = abs.(fpr .- fnr)
    idx = argmin(differences)
    eer = (fpr[idx] + fnr[idx]) / 2
    return eer
end

#----------------------------------------------------------------------------------------------------------------------------------------
#
#                       Preprocessing Instantaneous Phase Difference (Emma)
#
#----------------------------------------------------------------------------------------------------------------------------------------



function moving_average(x, window_size)
    n = length(x)
    y = similar(x)
    half = div(window_size, 2)
    for i in 1:n
        from = max(1, i - half)
        to = min(n, i + half)
        y[i] = mean(x[from:to])
    end
    return y
end


function processBDD(X,Y)
    Xtemp = []

   for i in 1:size(X, 2)
        phase_diff = diff(unwrap(angle.(X[:, i])))
        meansig = moving_average(phase_diff,4)
        p1 = minimum(meansig)
        p99 = maximum(meansig)
        phase_diff .-= p1
        phase_diff ./= (p99-p1)
        phase_diff .-= 0.5
        phase_diff = abs.(rfft(phase_diff))
        push!(Xtemp,phase_diff)
    end    
    return hcat(Xtemp...), Y[:,1]
end



# =============================================================================
# Evaluation
# =============================================================================

function compute_auc(fpr, tpr)
    order = sortperm(fpr)
    fpr = fpr[order]
    tpr = tpr[order]

    return sum(
        diff(fpr) .* (tpr[1:end-1] + tpr[2:end]) ./ 2
    )
end

function compute_eer(fpr, tpr)
    fnr = 1 .- tpr
    index = argmin(abs.(fpr .- fnr))
    return (fpr[index] + fnr[index]) / 2
end


function select_signals(labels, n; start=1)
    indices = Int[]
    for class in unique(labels)
        class_indices = findall(==(class), labels[:, 1])
        first = start
        last = min(start + n - 1, length(class_indices))
        first <= last && append!(indices, class_indices[first:last])
    end
    return indices
end

function preprocess(X, Y, selected_features, lda; normalize_labels=false)
    normalize_labels && (Y = Y .- minimum(Y) .+ 1)
    X, labels = processBDD(X, Y)
    X = X[selected_features, :]
    return MultivariateStats.predict(lda, X), labels
end

function load_data(path, selected_features, lda; indices=nothing, normalize_labels=false)
    X, Y = loadBDD(path)
    if indices !== nothing
        X, Y = X[:, indices], Y[indices]
    end
    return preprocess(X, Y, selected_features, lda; normalize_labels)
end

# =============================================================================
# Feature extraction
# =============================================================================

function train_feature_extractor(config)
    X, Y = loadBDD(config.training_path)
    X, labels = processBDD(X, Y)

    ensemble = pyimport("sklearn.ensemble")
    classifier = ensemble.ExtraTreesClassifier(
        n_estimators=config.n_estimators,
        random_state=config.seed
    )
    classifier.fit(PyCall.PyObject(X'), PyCall.PyObject(labels))

    feature_order = sortperm(classifier.feature_importances_, rev=true)
    selected_features = feature_order[1:config.n_features]
    serialize(config.feature_file, selected_features)

    lda = MultivariateStats.fit(
        MulticlassLDA,
        X[selected_features, :],
        labels;
        outdim=config.lda_dimension
    )
    serialize(config.lda_file, lda)

    return selected_features, lda
end

function load_feature_extractor(config)
    return deserialize(config.feature_file), deserialize(config.lda_file)
end

function get_feature_extractor(config)
    return config.train_features ?
        train_feature_extractor(config) :
        load_feature_extractor(config)
end

# =============================================================================
# Classifiers
# =============================================================================

function train_gmm(X, labels)
    models = Dict{Int,GMM}()
    for class in sort(unique(labels))
        data = permutedims(X[:, labels .== class])
        models[class] = GMM(1, data, method=:kmeans)
    end
    return models
end

function train_classifier(X, labels, config)
    if config.classifier == "GMM"
        return train_gmm(X, labels)
    elseif config.classifier == "kNN"
        return KDTree(X)
    else
        error("Unknown classifier: $(config.classifier)")
    end
end

function gmm_likelihoods(X, models)
    data = permutedims(X)
    labels = sort(collect(keys(models)))
    scores = hcat([
        maximum(llpg(models[label], data), dims=2)[:, 1]
        for label in labels
    ]...)
    return labels, scores
end

function gmm_scores(X, models)
    _, scores = gmm_likelihoods(X, models)
    return maximum(scores, dims=2)[:, 1]
end

function knn_scores(X, tree, k)
    scores = Float64[]
    for i in axes(X, 2)
        _, distances = knn(tree, X[:, i], k, true)
        push!(scores, mean(distances))
    end
    return scores
end

# =============================================================================
# Enrollment
# =============================================================================

function enrollment(config, selected_features, lda)
    X, Y = loadBDD(config.enrollment_path)
    indices = select_signals(Y, config.n_enrollment)
    X, labels = preprocess(
        X[:, indices],
        Y[indices],
        selected_features,
        lda;
        normalize_labels=config.method != "RogueDetection"
    )
    classifier = train_classifier(X, labels, config)
    return (model=classifier, labels=labels)
end

# =============================================================================
# Test data
# =============================================================================

function load_test_data(config)
    if config.method == "RogueDetection"
        X1, Y1 = loadBDD(config.enrollment_path)
        X2, Y2 = loadBDD(config.rogue_path)
        return cat(X1, X2, dims=2), cat(Y1, Y2, dims=1)
    elseif config.method == "Authentication"
        X, Y = loadBDD(config.authentication_path)
        return X, Y .- minimum(Y) .+ 1
    else
        error("Unknown method: $(config.method)")
    end
end

function test_data(config, selected_features, lda)
    X, Y = load_test_data(config)
    indices = select_signals(Y, config.n_test; start=config.n_test + 1)
    return preprocess(
        X[:, indices],
        Y[indices],
        selected_features,
        lda
    )
end

# =============================================================================
# Authentication
# =============================================================================

function predict_knn(X, tree, labels, k)
    predictions = Int[]
    for i in axes(X, 2)
        indices, distances = knn(tree, X[:, i], k, true)
        votes = Dict{Int,Float64}()
        for (index, distance) in zip(indices, distances)
            label = labels[index]
            votes[label] = get(votes, label, 0.0) + 1 / (distance + eps())
        end
        push!(predictions, argmax(votes))
    end
    return predictions
end

function predict_gmm(X, models)
    labels, scores = gmm_likelihoods(X, models)
    return [labels[argmax(scores[i, :])] for i in axes(scores, 1)]
end

function authenticate(X, labels, enrollment, config)
    predictions = config.classifier == "kNN" ?
        predict_knn(X, enrollment.model, enrollment.labels, config.knn_neighbors) :
        predict_gmm(X, enrollment.model)
    accuracy = 100 * mean(predictions .== labels)
    println("Accuracy = $(round(accuracy, digits=2)) %")
    return predictions, accuracy
end

# =============================================================================
# Rogue detection
# =============================================================================

function roc_curve(scores, labels, threshold, classifier)
    auth = scores[labels .<= threshold]
    rogue = scores[labels .> threshold]

    thresholds = range(minimum(scores), maximum(scores), length=100)
    tpr, fpr = Float64[], Float64[]

    for value in thresholds
        if classifier == "GMM"
            # Score élevé = authentique
            tp = sum(auth .>= value)
            fn = sum(auth .< value)
            fp = sum(rogue .>= value)
            tn = sum(rogue .< value)
        else
            # Score faible = authentique (distance kNN)
            tp = sum(auth .<= value)
            fn = sum(auth .> value)
            fp = sum(rogue .<= value)
            tn = sum(rogue .> value)
        end

        push!(tpr, tp / max(tp + fn, 1))
        push!(fpr, fp / max(fp + tn, 1))
    end

    return fpr, tpr
end

function detect_rogues(X, labels, enrollment, config)
    scores = config.classifier == "GMM" ?
        gmm_scores(X, enrollment.model) :
        knn_scores(X, enrollment.model, config.knn_neighbors)

    fpr, tpr = roc_curve(scores, labels, config.device_threshold,config.classifier)
    auc = compute_auc(fpr, tpr)
    eer = compute_eer(fpr, tpr)

    println("AUC = $(round(auc, digits=4))")
    println("EER = $(round(eer, digits=4))")

    plot_roc(fpr, tpr, config.roc_file)
    return (scores=scores, fpr=fpr, tpr=tpr, auc=auc, eer=eer)
end

function plot_roc(fpr, tpr, filename)
    fig = Figure()
    ax = Axis(
        fig[1, 1],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC Curve"
    )
    lines!(ax, fpr, tpr, label="ROC")
    lines!(ax, [0, 1], [0, 1], linestyle=:dash, color=:black, label="Random")
    axislegend(ax, position=:rb)
    save(filename, fig)
    display(fig)
end

# =============================================================================
# Main
# =============================================================================

function main()
    config = Config()
    initialize(config)

    #Either train or load the feature extractor
    selected_features, lda = get_feature_extractor(config)

    #Enrollment
    enrollment_data = enrollment(config, selected_features, lda)

    #Datatest load
    X_test, labels_test = test_data(config, selected_features, lda)

    #evaluation
    if config.method == "Authentication"
        return authenticate(X_test, labels_test, enrollment_data, config)
    elseif config.method == "RogueDetection"
        return detect_rogues(X_test, labels_test, enrollment_data, config)
    else
        error("Unknown method: $(config.method)")
    end
end

results = main()
