using Statistics

function log_mean(log_x)
    mx = maximum(log_x)
    log(mean(exp.(log_x .- mx))) + mx
end
function relative_std(log_x, log_m=log_mean(log_x))
    std(exp.(log_x .- log_m), mean=1.0)
end

function norm_const_relative_std(L; global_mean=true)
    n_Δs = size(L,2)
    n_res = size(L,1)
    s_rel = zeros(n_Δs, n_res)
    for j = 1:n_Δs
        if global_mean
            Lm = log_mean(L[:,j,:]) # All are unbiased, use global mean
            s_rel[j,:] = mapslices(x -> relative_std(x, Lm), L[:,j,:], dims=2)
        else
            for i = 1:n_res
                s_rel[j,i] = relative_std( L[i,j,:])
            end
        end
    end
    s_rel
end

function norm_const_relative_std_pooled(L; global_mean=true, 
    means=nothing)
    n_Δs = size(L,2)
    n_res = size(L,1)
    n_Ns = size(L,4)
    s_rel = zeros(n_Δs, n_res, n_Ns)
    for j = 1:n_Δs
        if global_mean
            if isnothing(means)
                Lm = log_mean(L[:,j,:,:]) # All are unbiased, use global mean
            else
                Lm = means[j]
            end
            for i = 1:n_Ns
                s_rel[j,:,i] = mapslices(x -> relative_std(x, Lm), L[:,j,:,i], dims=2)
            end
        else
            for k = 1:n_res
                for i = 1:n_Ns
                    s_rel[j,k,i] = relative_std( L[k,j,:,i])
                end
            end
        end
    end
    s_rel
end

function show_norm_const_est(Δs, s; col=col, ind=1:size(s)[2], yaxis=:log,
    ylabel = "log(Std.)")
    plot(log2.(Δs), s[:,ind], labels=lab[:,ind], linewidth=2, legend=:topright,
           xlabel="log2 Δ",  ylabel=ylabel, yaxis=yaxis, xticks=log2.(Δs), c=col[:,ind])
end

function show_norm_const_est_styles(Δs, s, styles; res=AllResmaplings, yaxis=:log,
    ylabel = "log(Std.)", legend=:topright, xlabel="log2 Δ", xtrans=log2)

    p = plot(; legend=legend,linewidth=2,xlabel=xlabel,  ylabel=ylabel,
    yaxis=yaxis, xticks=xtrans.(Δs))

    lab = keys(res)
    for (i,r) in enumerate(res)
        key = lab[i]
        if key ∉ keys(styles) 
            continue
        end
        st = styles[key]
        plot!(p,xtrans.(Δs), s[:,i], label=String(key), c=st.c, markershape=st.marker)
    end
    p
end

function smoothing_mse(X, L)
    n_res, n_Δs, rep, d_f = size(X)
    E = similar(X);   S = zeros(n_Δs, d_f)
    err = similar(E); s_filt = zeros(n_res, n_Δs, d_f)
    for j = 1:n_Δs
        Lm = log_mean(L[:,j,:]) # All are unbiased, use global mean
        for i = 1:n_res
            for k = 1:rep
                # Calculate unbiased estimators (weight corrected)
                E[i,j,k,:] .= X[i,j,k,:] * exp(L[i,j,k]-Lm)
            end
        end
        # The "ground truth":
        S[j,:] = mapslices(mean, E[:,j,:,:], dims=(1,2))
        for i = 1:n_res
            for k = 1:rep
                # Calculate unbiased estimators (weight corrected)
                err[i,j,k,:] = (E[i,j,k,:] - S[j,:]).^2 
            end
        end   
        for i = 1:n_res
            s_filt[i, j, :] = mapslices(mean, err[i,j,:,:], dims=1)
        end
    end
    s_filt
end

function smoothing_mse_pooled(X, L)
    n_res, n_Δs, rep, d_f, n_Ns = size(X)
    E = similar(X);   S = zeros(n_Δs, d_f)
    err = similar(E); s_filt = zeros(n_res, n_Δs, d_f, n_Ns)
    for j = 1:n_Δs
        Lm = log_mean(L[:,j,:,:]) # All are unbiased, use global mean
        for i = 1:n_res
            for k = 1:rep
                for l = 1:n_Ns
                    # Calculate unbiased estimators (weight corrected)
                    E[i,j,k,:,l] .= X[i,j,k,:,l] * exp(L[i,j,k,l]-Lm)
                end
            end
        end
        # The "ground truth":
        S[j,:] = mapslices(mean, E[:,j,:,:,:], dims=(1,2,4))
        for i = 1:n_res
            for k = 1:rep
                for l = 1:n_Ns
                    # Calculate errors
                    err[i,j,k,:,l] = (E[i,j,k,:,l] - S[j,:]).^2 
                end
            end
        end   
        for i = 1:n_res
            for l = 1:n_Ns
                s_filt[i, j, :, l] = mapslices(mean, err[i,j,:,:,l], dims=1)
            end
        end
    end
    s_filt
end

function show_smoothing_mse(Δs, s_filt, k; col=col, ind=1:size(s_filt)[1], yaxis=:log)
    plot(log2.(Δs), s_filt[ind,:,k]', labels=lab[:,ind], xlabel="log2 Δ", yaxis=yaxis, xticks=log2.(Δs),linewidth=2,
        title="Forward smoothing MSE, t=$(k-1)", c=col[:,ind])
end

function twinplot(p1, p2; widths=(0.5, 0.5))
    plot(p1,p2, layout=grid(1,2,widths=widths), size=(800,300), left_margin=4Plots.mm, bottom_margin=4Plots.mm)
end

function horizplot(ps...; size=(800,300), trim=4Plots.mm,
    widths=(1/length(ps) for i=1:length(ps)))
    plot(ps..., layout=grid(1,length(ps),widths=widths), size=size, left_margin=trim, bottom_margin=trim)
end
