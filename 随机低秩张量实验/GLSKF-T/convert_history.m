function history = convert_history(raw_history, data_name, seed, missing_rate, rg, rho, gamma)
n = height(raw_history);
iteration = raw_history.iteration;
elapsed_time_seconds = raw_history.elapsed_time_seconds;
relative_change = raw_history.relative_change;

if ismember('MSE', raw_history.Properties.VariableNames)
    MSE = raw_history.MSE;
    RMSE = raw_history.RMSE;
else
    MSE = NaN(n, 1);
    RMSE = NaN(n, 1);
end

RSE = NaN(n, 1);
MAE = NaN(n, 1);
dataset = repmat({data_name}, n, 1);
method = repmat({'GLSKF-T'}, n, 1);
variant = repmat({'observed_global_update'}, n, 1);
seed_col = repmat(seed, n, 1);
missing_col = repmat(missing_rate, n, 1);
parameter_settings = repmat({sprintf('rg=%g, rho=%g, gamma=%g', rg, rho, gamma)}, n, 1);
convergence_status = repmat({'ok'}, n, 1);

history = table(dataset, method, variant, seed_col, missing_col, iteration, ...
    elapsed_time_seconds, MSE, RMSE, RSE, MAE, relative_change, ...
    parameter_settings, convergence_status);
history.Properties.VariableNames = {'dataset', 'method', 'variant', 'seed', ...
    'missing_rate', 'iteration', 'elapsed_time_seconds', 'MSE', 'RMSE', ...
    'RSE', 'MAE', 'relative_change', 'parameter_settings', 'convergence_status'};
end
