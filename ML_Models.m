%% ================================================================
%   FULL MODEL PIPELINE: SVM, RFR, GPR
%   - 10-fold CV (RMSE + R2 tablosu)
%   - Train/Test split (RMSE + R2 tablosu)
%   - Advanced Hyperparameter Optimization
%   - Model Saving / Loading
% ================================================================

clc; clear; close all;

%% -------------------- VERI SETI --------------------
load Data_set.mat
data = Whole_data;

X = data(:, [1 3]);   % Features
Y = data(:, 2);       % Target


%% ================================================================
%                 TRAIN-TEST SPLIT (%80-%20)
% ================================================================
rng(42)
idx = randperm(size(X,1));
trainN = round(0.70 * length(idx));

X_train = X(idx(1:trainN),:);
Y_train = Y(idx(1:trainN));
X_test  = X(idx(trainN+1:end),:);
Y_test  = Y(idx(trainN+1:end));

%% ================================================================
%            ADVANCED HYPERPARAMETER OPTIMIZATION
% ================================================================

fprintf("\n===== ADVANCED SVM OPTIMIZATION =====\n");

svm_opt = fitrsvm(X_train, Y_train, ...
    "KernelFunction", "rbf", ...
    "Standardize", true, ...
    "OptimizeHyperparameters", {'BoxConstraint','KernelScale','Epsilon'}, ...
    "HyperparameterOptimizationOptions", struct( 'kfold',15, ...
        "AcquisitionFunctionName","expected-improvement-plus", ...
        "MaxObjectiveEvaluations",100, ...
        "ShowPlots", false ...
    ));

fprintf("✓ SVM optimization complete.\n");

%% RFR
fprintf("\n===== ADVANCED RFR OPTIMIZATION =====\n");

rfr_opt = fitrensemble(X_train, Y_train, ...
    "Method","Bag", ...
    "OptimizeHyperparameters", {'NumLearningCycles','MinLeafSize','MaxNumSplits'}, ...
    "HyperparameterOptimizationOptions", struct('kfold',15, ...
        "AcquisitionFunctionName","expected-improvement-plus", ...
        "MaxObjectiveEvaluations",100, ...
        "ShowPlots", false ...
    ));

fprintf("✓ RFR optimization complete.\n");

%% GPR
%% ================================================================

fprintf("\n===== ADVANCED GPR OPTIMIZATION =====\n");

gpr_opt = fitrgp(X_train, Y_train, ...
    "KernelFunction","matern52", ...
    "Standardize", true, ...
    "OptimizeHyperparameters", {'KernelScale','Sigma','BasisFunction'}, ...
    "HyperparameterOptimizationOptions", struct( 'kfold',15,...
        "AcquisitionFunctionName","expected-improvement-plus", ...
        "MaxObjectiveEvaluations",100, ...
        "ShowPlots", false ...
    ));

fprintf("✓ GPR optimization complete (no errors).\n");

%% ================================================================
%     PERFORMANCE EVALUATION AFTER HYPERPARAMETER OPTIMIZATION
% ================================================================

fprintf("\n===== EVALUATION OF OPTIMIZED MODELS (TRAIN–TEST SPLIT) =====\n");

%% ================================================================
%     PERFORMANCE EVALUATION AFTER HYPERPARAMETER OPTIMIZATION
%     (uses SAME TRAIN/TEST split as earlier)
% ================================================================

fprintf("\n===== EVALUATION OF OPTIMIZED MODELS USING SAME TRAIN–TEST SET =====\n");

%% --------- OPTIMIZED SVM PERFORMANCE ----------
pred_train = predict(svm_opt, X_train);
pred_test  = predict(svm_opt, X_test);

svm_rmse_train = sqrt(mean((pred_train - Y_train).^2));
svm_r2_train   = 1 - sum((pred_train - Y_train).^2) / sum((Y_train - mean(Y_train)).^2);

svm_rmse_test = sqrt(mean((pred_test - Y_test).^2));
svm_r2_test   = 1 - sum((pred_test - Y_test).^2) / sum((Y_test - mean(Y_test)).^2);

%% --------- OPTIMIZED RFR PERFORMANCE ----------
pred_train = predict(rfr_opt, X_train);
pred_test  = predict(rfr_opt, X_test);

rfr_rmse_train = sqrt(mean((pred_train - Y_train).^2));
rfr_r2_train   = 1 - sum((pred_train - Y_train).^2) / sum((Y_train - mean(Y_train)).^2);

rfr_rmse_test = sqrt(mean((pred_test - Y_test).^2));
rfr_r2_test   = 1 - sum((pred_test - Y_test).^2) / sum((Y_test - mean(Y_test)).^2);

%% --------- OPTIMIZED GPR PERFORMANCE ----------
pred_train = predict(gpr_opt, X_train);
pred_test  = predict(gpr_opt, X_test);

gpr_rmse_train = sqrt(mean((pred_train - Y_train).^2));
gpr_r2_train   = 1 - sum((pred_train - Y_train).^2) / sum((Y_train - mean(Y_train)).^2);

gpr_rmse_test = sqrt(mean((pred_test - Y_test).^2));
gpr_r2_test   = 1 - sum((pred_test - Y_test).^2) / sum((Y_test - mean(Y_test)).^2);

%% ---- Table ----
Optimized_Results = table( ...
    [svm_rmse_train; rfr_rmse_train; gpr_rmse_train], ...
    [svm_r2_train;   rfr_r2_train;   gpr_r2_train], ...
    [svm_rmse_test;  rfr_rmse_test;  gpr_rmse_test], ...
    [svm_r2_test;    rfr_r2_test;    gpr_r2_test], ...
    'VariableNames', {'Train_RMSE','Train_R2','Test_RMSE','Test_R2'}, ...
    'RowNames', {'SVM','RFR','GPR'} ...
);

disp("===== OPTIMIZED MODELS PERFORMANCE TABLE =====");
disp(Optimized_Results);

%% ================================================================




