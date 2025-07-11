# Import necessary libraries.
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Perform the square root on the positive part of a number but keep the sign.
def signed_sqrt(x_sq):
    return np.sign(x_sq) * np.sqrt(np.abs(x_sq))


# Perform the square on the positive part of a number but keep the sign.
def signed_square(x):
    return np.sign(x) * x ** 2


# Custom regressor that averages the predictions of multiple regressors.
class AveragingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimators):
        self.estimators = estimators
        self.estimators_ = [estimator[1] for estimator in estimators]  # We initialize using tuples of (name, estimator) pairs, like ensemble_model = AveragingRegressor(estimators=[(f"fold_{i}", model) for i, model in enumerate(fold_models)]).

    def fit(self, X, y=None):
        # Since the regressors are already fitted, we do nothing here.
        return self

    def predict(self, X):
        # Retrieve predictions from each pre-fitted model.
        predictions = np.column_stack([estimator.predict(X) for estimator in self.estimators_])
        # Return the simple average across models.
        return predictions.mean(axis=1)


# Plug-in estimate for the bias of the estimator.
def plug_in_bias(X_hat, X):

    # Ensure X is a fixed, known scalar that we are attempting to estimate.
    if not np.isscalar(X):
        raise ValueError("X must be a scalar.")

    # Return the bias depending on the dimensionality of X_hat.
    if X_hat.ndim == 1:
        return np.mean(X_hat) - X
    else:
        return np.mean(X_hat, axis=0) - X


# Plug-in estimate for the variance of the estimator.
def plug_in_variance(X_hat, bias=False):
    # unbiased is ddof=1 (1/N-1), biased is ddof=0 (1/N)
    if bias:
        ddof = 0
    else:
        ddof = 1
    return np.var(X_hat, ddof=ddof)


# Plug-in estimate for the covariance of the estimator.
def plug_in_covariance(X_hat, bias=False):
    # unbiased is bias=False (1/N-1), biased is bias=True (1/N)
    return np.cov(X_hat, rowvar=False, bias=bias)


def compute_metrics(model, X_test, y_test, var_bias=True, metric='r2_score', return_residuals=False):
    """
    Calculate and return various regression metrics for the given model and test data.
    """

    # Predict on the test set.
    y_pred = model.predict(X_test)

    # Ensure y_test and y_pred are numpy arrays.
    y_pred = np.asarray(y_pred)
    y_test = np.asarray(y_test)
    
    # Ensure y_test and y_pred have the same length.
    if len(y_pred) != len(y_test):
        raise ValueError("y_true and y_pred must have the same length.")
    
    # R^2 score.
    if metric == 'r2_score':
        r2 = r2_score(y_test, y_pred)
    elif metric == 'mean_absolute_error':
        r2 = mean_absolute_error(y_test, y_pred)
    elif metric == 'root_mean_squared_error':
        r2 = root_mean_squared_error(y_test, y_pred)

    # Calculate the residual errors. Note this is different from Wasserman (who is y_test - y_pred) and Wasserman is probably the typical case even though to me the order is a bit unintuitive.
    residuals = y_pred - y_test

    # Bias and squared bias.
    bias = plug_in_bias(residuals, 0)  # Yields same as original, np.mean(y_pred - y_test).
    squared_bias = bias ** 2
    
    # Variance of the predictions.
    variance = plug_in_variance(residuals, bias=var_bias)  # Totally different from original, np.var(y_pred, ddof=1) (different statistic).
    
    # Mean Squared Error (MSE). This is specifically for the MSE of the predictions on the test set because that is what you want to know about a ML model's performance. It is a standard metric that is not an estimate of an unknown quantity like the plug-in biases and variances are. (Of course, it turns out that with the true residual=0, it is equal to the estimate of the residual MSE, which allows the decomposition into squared bias + variance. See notes_on_2025-05-12.lyx.)
    mse = np.mean(residuals ** 2)  # Yields same as original, np.mean((y_pred - y_test) ** 2).

    # Decomposition squared error.
    ensemble_decomposition_squared_error = mse - squared_bias - variance
    
    # If we're using an ensemble model, calculation the various contributions to the ensemble MSE of the component models.
    is_ensemble_model = isinstance(model, AveragingRegressor)
    if is_ensemble_model:

        # Get the predictions for each component model.
        component_residuals = np.column_stack([component_model.predict(X_test) - y_test for component_model in model.estimators_])

        # Calculate the biases and covariance of the component models.
        component_biases = plug_in_bias(component_residuals, 0)
        component_covariance = plug_in_covariance(component_residuals, bias=var_bias)
        if np.isscalar(component_covariance) or component_covariance.shape == ():
            # Make it a 1x1 array for consistency
            component_covariance = np.array([[component_covariance]])

        # Store the number of component models and get the indices of the lower diagonal of an M x M matrix for ease of later implementation.
        M = component_residuals.shape[1]  # Used to be called num_component_models.
        lower_diagonal_indices = np.tril_indices(M, k=-1)

        # Implement the formula (derived in notes_on_2025-05-12.lyx) for the ensemble MSE in terms of the bias and covariance of the component models:
        # \frac{1}{n}\sum_{i=1}^{n}\left(\hat{Y}_{i}-Y_{i}\right)^{2}\approx\left[\overline{\text{bias}_{\epsilon=0}^{2}\left(\hat{\epsilon}_{j}\right)}\right]_{j}-\frac{M-1}{2M}\left[\overline{\left(\text{bias}_{\epsilon=0}\left(\hat{\epsilon}_{k}\right)-\text{bias}_{\epsilon=0}\left(\hat{\epsilon}_{j}\right)\right)^{2}}\right]_{j<k}+\frac{1}{M}\left[\overline{\text{V}\left(\hat{\epsilon}_{j}\right)}\right]_{j}+\frac{M-1}{M}\left[\overline{\text{Cov}\left(\hat{\epsilon}_{j},\hat{\epsilon}_{k}\right)}\right]_{j<k}
        component_contributions_to_ensemble_mse = dict()
        component_contributions_to_ensemble_mse['Non-interaction bias'] = (component_biases ** 2).mean()
        component_contributions_to_ensemble_mse['Interaction bias'] = ((component_biases.reshape(-1, 1) - component_biases.reshape(1, -1)) ** 2)[lower_diagonal_indices].mean() * (1 - M) / (2 * M) if M > 1 else 0
        component_contributions_to_ensemble_mse['Non-interaction variance'] = component_covariance.diagonal().mean() / M
        component_contributions_to_ensemble_mse['Interaction variance'] = component_covariance[lower_diagonal_indices].mean() * (M - 1) / M if M > 1 else 0

        # Component decomposition squared error.
        component_decomposition_squared_error = mse - component_contributions_to_ensemble_mse['Non-interaction bias'] - component_contributions_to_ensemble_mse['Interaction bias'] - component_contributions_to_ensemble_mse['Non-interaction variance'] - component_contributions_to_ensemble_mse['Interaction variance']

    # Return the metrics as a dictionary.
    return {
        'r^2 score': r2,
        'bias': bias,
        'std': np.sqrt(variance),
        'error': np.sqrt(mse),  # This is the RMSE.
        'decomposition sq. error': ensemble_decomposition_squared_error,
        'component non-interaction bias sq. error': component_contributions_to_ensemble_mse['Non-interaction bias'] if is_ensemble_model else None,
        'component interaction bias sq. error': component_contributions_to_ensemble_mse['Interaction bias'] if is_ensemble_model else None,
        'component non-interaction variance sq. error': component_contributions_to_ensemble_mse['Non-interaction variance'] if is_ensemble_model else None,
        'component interaction variance sq. error': component_contributions_to_ensemble_mse['Interaction variance'] if is_ensemble_model else None,
        'component decomposition sq. error': component_decomposition_squared_error if is_ensemble_model else None,
        'residuals': residuals if return_residuals else None,
    }


# Define some common regression models and their hyperparameter grids.
def get_common_models_and_hyperparameter_grids():
    return {
        "Linear Regression": {
            "model": [LinearRegression()],
            "params": {}
        },
        "Ridge Regression": {
            "model": [Ridge()],
            "params": {
                "alpha": [0.1, 1.0, 10.0]
            }
        },
        "Lasso Regression": {
            "model": [Lasso()],
            "params": {
                "alpha": [0.1, 1.0, 10.0]
            }
        },
        "Decision Tree Regressor": {
            "model": [DecisionTreeRegressor()],
            "params": {
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10]
            }
        },
        "Random Forest Regressor": {
            "model": [RandomForestRegressor()],
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10]
            }
        },
        "Gradient Boosting Regressor": {
            "model": [GradientBoostingRegressor()],
            "params": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 10]
            }
        },
        "Support Vector Regressor": {
            "model": [SVR()],
            "params": {
                "C": [0.1, 1.0, 10.0],
                "epsilon": [0.1, 0.2, 0.5],
                "kernel": ["linear", "rbf"]
            }
        },
        "Polynomial Regression (Degree 2)": {
            "model": [PolynomialFeatures(degree=2), StandardScaler(), LinearRegression()],
            "params": {}
        },
        "K-Nearest Neighbors Regressor": {
            "model": [KNeighborsRegressor()],
            "params": {
                "n_neighbors": [3, 5, 10],
                "weights": ["uniform", "distance"]
            }
        },
        "Bayesian Regression": {
            "model": [BayesianRidge()],
            "params": {
                "alpha_1": [1e-6, 1e-5, 1e-4],
                "lambda_1": [1e-6, 1e-5, 1e-4]
            }
        }
    }


# Create a copy of the input dataframes (to avoid modifying the originals) and drop rows where any value in X or y is missing.
def initial_row_filter(X, y, write_func=print, error_func=print):

    # Ensure X and y are independent copies to avoid modifying views.
    X = X.copy()
    y = y.copy()

    # Drop rows where any value in X or y is missing.
    initial_rows = X.shape[0]
    mask = ~X.isnull().any(axis=1) & ~y.isnull()
    X, y = X[mask], y[mask]
    final_rows = X.shape[0]
    dropped_rows = initial_rows - final_rows

    # Display the number of rows before and after dropping missing values.
    write_func(f"Initial rows: {initial_rows}")
    write_func(f"Dropped rows: {dropped_rows}")
    write_func(f"Final rows: {final_rows}")

    # Check if X or y is empty after dropping missing values.
    if X.shape[0] == 0 or y.shape[0] == 0:
        error_func("No data available after dropping rows with missing values. Please check your dataset.")
        return
    
    # Return the filtered X and y.
    return X, y


# Evaluate all the regression models relating the independent and dependent variables.
def fit(X, y, numeric_columns, categorical_columns, model_params=None, apply_pca=True, num_pca_components=0.95, generalization_holdout_size=0.2, num_models=5, num_inner_cv_splits=5, random_state_generalization_split=42, random_state_outer_model_folds=42, random_state_inner_cv_folds=42, num_jobs_for_gridsearchcv=-1, write_func=print, error_func=print, max_one_hot_encoding_categories=100, num_repetitions=1, metric='r2_score', apply_scaling=True):

    # Create copies of the input dataframes (to avoid modifying the originals) and drop rows where any value in X or y is missing.
    X, y = initial_row_filter(X, y, write_func=write_func, error_func=error_func)

    # Use local copies of the numeric and categorical columns.
    numeric_columns = numeric_columns.copy()
    categorical_columns = categorical_columns.copy()

    # Save the columns to enforce order for prediction (can use a ColumnTransformer for this instead).
    feature_columns = X.columns.tolist()

    # Assert mutual exclusivity between numeric and categorical columns.
    if set(numeric_columns).intersection(categorical_columns):
        error_func("Numeric and categorical columns should be mutually exclusive. Please check your dataset.")
        return

    # For categorical variables that are numeric, move them from categorical to numeric columns.
    for col in X.columns:
        if col in categorical_columns and X[col].dtype != 'object':
            numeric_columns.append(col)
            categorical_columns.remove(col)

    # Check for columns with too many unique categories.
    columns_with_many_categories = {}
    for col in X.columns:
        if col in categorical_columns and X[col].dtype == 'object':
            nunique_values_in_column = X[col].nunique()
            if nunique_values_in_column > max_one_hot_encoding_categories:
                columns_with_many_categories[col] = nunique_values_in_column

    # Warn the user if any columns exceed the threshold.
    if columns_with_many_categories:
        warning_message = (
            f"The following categorical columns have more than {max_one_hot_encoding_categories} unique categories and may cause performance issues:\n"
            + "\n".join([f"{col}: {count} categories" for col, count in columns_with_many_categories.items()])
            + f"\nPlease remove these columns from the analysis or talk to Andrew about implementing alternative encoding strategies."
        )
        error_func(warning_message)
        return

    # Create a one-hot encoder for categorical features.
    columns_to_encode = [col for col in X.columns if col in categorical_columns]
    encoder = ColumnTransformer(
        transformers=[('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), columns_to_encode)],
        remainder='passthrough'
    )

    # Define the standard scaler to add to the pipeline if scaling is requested.
    scaler = StandardScaler() if apply_scaling else FunctionTransformer(lambda x: x, validate=False)

    # Define models and hyperparameter grids if not provided.
    if model_params is None:
        model_params = get_common_models_and_hyperparameter_grids()

    # Iterate over the number of times to repeat the full workflow. When seeds are not set, particularly random_state_generalization_split, the results will be different each time.
    repetitions_list = []
    train_test_generalization_indices = []
    num_models_orig = num_models
    for irepetition in range(num_repetitions):
        write_func(f"Running repetition {irepetition + 1} of {num_repetitions}...")

        # Split X and y into nested CV and generalization holdout sets. Doing this using indices as of 5/27/25 in order to save the data for each repetition for later (by saving their indices).
        indices_generalization = np.arange(len(X))
        train_idx_generalization, test_idx_generalization = train_test_split(
            indices_generalization,
            test_size=generalization_holdout_size,
            random_state=random_state_generalization_split
        )
        X_nested_cv = X.iloc[train_idx_generalization]
        X_generalization_holdout = X.iloc[test_idx_generalization]
        y_nested_cv = y.iloc[train_idx_generalization]
        y_generalization_holdout = y.iloc[test_idx_generalization]

        # Since we ultimately just use this to subtract from the output of a pipeline which is a numpy array, convert to numpy.
        y_generalization_holdout = y_generalization_holdout.to_numpy()

        # Output the shapes for verification.
        write_func(f"X_nested_cv shape: {X_nested_cv.shape}, X_generalization_holdout shape: {X_generalization_holdout.shape}")
        write_func(f"y_nested_cv shape: {y_nested_cv.shape}, y_generalization_holdout shape: {y_generalization_holdout.shape}")

        # Iterate over the different numbers of models, i.e., numbers of folds, to run.
        num_models = num_models_orig
        if isinstance(num_models, list):
            num_models_list = num_models
        else:
            num_models_list = [num_models]
        num_models_dict = {}
        for num_models in num_models_list:
            write_func(f"Running with {num_models} models...")

            # Set up data splitting for the outer loop of nested cross-validation in order to optimize actual selection of the model. These are splits into train and test and it's done here to enable the same splits for all model types.
            # Can use binning to use stratified folds for regression problems in the future.
            if num_models == 1:
                # Mimic KFold shuffling
                all_indices = np.arange(len(X_nested_cv))
                np.random.RandomState(random_state_outer_model_folds).shuffle(all_indices)
                train_idx = all_indices
                test_idx = np.array([], dtype=int)
                outer_splits = [(train_idx, test_idx)]
            else:
                outer_model_folds = KFold(n_splits=num_models, shuffle=True, random_state=random_state_outer_model_folds)
                outer_splits = list(outer_model_folds.split(X_nested_cv, y_nested_cv))  # Store splits as a list of (train_idx, test_idx)

            # Set up data splitting for the inner loop of nested cross-validation in order to optimize hyperparameters.
            # Can use binning to use stratified folds for regression problems in the future.
            inner_cv_folds = KFold(n_splits=num_inner_cv_splits, shuffle=True, random_state=random_state_inner_cv_folds)

            # Iterate through models.
            ensemble_models = {}
            for model_name, mp in model_params.items():

                # Display model being evaluated.
                write_func(f"Evaluating model: {model_name}")

                # Define the steps in the pipeline so far.
                model_steps = [encoder, scaler] + mp["model"]

                # Optionally add PCA to the model pipeline in the appropriate position, just before the actual model.
                if apply_pca:
                    model_steps.insert(len(model_steps) - 1, PCA(n_components=num_pca_components))

                # Create the pipeline from the model steps.
                pipeline = make_pipeline(*model_steps)
                pipeline_model_name = pipeline.steps[-1][0]

                # Define the parameter grid for hyperparameter tuning.
                param_grid = {f"{pipeline_model_name}__{key}": value for key, value in mp["params"].items()}

                # Perform outer cross-validation to evaluate the model fairly for generalization on held-out test sets (_test for model evaluation and _generalization_holdout for error assessment).
                r2_scores_test = []
                biases_test = []
                stds_test = []
                errors_test = []
                decomposition_squared_errors_test = []
                r2_scores_generalization_holdout = []
                biases_generalization_holdout = []
                stds_generalization_holdout = []
                errors_generalization_holdout = []
                decomposition_squared_errors_generalization_holdout = []
                fold_models = []
                r2_scores_train = []
                y_test_pooled = []
                y_pred_pooled = []
                for train_idx, test_idx in outer_splits:

                    # Split data into outer train/test.
                    X_train = X_nested_cv.iloc[train_idx]
                    y_train = y_nested_cv.iloc[train_idx]
                    if num_models > 1:
                        X_test = X_nested_cv.iloc[test_idx]
                        y_test = y_nested_cv.iloc[test_idx]

                    # Inner CV for hyperparameter tuning.
                    grid_search = GridSearchCV(pipeline, param_grid, cv=inner_cv_folds, n_jobs=num_jobs_for_gridsearchcv, scoring='r2', verbose=3)
                    grid_search.fit(X_train, y_train)

                    # Use best model to predict on the outer test set in order to assess model generalization.
                    best_model = grid_search.best_estimator_
                    best_model.fit(X_train, y_train)

                    # Evaluate the best model on the training set just to make sure it generally performs best on the training set instead of the test or generalization sets. This is mainly for debugging and is unnecessary.
                    if metric == 'r2_score':
                        r2_scores_train.append(r2_score(y_train, best_model.predict(X_train)))
                    elif metric == 'mean_absolute_error':
                        r2_scores_train.append(mean_absolute_error(y_train, best_model.predict(X_train)))
                    elif metric == 'root_mean_squared_error':
                        r2_scores_train.append(root_mean_squared_error(y_train, best_model.predict(X_train)))
                    
                    # Calculate the metrics for the current fold model based using the current test set. Remember, if num_models == 1, then the test set is empty and we should just use the generalization holdout set for evaluation purposes.
                    if num_models > 1:
                        curr_fold_metrics = compute_metrics(best_model, X_test, y_test, metric=metric)
                        r2_scores_test.append(curr_fold_metrics['r^2 score'])  # This is what went into outer_scores before.
                        biases_test.append(curr_fold_metrics['bias'])
                        stds_test.append(curr_fold_metrics['std'])
                        errors_test.append(curr_fold_metrics['error'])
                        decomposition_squared_errors_test.append(curr_fold_metrics['decomposition sq. error'])
                        y_test_pooled.append(y_test)
                        y_pred_pooled.append(best_model.predict(X_test))

                    # Calculate the metrics for the current fold model using the generalization holdout set.
                    curr_fold_metrics = compute_metrics(best_model, X_generalization_holdout, y_generalization_holdout, metric=metric)
                    r2_scores_generalization_holdout.append(curr_fold_metrics['r^2 score'])
                    biases_generalization_holdout.append(curr_fold_metrics['bias'])
                    stds_generalization_holdout.append(curr_fold_metrics['std'])
                    errors_generalization_holdout.append(curr_fold_metrics['error'])
                    decomposition_squared_errors_generalization_holdout.append(curr_fold_metrics['decomposition sq. error'])

                    # Store the current fold model.
                    fold_models.append(best_model)

                # If num_models > 1 (so test set evaluation was performed), compute the metric for the pooled test set.
                if num_models > 1:
                    y_test_pooled = np.concatenate(y_test_pooled)
                    y_pred_pooled = np.concatenate(y_pred_pooled)
                    if metric == 'r2_score':
                        metric_test_pooled = r2_score(y_test_pooled, y_pred_pooled)
                    elif metric == 'mean_absolute_error':
                        metric_test_pooled = mean_absolute_error(y_test_pooled, y_pred_pooled)
                    elif metric == 'root_mean_squared_error':
                        metric_test_pooled = root_mean_squared_error(y_test_pooled, y_pred_pooled)

                # Create the ensemble model using the best models from each fold.
                ensemble_model = AveragingRegressor(estimators=[(f"fold_{i}", model) for i, model in enumerate(fold_models)])

                # Evaluate the ensemble on the generalization holdout set.
                ensemble_metrics = compute_metrics(ensemble_model, X_generalization_holdout, y_generalization_holdout, metric=metric, return_residuals=True)

                # Store test-set-related results for this model type.
                if num_models > 1:
                    test_dict = {
                        'R^2 scores (test sets)': [round(score, ndigits=3) for score in r2_scores_test],
                        'Biases (test sets)': [round(bias, ndigits=3) for bias in biases_test],
                        'Stds (test sets)': [round(std, ndigits=3) for std in stds_test],
                        'Errors (test sets)': [round(error, ndigits=3) for error in errors_test],
                        'Decomp. errors (test sets)': [round(signed_sqrt(decomposition_squared_error), ndigits=3) for decomposition_squared_error in decomposition_squared_errors_test],
                        'Mean R^2 score (test sets)': np.mean(r2_scores_test),  # This used to be called the Mean Generalization Accuracy.
                        'Mean bias (test sets)': np.mean(biases_test),
                        'Mean std (test sets)': np.mean(stds_test),
                        'Mean error (test sets)': np.mean(errors_test),
                        'Mean decomp. error (test sets)': signed_sqrt(np.mean(decomposition_squared_errors_test)),
                        'Test - train R^2 scores': [round(test_score - train_score, ndigits=2) for train_score, test_score in zip(r2_scores_train, r2_scores_test)],
                        'Pooled metric (test sets)': metric_test_pooled,
                    }
                else:  # num_models == 1
                    test_dict = {
                        'R^2 scores (test sets)': [np.nan],
                        'Biases (test sets)': [np.nan],
                        'Stds (test sets)': [np.nan],
                        'Errors (test sets)': [np.nan],
                        'Decomp. errors (test sets)': [np.nan],
                        'Mean R^2 score (test sets)': np.nan,
                        'Mean bias (test sets)': np.nan,
                        'Mean std (test sets)': np.nan,
                        'Mean error (test sets)': np.nan,
                        'Mean decomp. error (test sets)': np.nan,
                        'Test - train R^2 scores': [np.nan],
                        'Pooled metric (test sets)': np.nan,
                    }

                # Store all other results for this model type.
                rest_dict = {
                    'Ensemble model': ensemble_model,
                    'R^2 scores (gen. set)': [round(score, ndigits=3) for score in r2_scores_generalization_holdout],
                    'Mean R^2 score (gen. set)': np.mean(r2_scores_generalization_holdout),
                    'Mean bias (gen. set)': np.mean(biases_generalization_holdout),
                    'Mean std (gen. set)': np.mean(stds_generalization_holdout),
                    'Mean error (gen. set)': np.mean(errors_generalization_holdout),
                    'Mean decomp. error (gen. set)': signed_sqrt(np.mean(decomposition_squared_errors_generalization_holdout)),
                    'Gen. - train R^2 scores': [round(gen_score - train_score, ndigits=2) for train_score, gen_score in zip(r2_scores_train, r2_scores_generalization_holdout)],
                    'Ensemble R^2 score': ensemble_metrics['r^2 score'],
                    'Ensemble bias': ensemble_metrics['bias'],
                    'Ensemble std': ensemble_metrics['std'],
                    'Ensemble error': ensemble_metrics['error'],
                    'Ensemble decomp. error': signed_sqrt(ensemble_metrics['decomposition sq. error']),
                    'Ensemble residuals': ensemble_metrics['residuals'],
                    'Component non-interaction bias error': signed_sqrt(ensemble_metrics['component non-interaction bias sq. error']),
                    'Component interaction bias error': signed_sqrt(ensemble_metrics['component interaction bias sq. error']),
                    'Component non-interaction variance error': signed_sqrt(ensemble_metrics['component non-interaction variance sq. error']),
                    'Component interaction variance error': signed_sqrt(ensemble_metrics['component interaction variance sq. error']),
                    'Component decomp. error': signed_sqrt(ensemble_metrics['component decomposition sq. error']),
                    'R^2 scores (train sets)': [round(score, ndigits=3) for score in r2_scores_train],
                    'Mean R^2 score (train sets)': np.mean(r2_scores_train),
                }

                # Combine all results for this model type into a single dictionary.
                ensemble_models[model_name] = {**test_dict, **rest_dict}

            # Store the results for the current number of models.
            num_models_dict[num_models] = ensemble_models

        # Store the results for the current repetition.
        repetitions_list.append(num_models_dict)
        train_test_generalization_indices.append((train_idx_generalization, test_idx_generalization))
        
    # Create a dictionary mapping original column names to the determined categories, fitted, just for category extraction for the UI/UX.
    encoder_for_ui = ColumnTransformer(
        transformers=[('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), columns_to_encode)],
        remainder='passthrough'
    )
    encoder_for_ui.fit(X)  # Pre-fit to ensure consistent feature space
    onehot_encoder = encoder_for_ui.named_transformers_['onehot']
    encoder_categories = (onehot_encoder.categories_ if hasattr(onehot_encoder, 'categories_') else [])
    encoded_column_categories = {
        col: categories.tolist()
        for col, categories in zip(columns_to_encode, encoder_categories)
    }

    # Return the results.
    return {"numeric_columns": numeric_columns, "categorical_columns": categorical_columns, "feature_columns": feature_columns, "encoded_column_categories": encoded_column_categories, "repetitions_list": repetitions_list, "train_test_generalization_indices": train_test_generalization_indices, "encoder": encoder}


# Simple global prediction interval.
def compute_global_simple_intervals(model, X_test, error, bias, std, mse_components=False):

    # Predict using the ensemble.
    y_test = model.predict(X_test)

    # As a first-order approximation, estimate the standard deviation of the prediction as the sqrt(MSE) on the generalization holdout set.
    if not mse_components:
        bias = 0
        std = error  # sqrt(MSE)

    # Calculate the 95% confidence interval.
    prediction = y_test - bias
    lower_bound = prediction - 1.96 * std
    upper_bound = prediction + 1.96 * std

    # Get a descriptive name of the method.
    if not mse_components:
        method_name = 'rmse'
    else:
        method_name = 'bias + std'

    # Create a DataFrame to hold the predictions and intervals.
    predictions_df = pd.DataFrame({
        f'Lower bound (95%) ({method_name})': lower_bound,
        f'Prediction ({method_name})': prediction,
        f'Upper bound (95%) ({method_name})': upper_bound
    }, index=X_test.index)

    # Return the DataFrame with predictions and rough 95% prediction intervals.
    return predictions_df


# Compute locally adaptive conformal prediction intervals for regression using k-NN residuals. Note this is a state-of-the-art prediction interval method for regression tasks.
def compute_local_conformal_intervals(X_calib, y_calib, model, X_test, encoder, alpha=0.05, k=30, num_pca_components=0.95):

    # Normalize the calibration and test features using the same pipeline. This is important to ensure that the k-NN search is meaningful.
    normalization_pipeline = make_pipeline(encoder, StandardScaler(), PCA(n_components=num_pca_components))
    normalization_pipeline.fit(X_calib)
    X_calib_norm = normalization_pipeline.transform(X_calib)
    X_test_norm = normalization_pipeline.transform(X_test)

    # Fit k-NN on calibration features.
    nn = NearestNeighbors(n_neighbors=min(k, len(X_calib_norm)))
    nn.fit(X_calib_norm)

    # Predict on the data and calculate the residuals.
    y_test = model.predict(X_test)
    y_calib_pred = model.predict(X_calib)
    residuals = np.abs(y_calib - y_calib_pred)

    # For each test point, find the k nearest calibration points and compute the local quantile of the corresponding residuals.
    lower_bound = np.zeros_like(y_test)
    upper_bound = np.zeros_like(y_test)
    for i, x in enumerate(X_test_norm):
        _, idx = nn.kneighbors([x])
        local_residuals = residuals.iloc[idx[0]]
        q = np.quantile(local_residuals, 1 - alpha)
        lower_bound[i] = y_test[i] - q
        upper_bound[i] = y_test[i] + q

    # Create a DataFrame to hold the predictions and intervals.
    method_name = "conformal"
    predictions_df = pd.DataFrame({
        f'Lower bound (95%) ({method_name})': lower_bound,
        f'Prediction ({method_name})': y_test,
        f'Upper bound (95%) ({method_name})': upper_bound
    }, index=X_test.index)

    # Return the DataFrame with predictions and rough 95% prediction intervals.
    return predictions_df


# Perform the modeling.
def perform_modeling(df_to_analyze, models_to_run, independent_vars, dependent_var, numeric_columns, multiselect_columns, apply_pca, num_pca_components, generalization_holdout_frac, num_models, num_inner_cv_splits, num_jobs_for_gridsearchcv, write_func, error_func, random_state_generalization_split=42, num_repetitions=1, metric='r2_score'):

    # Determine the dictionary of models and parameters to run.
    full_model_params = get_common_models_and_hyperparameter_grids()
    model_params = {model_name: full_model_params[model_name] for model_name in models_to_run}

    # Perform nested cross-validation to evaluate the models.
    model_fitting_results = fit(df_to_analyze[independent_vars], df_to_analyze[dependent_var], numeric_columns, multiselect_columns, model_params=model_params, apply_pca=apply_pca, num_pca_components=num_pca_components, generalization_holdout_size=generalization_holdout_frac, num_models=num_models, num_inner_cv_splits=num_inner_cv_splits, random_state_generalization_split=random_state_generalization_split, random_state_outer_model_folds=42, random_state_inner_cv_folds=42, num_jobs_for_gridsearchcv=num_jobs_for_gridsearchcv, write_func=write_func, error_func=error_func, num_repetitions=num_repetitions, metric=metric)

    # Store the results in the session state.
    if model_fitting_results is not None:

        # Store the modeling repetition data.
        repetitions_list = model_fitting_results['repetitions_list']
        
        # Extract properties of the modeling repetition data.
        num_repetitions = len(repetitions_list)
        nums_models = sorted(list(repetitions_list[0].keys()))
        model_types = sorted(list(repetitions_list[0][nums_models[0]].keys()))

        # Obtain a list of single-number metrics that are returned from fit().
        single_number_metrics = ['Mean R^2 score (test sets)', 'Mean bias (test sets)', 'Mean std (test sets)', 'Mean error (test sets)', 'Mean decomp. error (test sets)', 'Mean R^2 score (gen. set)', 'Mean bias (gen. set)', 'Mean std (gen. set)', 'Mean error (gen. set)', 'Mean decomp. error (gen. set)', 'Ensemble R^2 score', 'Ensemble bias', 'Ensemble std', 'Ensemble error', 'Ensemble decomp. error', 'Component non-interaction bias error', 'Component interaction bias error', 'Component non-interaction variance error', 'Component interaction variance error', 'Component decomp. error', 'Mean R^2 score (train sets)', 'Pooled metric (test sets)']

        # Populate a 4D array holding all the metrics.
        all_single_number_metrics = np.zeros((num_repetitions, len(nums_models), len(model_types), len(single_number_metrics)))
        for irepetition in range(num_repetitions):
            for inum_models, num_models in enumerate(nums_models):
                for imodel_type, model_type in enumerate(model_types):
                    for isingle_number_metric, single_number_metric in enumerate(single_number_metrics):
                        all_single_number_metrics[irepetition, inum_models, imodel_type, isingle_number_metric] = repetitions_list[irepetition][num_models][model_type][single_number_metric]

        # Create a mapping of single-number metrics to their indices.
        metrics_indices = dict(zip(single_number_metrics, range(len(single_number_metrics))))

        # Data for the plots of the R^2 scores.
        r2_score_metrics = ['Mean R^2 score (train sets)', 'Mean R^2 score (test sets)', 'Mean R^2 score (gen. set)', 'Ensemble R^2 score', 'Pooled metric (test sets)']
        r2_score_indices = [metrics_indices[metric] for metric in r2_score_metrics]
        r2_score_data = {}
        r2_scores = all_single_number_metrics[:, :, :, r2_score_indices]
        r2_score_data['Mean'] = r2_scores.mean(axis=0)
        r2_score_data['Std'] = r2_scores.std(axis=0)
        r2_score_data['Min'] = r2_scores.min(axis=0)
        r2_score_data['Max'] = r2_scores.max(axis=0)
        r2_score_data['columns'] = [nums_models, model_types, r2_score_metrics]
        r2_score_data['All repetitions'] = r2_scores

        # Data for the plots of the ensemble mean squared error decompositions.
        ensemble_mse_decomposition_metrics = ['Ensemble error', 'Component non-interaction bias error', 'Component interaction bias error', 'Component non-interaction variance error', 'Component interaction variance error']
        ensemble_mse_decomposition_indices = [metrics_indices[metric] for metric in ensemble_mse_decomposition_metrics]
        ensemble_mse_decomposition_data = {}
        mse_metrics = signed_square(all_single_number_metrics[:, :, :, ensemble_mse_decomposition_indices])
        ensemble_mse_decomposition_data['Mean'] = mse_metrics.mean(axis=0)
        ensemble_mse_decomposition_data['Std'] = mse_metrics.std(axis=0)
        ensemble_mse_decomposition_data['Min'] = mse_metrics.min(axis=0)
        ensemble_mse_decomposition_data['Max'] = mse_metrics.max(axis=0)
        ensemble_mse_decomposition_data['columns'] = [nums_models, model_types, ensemble_mse_decomposition_metrics]
        ensemble_mse_decomposition_data['All repetitions'] = mse_metrics

        # Data for the plots of the ensemble R^2 scores.
        ensemble_r2_scores = {}
        ensemble_r2_score = all_single_number_metrics[:, :, :, metrics_indices['Ensemble R^2 score']]
        ensemble_r2_scores['Mean'] = ensemble_r2_score.mean(axis=0)
        ensemble_r2_scores['Std'] = ensemble_r2_score.std(axis=0)
        ensemble_r2_scores['Min'] = ensemble_r2_score.min(axis=0)
        ensemble_r2_scores['Max'] = ensemble_r2_score.max(axis=0)
        ensemble_r2_scores['columns'] = [nums_models, model_types]
        ensemble_r2_scores['All repetitions'] = ensemble_r2_score

        # Data for the plots of the ensemble errors.
        ensemble_errors = {}
        ensemble_error = all_single_number_metrics[:, :, :, metrics_indices['Ensemble error']]
        ensemble_errors['Mean'] = ensemble_error.mean(axis=0)
        ensemble_errors['Std'] = ensemble_error.std(axis=0)
        ensemble_errors['Min'] = ensemble_error.min(axis=0)
        ensemble_errors['Max'] = ensemble_error.max(axis=0)
        ensemble_errors['columns'] = [nums_models, model_types]
        ensemble_errors['All repetitions'] = ensemble_error

        # Create a dictionary to hold all the data for the plots.
        data_for_plots = {
            'R^2 score': r2_score_data,
            'Ensemble mean squared error decomposition': ensemble_mse_decomposition_data,
            'Ensemble R^2 score': ensemble_r2_scores,
            'Ensemble error': ensemble_errors  # This is exactly the RMSE.
        }

        # Return the data for plots.
        return model_fitting_results, data_for_plots


# Compute the permutation importance of the features in the ensemble model.
def get_permutation_importance(model, X_test, y_test, n_repeats=10, scoring='r2', random_state=42):

    # Compute permutation importance on the test set (for regression, we use R²).
    perm_result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,        # Use multiple shuffles for robust estimation.
        scoring=scoring,
        random_state=random_state,
    )

    # Create a DataFrame to display the results.
    perm_importance_df = pd.DataFrame({
        'feature': X_test.columns,  # original feature names
        'perm_importance_mean': perm_result.importances_mean,
        'perm_importance_std': perm_result.importances_std
    }).sort_values(by='perm_importance_mean', ascending=False)

    # Return the DataFrame with permutation importance results.
    return perm_importance_df


# Compute the Shapley importance of the features in the ensemble model using SHAP's KernelExplainer.
def get_shapley_importance(model, X_train, X_test, nsamples=100, background_sample_size=100, random_state=42):
    
    # Define a prediction function wrapping the custom ensemble.
    def ensemble_predict(data):
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=X_train.columns)
        return model.predict(data)

    # Select a background sample from the training data.
    background = X_train.sample(background_sample_size, random_state=random_state)

    # Initialize the KernelExplainer.
    explainer = shap.KernelExplainer(ensemble_predict, background)

    # Compute SHAP values for the test data.
    shap_values = explainer.shap_values(X_test, nsamples=nsamples)

    # Visualize the SHAP summary plot for the test set.
    shap.summary_plot(shap_values, X_test, feature_names=X_test.columns)
    fig_shap_summary_plot = plt.gcf()  # Get the current figure for the summary plot.

    # Compute aggregated SHAP values (mean absolute value per feature).
    aggregated_shap = np.mean(np.abs(shap_values), axis=0)

    # Create a DataFrame mapping feature names to their aggregated SHAP values.
    aggregated_df = pd.DataFrame({
        'feature': X_test.columns,   # these are the original feature names
        'mean_abs_shap': aggregated_shap
    }).sort_values(by='mean_abs_shap', ascending=False)

    # Return the DataFrame with aggregated SHAP values and the figure for the summary plot.
    return aggregated_df, fig_shap_summary_plot
