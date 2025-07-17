# Import necessary libraries.
import streamlit as st
import pandas as pd
import modeling
import copy
import streamlit_dataframe_editor as sde
import os
import plotting

# Set the prefix to the session state keys defined in this and other scripts.
st_key_prefix = 'predict_on_new_data.py__'
st_key_prefix_perform_modeling = 'perform_modeling.py__'
st_key_prefix_select_model = 'select_model.py__'
st_key_prefix_study_feature_importance = 'study_feature_importance.py__'


# Run the prediction using the selected model.
def run_prediction(chosen_ensemble_model, X_test, prediction_method, encoder, df_to_analyze, independent_vars, dependent_var, test_idx_generalization, alpha=0.05, k=30, num_pca_components=0.95):

    # Obtain the core required components.
    model = chosen_ensemble_model['Ensemble model']
    error = chosen_ensemble_model['Ensemble error']
    bias = chosen_ensemble_model['Ensemble bias']
    std = chosen_ensemble_model['Ensemble std']

    # Obtain conformal prediction interval settings if needed.
    if prediction_method == 'Locally adaptive conformal' or prediction_method == 'All methods':

        # Create copies of the input dataframes (to avoid modifying the originals) and drop rows where any value in X or y is missing.
        X, y = modeling.initial_row_filter(df_to_analyze[independent_vars], df_to_analyze[dependent_var], write_func=print, error_func=st.error)

        # Obtain the same train and test dataframes used for the fitting.
        X_generalization = X.iloc[test_idx_generalization]
        y_generalization = y.iloc[test_idx_generalization]

    # Get the prediction dataframe(s).
    if prediction_method == 'RMSE':
        predictions_df = modeling.compute_global_simple_intervals(model, X_test, error, bias, std, mse_components=False)
    elif prediction_method == 'Bias + std':
        predictions_df = modeling.compute_global_simple_intervals(model, X_test, error, bias, std, mse_components=True)
    elif prediction_method == 'Locally adaptive conformal':
        predictions_df = modeling.compute_local_conformal_intervals(X_generalization, y_generalization, model, X_test, encoder, alpha=alpha, k=k, num_pca_components=num_pca_components)
    elif prediction_method == 'All methods':
        predictions_df_1 = modeling.compute_global_simple_intervals(model, X_test, error, bias, std, mse_components=False)
        predictions_df_2 = modeling.compute_global_simple_intervals(model, X_test, error, bias, std, mse_components=True)
        predictions_df_3 = modeling.compute_local_conformal_intervals(X_generalization, y_generalization, model, X_test, encoder, alpha=alpha, k=k, num_pca_components=num_pca_components)
        predictions_df = pd.concat([predictions_df_1, predictions_df_2, predictions_df_3], axis=1)
    else:
        raise ValueError(f"Unknown prediction method: {prediction_method}")
    
    # Join the predictions dataframe(s) with the input dataframe.
    X_test_with_predictions = pd.concat([X_test, predictions_df], axis=1)

    # Save the predictions (and the setting, i.e., the selected model type) to the session state.
    st.session_state[st_key_prefix + 'df_predictions'] = X_test_with_predictions

    # Output a success message.
    st.success("Prediction complete.")


# Main function.
def main():

    # Page information.
    st.write("This page allows you to input new features (independent variables)—starting either from scratch or from an uploaded file—and use the model to predict the target (dependent) variable. In addition, 95% intervals around these predictions are calculated. Both the predictions and their intervals will be (1) appended as columns to the output data table and (2) plotted, with the predictions on the y-axis and a selected independent variable (by default, the most \"Shapley important\") on the x-axis. Three prediction interval methods are included: two \"global\" methods (same error for all predictions) and one \"local\" method (adaptive to each prediction). (:two: steps total.)")

    # Ensure the model selection results are available.
    if st_key_prefix_select_model + 'actually_used_settings_for_select_model_for_analysis_button' not in st.session_state:
        st.warning('Please select a model for analysis.')
        return
    
    # Get data from the perform modeling page.
    df_to_analyze = st.session_state[st_key_prefix_perform_modeling + "df_filtered"]
    independent_vars = st.session_state[st_key_prefix_perform_modeling + 'actually_used_settings_for_perform_modeling_button']['independent_vars']
    dependent_var = st.session_state[st_key_prefix_perform_modeling + 'actually_used_settings_for_perform_modeling_button']['dependent_var']
    numeric_columns = st.session_state[st_key_prefix_perform_modeling + 'numeric_columns']
    encoded_column_categories = st.session_state[st_key_prefix_perform_modeling + 'encoded_column_categories']
    feature_columns = st.session_state[st_key_prefix_perform_modeling + 'feature_columns']
    encoder = st.session_state[st_key_prefix_perform_modeling + 'encoder']

    # Get data from the model selection page.
    _, test_idx_generalization = st.session_state[st_key_prefix_select_model + 'train_test_generalization_indices']
    chosen_ensemble_model = st.session_state[st_key_prefix_select_model + 'chosen_ensemble_model_for_analysis']

    # Specify the column configuration for the data editor.
    # Note that SelectboxColumn restricts the options to the categories of the categorical columns but the model will now (as of 5/13/25) robustly handle unseen categories. So if in the future e.g. there's a new accept_new_options parameter to SelectboxColumn like there now is in st.selectbox(), that would allow users to perform predictions using unseen categories and would be ideal.
    column_config = {column_name: (
        st.column_config.NumberColumn(label=column_name, required=True)
        if column_name in numeric_columns else
        st.column_config.SelectboxColumn(label=column_name, required=True, options=encoded_column_categories[column_name])
    ) for column_name in feature_columns}

    # Define the data editor object.
    df_editor_key = st_key_prefix + 'dataframe_editor_new_inputs'
    if df_editor_key not in st.session_state:
        st.session_state[df_editor_key] = sde.DataframeEditor(df_name=st_key_prefix + 'df_new_inputs', default_df_contents=pd.DataFrame(columns=feature_columns))

    # Frame some widgets.
    with st.columns([1, 2])[0]:
        # with st.columns(1, border=True)[0]:
        with st.expander("Upload new inputs (optional):"):

            # Allow the user to upload a CSV or Excel file with new data.
            uploaded_input_file = st.file_uploader("Upload a CSV or Excel file with new data on which to predict (optional):", type=['csv', 'xls', 'xlsx'])

            # Allow the user to load new data into the editable dataframe to be used for prediction.
            if st.button(':arrows_counterclockwise: Load uploaded input file', use_container_width=True, disabled=(uploaded_input_file is None), help=":warning: This will overwrite any existing data in the data editor below."):

                # Read the uploaded file into a DataFrame.
                uploaded_input_file_name = uploaded_input_file.name
                file_extension = os.path.splitext(uploaded_input_file_name)[1].lower()
                if file_extension == '.csv':
                    df_uploaded = pd.read_csv(uploaded_input_file)
                elif file_extension == '.xls' or file_extension == '.xlsx':
                    df_uploaded = pd.read_excel(uploaded_input_file)
                
                # Load this dataframe into the editable dataframe.
                common_columns = set(df_uploaded.columns).intersection(set(feature_columns))
                if not common_columns:
                    st.error('The uploaded CSV file does not contain any of the required columns. Please upload a valid CSV file.')
                    return
                df_new_inputs = pd.DataFrame(columns=feature_columns)
                for column in common_columns:
                    df_new_inputs[column] = df_uploaded[column]
                st.session_state[df_editor_key].update_editor_contents(new_df_contents=df_new_inputs)

                # Success.
                st.session_state[st_key_prefix + 'actually_used_settings_for_load_uploaded_input_file_button'] = {
                    'uploaded_input_file_name': uploaded_input_file_name,
                    'feature_columns': feature_columns,
                }
                st.success('Uploaded file has been loaded successfully. You can now edit the data below if needed and run prediction.')

    # Display the editable dataframe.
    with st.columns(1, border=True)[0]:
        st.header(":one: Enter or edit new inputs for prediction")
        st.write("Note: Use dropdown menus for categorical variables.")
        st.session_state[df_editor_key].dataframe_editor(column_config=column_config, reset_data_editor_button_text=":warning: Clear data editor")

    # Frame some widgets.
    with st.columns([1, 2])[0]:
        with st.columns(1, border=True)[0]:

            # Display a header.
            st.header(":two: Run prediction")

            # Allow the user to select the prediction method.
            key = st_key_prefix + 'prediction_method'
            prediction_method_options = ['RMSE', 'Bias + std', 'Locally adaptive conformal', 'All methods']
            if key not in st.session_state:
                st.session_state[key] = prediction_method_options[0]
            prediction_method = st.selectbox("Select the prediction method:", options=prediction_method_options, key=key)

            # Allow the user to select k for locally adaptive conformal prediction.
            if prediction_method == 'Locally adaptive conformal' or prediction_method == 'All methods':
                key = st_key_prefix + 'k_for_locally_adaptive_conformal_prediction'
                if key not in st.session_state:
                    st.session_state[key] = 30
                k_for_locally_adaptive_conformal_prediction = st.number_input("Select k (number of nearest neighbors) for locally adaptive conformal prediction:", min_value=1, max_value=1000, step=1, key=key)
            else:
                k_for_locally_adaptive_conformal_prediction = None

            # Allow the user to run the prediction.
            if st.button("Run prediction", use_container_width=True):
                X_test = st.session_state[df_editor_key].reconstruct_edited_dataframe()  # Get the current contents of the edited DataFrame.
                if X_test.empty:
                    st.warning('The input dataframe is empty. Please enter data to run predictions.')
                else:
                    st.session_state[st_key_prefix + 'actually_used_settings_for_run_prediction_button'] = {
                        ('chosen_ensemble_model', 'test_idx_generalization', 'encoder'): copy.deepcopy(st.session_state[st_key_prefix_select_model + 'actually_used_settings_for_select_model_for_analysis_button']),
                        'X_test': X_test.copy(),
                        'prediction_method': prediction_method,
                        'df_to_analyze': copy.deepcopy(st.session_state[st_key_prefix_perform_modeling + 'actually_used_settings_for_retrieve_filtered_frame_button']),
                        'independent_vars': copy.deepcopy(independent_vars),
                        'dependent_var': dependent_var,
                        'k_for_locally_adaptive_conformal_prediction': k_for_locally_adaptive_conformal_prediction,
                    }
                    run_prediction(chosen_ensemble_model, X_test, prediction_method, encoder, df_to_analyze, independent_vars, dependent_var, test_idx_generalization, k=k_for_locally_adaptive_conformal_prediction)

        # Ensure prediction results are available.
        if st_key_prefix + 'actually_used_settings_for_run_prediction_button' not in st.session_state:
            st.info('Please run prediction.')        
            return
        
    # Frame the predictions section.
    with st.columns(1, border=True)[0]:

        # Write a header.
        st.header("🔮 Predictions")

        # Write the predictions concatenated with the new data.
        with st.expander("View prediction results table:", expanded=False):
            st.dataframe(st.session_state[st_key_prefix + 'df_predictions'])

        # Get a handle for the predictions DataFrame.
        df_predictions = st.session_state[st_key_prefix + 'df_predictions']

        # User selects x-axis variable, defaulting to that with the highest Shapley importance.
        with st.columns([1, 3])[0]:
            key = st_key_prefix + 'x_axis_col_for_prediction_plot'
            if key not in st.session_state:
                if st_key_prefix_study_feature_importance + 'joined_df' in st.session_state:
                    joined_df = st.session_state[st_key_prefix_study_feature_importance + 'joined_df']
                    best_feature = joined_df.sort_values(by='mean_abs_shap', ascending=False).index[0]
                    if best_feature in feature_columns:
                        st.session_state[key] = best_feature
                    else:
                        st.session_state[key] = feature_columns[0]
                else:
                    st.session_state[key] = feature_columns[0]
            x_axis_col_for_prediction_plot = st.selectbox("Select independent variable to plot on the x-axis:", options=feature_columns, key=key)

        # Define prediction methods and column mappings
        prediction_methods = ['RMSE', 'Bias + std', 'Locally adaptive conformal']
        method_to_col_map = {
            'RMSE': 'Prediction (rmse)',
            'Bias + std': 'Prediction (bias + std)',
            'Locally adaptive conformal': 'Prediction (conformal)',
        }
        method_to_low_col_map = {
            'RMSE': 'Lower bound (95%) (rmse)',
            'Bias + std': 'Lower bound (95%) (bias + std)',
            'Locally adaptive conformal': 'Lower bound (95%) (conformal)',
        }
        method_to_high_col_map = {
            'RMSE': 'Upper bound (95%) (rmse)',
            'Bias + std': 'Upper bound (95%) (bias + std)',
            'Locally adaptive conformal': 'Upper bound (95%) (conformal)',
        }

        # Only include methods present in the dataframe.
        available_methods = [m for m in prediction_methods if method_to_col_map[m] in df_predictions.columns]
        method_to_col_map = {m: method_to_col_map[m] for m in available_methods}
        method_to_low_col_map = {m: method_to_low_col_map[m] for m in available_methods}
        method_to_high_col_map = {m: method_to_high_col_map[m] for m in available_methods}

        # Plot predictions with error bars.        
        fig = plotting.plot_predictions_with_error_bars(
            df_predictions,
            x_col=x_axis_col_for_prediction_plot,
            prediction_methods=available_methods,
            method_to_col_map=method_to_col_map,
            method_to_low_col_map=method_to_low_col_map,
            method_to_high_col_map=method_to_high_col_map,
            title=f"Predictions with error bars (method: {(st.session_state[st_key_prefix + 'actually_used_settings_for_run_prediction_button']['prediction_method']).lower()})",
            xaxis_title=x_axis_col_for_prediction_plot,
            yaxis_title="Prediction",
        )
        st.plotly_chart(fig, use_container_width=True)


# Run the main function.
if __name__ == "__main__":
    main()
