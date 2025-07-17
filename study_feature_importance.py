# Import necessary libraries.
import streamlit as st
import modeling
import pandas as pd
import plotting
import copy

# Set the prefix to the session state keys defined in this and other scripts.
st_key_prefix = 'study_feature_importance.py__'
st_key_prefix_select_model = 'select_model.py__'
st_key_prefix_perform_modeling = 'perform_modeling.py__'
st_key_prefix_polars_data_loading = 'polars_data_loading.py__'


# Perform the feature importance analysis for the selected model type.
def analyze_feature_importance(chosen_ensemble_model, df_to_analyze, independent_vars, dependent_var, train_idx_generalization, test_idx_generalization, n_permutation_importance_repeats, random_state_permutation_importance, nsamples_shapley_importance, background_sample_size_shapley_importance, random_state_shapley_importance):

    # Extract the selected ensemble model.
    ensemble_model = chosen_ensemble_model['Ensemble model']

    # Create copies of the input dataframes (to avoid modifying the originals) and drop rows where any value in X or y is missing.
    X, y = modeling.initial_row_filter(df_to_analyze[independent_vars], df_to_analyze[dependent_var], write_func=print, error_func=st.error)

    # Obtain the same train and test dataframes used for the fitting.
    X_train = X.iloc[train_idx_generalization]
    X_test = X.iloc[test_idx_generalization]
    y_test = y.iloc[test_idx_generalization]

    # Run the feature importance analysis.
    perm_importance_df = modeling.get_permutation_importance(ensemble_model, X_test, y_test, n_repeats=n_permutation_importance_repeats, scoring='r2', random_state=random_state_permutation_importance)
    shapley_importance_df, fig_shap_summary_plot = modeling.get_shapley_importance(ensemble_model, X_train, X_test, nsamples=nsamples_shapley_importance, background_sample_size=background_sample_size_shapley_importance, random_state=random_state_shapley_importance)

    # Join on 'feature' and set as index.
    joined_df = pd.merge(perm_importance_df, shapley_importance_df, on='feature').sort_values(by='perm_importance_mean', ascending=False)

    # Create plotly plots for the feature importance.
    fig_perm_importance = plotting.plot_feature_importance(joined_df, x='feature', y='perm_importance_mean', title='Mean permutation importance vs. feature', labels={'perm_importance_mean': 'Mean(permutation importance)', 'feature': 'Feature'})
    fig_shap_importance = plotting.plot_feature_importance(joined_df, x='feature', y='mean_abs_shap', title='Mean absolute SHAP value vs. feature', labels={'mean_abs_shap': 'Mean(|SHAP value|)', 'feature': 'Feature'})

    # Return the results.
    return joined_df, fig_perm_importance, fig_shap_importance, fig_shap_summary_plot


# Main function.
def main():

    # Page information.
    st.write("This page allows you to study how important each of the independent variables are to the model. Two methods for evaluating this \"feature importance\" are available—permutation importance and Shapley importance—and they work very differently. Permutation importance evaluates the increase in model error after shuffling a feature’s values, whereas Shapley importance shows how much each feature pushes a model’s prediction up or down using only model outputs — not the true labels. (:one: step total.)")

    # Load the modeling data.
    if st_key_prefix_perform_modeling + 'actually_used_settings_for_perform_modeling_button' not in st.session_state:
        st.warning('No modeling data found. Please perform modeling.')
        return

    # Get the filtered data (which is the data to analyze by my design) and independent and dependent variables.
    df_to_analyze = st.session_state[st_key_prefix_perform_modeling + "df_filtered"]  # This isn't actually defined via the perform modeling button, but is rather defined at the beginning of that modeling page. In order to stick to hiding blocks like the prior one around "actually_used..." keys, we therefore choose the main key that is used in the perform modeling page. Again, this is not strict, but otherwise, it's unclear what error message to display if st_key_prefix_perform_modeling + 'df_filtered' is not in the session state, for example. It's reasonable enough to assume that if the user is here, they have performed modeling.
    independent_vars = st.session_state[st_key_prefix_perform_modeling + 'actually_used_settings_for_perform_modeling_button']['independent_vars']
    dependent_var = st.session_state[st_key_prefix_perform_modeling + 'actually_used_settings_for_perform_modeling_button']['dependent_var']

    # Ensure the model selection results are available.
    if st_key_prefix_select_model + 'actually_used_settings_for_select_model_for_analysis_button' not in st.session_state:
        st.warning('Please select a model for analysis.')
        return
    
    # Get the data for the selected model.
    train_idx_generalization, test_idx_generalization = st.session_state[st_key_prefix_select_model + 'train_test_generalization_indices']
    chosen_ensemble_model = st.session_state[st_key_prefix_select_model + 'chosen_ensemble_model_for_analysis']

    with st.columns([2, 1])[0]:

        # Create two settings columns, all hidden in an expander.
        with st.expander('Feature importance settings (click to expand):', expanded=False):
            settings_cols = st.columns(2, border=True)

            # In the first main column...
            with settings_cols[0]:

                # Section header.
                st.subheader('Permutation importance settings')

                # Allow the user to select the number of permutation importance repeats.
                key = st_key_prefix + 'n_permutation_importance_repeats'
                if key not in st.session_state:
                    st.session_state[key] = 10
                n_permutation_importance_repeats = st.number_input('Select the number of permutation importance repeats:', min_value=1, key=key)

                # Allow the user to specify whether they want to specify the seed for the permutation importance calculation.
                key = st_key_prefix + 'set_random_state_permutation_importance_seed'
                if key not in st.session_state:
                    st.session_state[key] = True
                set_random_state_permutation_importance_seed = st.checkbox("Set random state for the permutation importance?", key=key)

                # Allow the user to specify the random state for the permutation importance calculation.
                key = st_key_prefix + 'random_state_permutation_importance'
                if key not in st.session_state:
                    st.session_state[key] = 42
                random_state_permutation_importance = st.number_input("Random state for the permutation importance calculation:", min_value=0, step=1, key=key, disabled=not set_random_state_permutation_importance_seed)

                # Set the random state for the permutation importance.
                if not set_random_state_permutation_importance_seed:
                    random_state_permutation_importance = None

            # In the second main column...
            with settings_cols[1]:

                # Section header.
                st.subheader('Shapley importance settings')

                # Allow the user to specify the number of samples for the Shapley importance calculation.
                key = st_key_prefix + 'nsamples_shapley_importance'
                if key not in st.session_state:
                    st.session_state[key] = 100
                nsamples_shapley_importance = st.number_input("Number of samples for the Shapley importance calculation:", min_value=1, key=key)

                # Allow the user to specify the background sample size for the Shapley importance calculation.
                key = st_key_prefix + 'background_sample_size_shapley_importance'
                if key not in st.session_state:
                    st.session_state[key] = 100
                background_sample_size_shapley_importance = st.number_input("Background sample size for the Shapley importance calculation:", min_value=1, key=key)

                # Allow the user to specify whether they want to set the random state for the Shapley importance calculation.
                key = st_key_prefix + 'set_random_state_shapley_importance_seed'
                if key not in st.session_state:
                    st.session_state[key] = True
                set_random_state_shapley_importance_seed = st.checkbox("Set random state for the Shapley importance?", key=key)

                # Allow the user to specify the random state for the Shapley importance calculation.
                key = st_key_prefix + 'random_state_shapley_importance'
                if key not in st.session_state:
                    st.session_state[key] = 42
                random_state_shapley_importance = st.number_input("Random state for the Shapley importance calculation:", min_value=0, step=1, key=key, disabled=not set_random_state_shapley_importance_seed)

                # Set the random state for the Shapley importance.
                if not set_random_state_shapley_importance_seed:
                    random_state_shapley_importance = None

    # Allow the user to run feature importance analysis.
    if st.button(':one: Study feature importance'):
        st.session_state[st_key_prefix + 'actually_used_settings_for_study_feature_importance_button'] = {
            ('chosen_ensemble_model', 'train_idx_generalization', 'test_idx_generalization'): copy.deepcopy(st.session_state[st_key_prefix_select_model + 'actually_used_settings_for_select_model_for_analysis_button']),
            'df_to_analyze': copy.deepcopy(st.session_state[st_key_prefix_perform_modeling + 'actually_used_settings_for_retrieve_filtered_frame_button']),
            'independent_vars': copy.deepcopy(independent_vars),
            'dependent_var': dependent_var,
            'n_permutation_importance_repeats': n_permutation_importance_repeats,
            'random_state_permutation_importance': random_state_permutation_importance,
            'nsamples_shapley_importance': nsamples_shapley_importance,
            'background_sample_size_shapley_importance': background_sample_size_shapley_importance,
            'random_state_shapley_importance': random_state_shapley_importance,
        }
        joined_df, fig_perm_importance, fig_shap_importance, fig_shap_summary_plot = analyze_feature_importance(chosen_ensemble_model, df_to_analyze, independent_vars, dependent_var, train_idx_generalization, test_idx_generalization, n_permutation_importance_repeats, random_state_permutation_importance, nsamples_shapley_importance, background_sample_size_shapley_importance, random_state_shapley_importance)
        st.session_state[st_key_prefix + 'joined_df'] = joined_df.set_index('feature')
        st.session_state[st_key_prefix + 'fig_perm_importance'] = fig_perm_importance
        st.session_state[st_key_prefix + 'fig_shap_importance'] = fig_shap_importance
        st.session_state[st_key_prefix + 'fig_shap_summary_plot'] = fig_shap_summary_plot

    # Make two columns for displaying the results.
    results_cols = st.columns(2)

    # Display the results if they are available.
    if st_key_prefix + 'joined_df' in st.session_state:

        # In the first main column...
        with results_cols[0]:
        
            # Display the permutation importance plot.
            with st.columns(1, border=True)[0]:
                st.subheader('Permutation importance')
                st.write("Permutation importance measures how much a model’s performance drops when a feature is shuffled, with higher values indicating more importance, negative values suggesting the feature may be harmful (misleading the model more than it helps), and all values interpreted relative to each other.")
                st.plotly_chart(st.session_state[st_key_prefix + 'fig_perm_importance'])

            # Display the SHAP importance plot.
            with st.columns(1, border=True)[0]:
                st.subheader('Shapley importance')
                st.write("Shapley importance calculates how much each feature contributes to the model’s predictions on average, using only the model’s outputs, with higher absolute values indicating greater influence and importance judged by comparing features to one another.")
                st.plotly_chart(st.session_state[st_key_prefix + 'fig_shap_importance'])

        # In the second main column...
        with results_cols[1]:
        
            # Display the joined DataFrame.
            with st.expander('View feature importance results table:', expanded=False):
                st.dataframe(st.session_state[st_key_prefix + 'joined_df'])

            # Display the SHAP summary plot.
            with st.expander('View Shapley summary plot:', expanded=False):
                st.write("This plot shows how each feature contributes to the model’s predictions across the dataset. Each dot represents a single prediction, and its position on the x-axis shows the SHAP value (i.e., how much that feature pushed the prediction up or down). The color of the dot indicates the feature’s value for that instance (pink for high, blue for low, and gray for missing). Features are sorted top to bottom by their average absolute SHAP value, so the most influential features appear at the top. This plot helps you see not only which features matter most, but also how their values affect predictions — for example, whether high values of a feature tend to increase or decrease the model’s output.")
                st.pyplot(st.session_state[st_key_prefix + 'fig_shap_summary_plot'])


# Run the main function.
if __name__ == "__main__":
    main()
