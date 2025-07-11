# Import necessary libraries.
import streamlit as st
import polars_data_loading as pdl
import modeling
import copy
import ast

# Set the prefix to the session state keys defined in this script.
st_key_prefix = 'perform_modeling.py__'
st_key_prefix_load_and_filter_data = 'load_and_filter_data.py__'


# Perform the modeling (Streamlit version of modeling.perform_modeling()).
def perform_modeling(df_to_analyze, models_to_run, independent_vars, dependent_var, numeric_columns, multiselect_columns, apply_pca, num_pca_components, generalization_holdout_frac, num_models, num_inner_cv_splits, num_jobs_for_gridsearchcv, write_func=st.write, error_func=st.error, random_state_generalization_split=42, num_repetitions=1, metric='r2_score'):

    # Perform the modeling.
    # Here is probably where we could wrap this in an async function.
    results = modeling.perform_modeling(df_to_analyze, models_to_run, independent_vars, dependent_var, numeric_columns, multiselect_columns, apply_pca, num_pca_components, generalization_holdout_frac, num_models, num_inner_cv_splits, num_jobs_for_gridsearchcv, write_func, error_func, random_state_generalization_split=random_state_generalization_split, num_repetitions=num_repetitions, metric=metric)

    # If the modeling was successful...
    if results is not None:

        # Unpack the results.
        model_fitting_results, data_for_plots = results

        # Store the results in the session state from the repeat modeling.
        st.session_state[st_key_prefix + 'model_fitting_results'] = model_fitting_results
        st.session_state[st_key_prefix + 'numeric_columns'] = model_fitting_results['numeric_columns']
        st.session_state[st_key_prefix + 'categorical_columns'] = model_fitting_results['categorical_columns']
        st.session_state[st_key_prefix + 'encoded_column_categories'] = model_fitting_results['encoded_column_categories']
        st.session_state[st_key_prefix + 'feature_columns'] = model_fitting_results['feature_columns']
        st.session_state[st_key_prefix + 'encoder'] = model_fitting_results['encoder']
        st.session_state[st_key_prefix + 'data_for_repeat_modeling_plots'] = data_for_plots

        # Output a success message.
        st.success("Modeling completed successfully.")

    # Otherwise, output an error message.
    else:
        st.error("Modeling failed.")


# Retrieve the filtered frame.
def retrieve_filtered_frame(filtered_frame, frame_type_to_use):

    # Obtain the dataframe (polars or pandas) or lazyframe for further processing.
    frame = pdl.generate_or_return_frame(filtered_frame)

    # Convert this dataframe to pandas depending on its type.
    if frame_type_to_use == 'Eager':
        st.session_state[st_key_prefix + 'df_filtered'] = frame.to_pandas()
    elif frame_type_to_use == 'Lazy':
        st.session_state[st_key_prefix + 'df_filtered'] = frame.collect().to_pandas()
    else:
        st.session_state[st_key_prefix + 'df_filtered'] = frame

    # Output a success message.
    st.success("Filtered frame retrieved.")


# Convert the "registered," filtered dataframe to a pandas dataframe.
def load_filtered_data():

    # Obtain a pandas dataframe from the registered dataset from the loading and filtering page.
    key_df_filtered = st_key_prefix + 'df_filtered'
    key = st_key_prefix + 'actually_used_settings_for_retrieve_filtered_frame_button'
    if st.button('Retrieve filtered frame') or key_df_filtered not in st.session_state:

        # Ensure a filtered dataframe has been registered on the previous page.
        if f'{st_key_prefix_load_and_filter_data}filtered_frame' not in st.session_state:
            st.warning('Please load and filter the data first.')
            return

        # Define the settings for retrieving the filtered frame.
        retrieve_filtered_frame_settings = {'frame_type_to_use': st.session_state[f'{st_key_prefix_load_and_filter_data}frame_type_to_use']}

        # Store the settings in the session state.
        st.session_state[key] = {**retrieve_filtered_frame_settings, **{
            'loaded_dataset_name': st.session_state[f'{st_key_prefix_load_and_filter_data}loaded_dataset_name'],
            'actually_used_settings_for_save_frame_for_analysis_button': copy.deepcopy(st.session_state[f'{st_key_prefix_load_and_filter_data}actually_used_settings_for_save_frame_for_analysis_button']),
        }}

        # Retrieve the filtered frame.
        retrieve_filtered_frame(st.session_state[f'{st_key_prefix_load_and_filter_data}filtered_frame'], **retrieve_filtered_frame_settings)

    current_settings = str(st.session_state[key]) if key in st.session_state else None
    with st.popover('Currently retrieved frame'):
        st.write(current_settings)

    # Return the data from memory.
    return {
        "df_filtered": st.session_state[key_df_filtered],
        "numeric_columns": st.session_state[f'{st_key_prefix_load_and_filter_data}lazyframe_info']['column_types']['numeric'],
        "multiselect_columns": st.session_state[f'{st_key_prefix_load_and_filter_data}lazyframe_info']['column_types']['multiselect'],
        }


# Automatically add all new columns that are not the dependent variable (if present) and that have no null values. Also, ensure that potential new multiselect columns have more than one unique value.
def kitchen_sink(df, numeric_columns, multiselect_columns, consider_numeric=True, consider_multiselect=True):

    # Get the keys for the relevant session state variables.
    independent_vars_key = st_key_prefix + 'independent_vars'
    dependent_var_key = st_key_prefix + 'dependent_var'

    # Get the current independent and dependent variables, if present.
    current_independent_vars = st.session_state[independent_vars_key]
    dep_var = st.session_state.get(dependent_var_key, None)

    # Set columns to consider adding.
    if consider_numeric and consider_multiselect:
        column_choices = numeric_columns + multiselect_columns
    elif consider_numeric and not consider_multiselect:
        column_choices = numeric_columns
    elif not consider_numeric and consider_multiselect:
        column_choices = multiselect_columns
    else:
        st.error("No columns to consider adding.")
        return

    # Get the additional columns with no nulls.
    valid_columns = [
        col for col in column_choices
        if (col != dep_var) and (col not in current_independent_vars) and (not df[col].isnull().any()) and (df[col].nunique() > 1 if col in multiselect_columns else True)
    ]

    # Combine the current independent variables with the valid columns.
    new_independent_vars = current_independent_vars + valid_columns

    # Update the session state.
    st.session_state[independent_vars_key] = new_independent_vars


# Plan to execute the modeling when the button is clicked.
def execute_perform_modeling_button(models_to_run, independent_vars, dependent_var, numeric_columns, multiselect_columns, apply_pca, num_pca_components, generalization_holdout_frac, num_models_list_string, num_inner_cv_splits, num_jobs_for_gridsearchcv, random_state_generalization_split, num_repetitions, metric, key):

    # Convert the string numbers of models to a list of integers.
    try:
        num_models_list = ast.literal_eval(num_models_list_string)
        if not isinstance(num_models_list, list) or len(num_models_list) == 0:
            st.error("Please enter a non-empty list of integers, e.g., [1, 2, 3, 4, 5, 10].")
            return
    except Exception:
        st.error("Invalid input. Please enter a list of integers, e.g., [1, 2, 3, 4, 5, 10].")
        return

    # Store the settings for the repeat modeling in the session state.
    modeling_settings = {'models_to_run': copy.deepcopy(models_to_run), 'independent_vars': copy.deepcopy(independent_vars), 'dependent_var': dependent_var, 'numeric_columns': copy.deepcopy(numeric_columns), 'multiselect_columns': copy.deepcopy(multiselect_columns), 'apply_pca': apply_pca, 'num_pca_components': num_pca_components, 'generalization_holdout_frac': generalization_holdout_frac, 'num_models': copy.deepcopy(num_models_list), 'num_inner_cv_splits': num_inner_cv_splits, 'num_jobs_for_gridsearchcv': num_jobs_for_gridsearchcv, 'random_state_generalization_split': random_state_generalization_split, 'num_repetitions': num_repetitions, 'metric': metric}
    st.session_state[key] = modeling_settings

    # Store a flag to perform the modeling in another column.
    st.session_state[st_key_prefix + 'perform_modeling'] = True


# Main function.
def main():

    # Load the filtered data for downstream analysis.
    filtered_data = load_filtered_data()
    if filtered_data is None:
        return
    
    # Assign the filtered data to local variables.
    df_to_analyze = filtered_data["df_filtered"]
    numeric_columns = filtered_data["numeric_columns"]
    multiselect_columns = filtered_data["multiselect_columns"]

    # Filtered data section.
    with st.expander('Filtered data:', expanded=True):

        # Display the dataframe to analyze.
        st.write(df_to_analyze)

    # Join together both numeric and categorical column types into a single list as options for the variables to subsequently select.
    column_choices = numeric_columns + multiselect_columns

    # Create columns in the interface.
    main_columns = st.columns([0.33, 0.67])

    # In the first column...
    with main_columns[0]:

        # Modeling settings section.
        with st.columns(1, border=True)[0]:

            # Write a section header.
            st.header('Modeling Settings')

            # Allow the user to select the independent variable(s).
            key = st_key_prefix + 'independent_vars'
            if key not in st.session_state:
                st.session_state[key] = []
            independent_vars = st.multiselect('Select independent variable(s):', options=column_choices, key=key)

            # Allow the user to automatically add all columns that are not the dependent variable (if present) and that have no null values.
            with st.expander('Kitchen sink options:', expanded=False):
                st.button("Numeric kitchen sink", help="Add all numeric columns that are not the dependent variable (if selected) and that have no null values.", on_click=kitchen_sink, kwargs=dict(df=df_to_analyze, numeric_columns=numeric_columns, multiselect_columns=multiselect_columns, consider_numeric=True, consider_multiselect=False), use_container_width=True)
                st.button("Multiselect kitchen sink", help="Add all multiselect columns that are not the dependent variable (if selected), that have no null values, and that have more than one unique value.", on_click=kitchen_sink, kwargs=dict(df=df_to_analyze, numeric_columns=numeric_columns, multiselect_columns=multiselect_columns, consider_numeric=False, consider_multiselect=True), use_container_width=True)
                st.button("Entire kitchen sink", help="Add all numeric and multiselect columns that are not the dependent variable (if selected), that have no null values, and that have more than one unique value (if a multiselect column type).", on_click=kitchen_sink, kwargs=dict(df=df_to_analyze, numeric_columns=numeric_columns, multiselect_columns=multiselect_columns, consider_numeric=True, consider_multiselect=True), use_container_width=True)

            # Allow the user to select the dependent variable.
            key = st_key_prefix + 'dependent_var'
            if key not in st.session_state:
                st.session_state[key] = None
            dependent_var = st.selectbox('Select dependent variable:', options=column_choices, key=key)

            # Determine whether both independent and dependent variables have been set.
            variables_are_specified = independent_vars and dependent_var

            # Determine the available models to run.
            key = st_key_prefix + 'available_models'
            if key not in st.session_state:
                st.session_state[key] = [model_name for model_name in modeling.get_common_models_and_hyperparameter_grids().keys()]
            available_models = st.session_state[key]

            # Allow the user to specify the models to run.
            key = st_key_prefix + 'models_to_run'
            if key not in st.session_state:
                st.session_state[key] = []
            models_to_run = st.multiselect('Select models to run:', options=available_models, key=key, disabled=not variables_are_specified)

            # Determine if we can enable the remaining widgets.
            ready_to_fit = variables_are_specified and models_to_run

            # Allow the user to specify whether to apply PCA to reduce multicollinearity.
            key = st_key_prefix + 'apply_pca'
            if key not in st.session_state:
                st.session_state[key] = False
            apply_pca = st.checkbox("Apply PCA to reduce multicollinearity?", key=key, disabled=not ready_to_fit)

            # Allow the user to specify the number of PCA components.
            key = st_key_prefix + 'num_pca_components'
            if key not in st.session_state:
                st.session_state[key] = 0.95
            num_pca_components = st.number_input("Number of PCA components (integer or 0-1 fraction):", key=key, disabled=(not ready_to_fit) or not apply_pca)

            # Allow the user to specify the fraction of the dataset to use for the generalization holdout set.
            key = st_key_prefix + 'generalization_holdout_frac'
            if key not in st.session_state:
                st.session_state[key] = 0.2
            generalization_holdout_frac = st.number_input("Fraction of dataset to use for the generalization holdout set:", min_value=0.0, max_value=1.0, step=0.01, key=key, disabled=not ready_to_fit)

            # Allow the user to specify whether they want to specify the seed for the generalization split.
            key = st_key_prefix + 'set_random_state_generalization_seed'
            if key not in st.session_state:
                st.session_state[key] = False
            set_random_state_generalization_seed = st.checkbox("Set random state for the generalization split?", key=key, disabled=not ready_to_fit)

            # Allow the user to specify the random state for the generalization split.
            key = st_key_prefix + 'random_state_generalization_split'
            if key not in st.session_state:
                st.session_state[key] = 42
            random_state_generalization_split = st.number_input("Random state for the generalization split:", min_value=0, step=1, key=key, disabled=(not ready_to_fit) or (not set_random_state_generalization_seed))

            # Set the random state for the generalization split.
            if not set_random_state_generalization_seed:
                random_state_generalization_split = None

            # Allow the user to specify the number of cross-validation folds.
            key = st_key_prefix + 'num_inner_cv_splits'
            if key not in st.session_state:
                st.session_state[key] = 5
            num_inner_cv_splits = st.number_input("Number of cross-validation folds:", min_value=2, step=1, key=key, disabled=not ready_to_fit)

            # Allow the user to specify the number of jobs to use for the grid search.
            key = st_key_prefix + 'num_jobs_for_gridsearchcv'
            if key not in st.session_state:
                st.session_state[key] = -1
            num_jobs_for_gridsearchcv = st.number_input("Number of jobs for grid search (negative for all available):", min_value=-1, step=1, key=key, disabled=not ready_to_fit)

            # Allow the user to specify the metric to use not for the grid search (which is currently fixed as r2) but rather for the model evaluations separate from the grid search.
            key = st_key_prefix + 'metric'
            if key not in st.session_state:
                st.session_state[key] = 'r2_score'
            metric = st.selectbox("Metric to use for model evaluation post-HPO:", options=['r2_score', 'mean_absolute_error', 'root_mean_squared_error'], key=key, disabled=not ready_to_fit)

            # Allow the user to specify multiple numbers of models to ensemble.
            key = st_key_prefix + 'num_models_list_string'
            if key not in st.session_state:
                st.session_state[key] = '[1, 2, 3, 4, 5, 10]'
            num_models_list_string = st.text_input("Numbers of models to ensemble (list of integers):", key=key, disabled=not ready_to_fit)

            # Allow the user to specify the number of repetitions for the modeling.
            key = st_key_prefix + 'num_repetitions'
            if key not in st.session_state:
                st.session_state[key] = 1
            num_repetitions = st.number_input("Number of modeling repetitions:", min_value=1, step=1, key=key, disabled=not ready_to_fit)

            # Evaluate the ensembles for all numbers of models and repetitions.
            key = st_key_prefix + 'actually_used_settings_for_perform_modeling_button'
            st.button("Perform modeling", disabled=not ready_to_fit, on_click=execute_perform_modeling_button, args=(models_to_run, independent_vars, dependent_var, numeric_columns, multiselect_columns, apply_pca, num_pca_components, generalization_holdout_frac, num_models_list_string, num_inner_cv_splits, num_jobs_for_gridsearchcv, random_state_generalization_split, num_repetitions, metric, key))

            # # Display the current settings for the repeat modeling.
            # current_settings = str(st.session_state[key]) if key in st.session_state else None
            # with st.popover('Current model settings'):
            #     st.write(current_settings)

    # In the second column...
    with main_columns[1]:
    
        # Perform the repeat modeling.
        st.markdown("<h2 style='color: green;'>Live modeling output...</h2>", unsafe_allow_html=True)
        if (st_key_prefix + 'perform_modeling' in st.session_state) and st.session_state[st_key_prefix + 'perform_modeling']:
            incomplete_settings = st.session_state[st_key_prefix + 'actually_used_settings_for_perform_modeling_button']
            st.session_state[st_key_prefix + 'actually_used_settings_for_perform_modeling_button'] = {**incomplete_settings, **{
                'actually_used_settings_for_retrieve_filtered_frame_button': copy.deepcopy(st.session_state[f'{st_key_prefix}actually_used_settings_for_retrieve_filtered_frame_button']),
            }}
            with st.container(height=500):
                perform_modeling(df_to_analyze, **incomplete_settings)
            st.session_state[st_key_prefix + 'perform_modeling'] = False  # Reset the flag after performing the modeling.
        else:
            st.write('No modeling is being performed.')


# Run the main function.
if __name__ == "__main__":
    main()
