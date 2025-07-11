# Import necessary libraries.
import streamlit as st
import numpy as np
import pandas as pd
import copy
import plotting

# Set the prefix to the session state keys defined in this and other scripts.
st_key_prefix = 'select_model.py__'
st_key_prefix_perform_modeling = 'perform_modeling.py__'


# Get the best model type corresponding to the model types and number of component models and the repeitition whose ensemble R^2 score is closest to the highest mean ensemble R^2 score.
def set_best_model_repetition(data_for_repeat_modeling_plots):

    # Get the means of the ensemble R^2 scores and the columns corresponding to its indices. Also get the array of the ensemble R^2 scores for all repetitions.
    ensemble_r2_means = data_for_repeat_modeling_plots['Ensemble R^2 score']['Mean']
    ensemble_r2_all = data_for_repeat_modeling_plots['Ensemble R^2 score']['All repetitions']
    nums_models, model_types = data_for_repeat_modeling_plots['Ensemble R^2 score']['columns']

    # Find the maximum mean R^2 score and the corresponding model type and number of models.
    max_idx = np.unravel_index(np.argmax(ensemble_r2_means), ensemble_r2_means.shape)
    best_mean_r2 = ensemble_r2_means[max_idx]
    best_num_models = nums_models[max_idx[0]]
    best_model_type = model_types[max_idx[1]]

    # Get the scores for all repetitions for the best number of models and model type.
    best_repetitions = ensemble_r2_all[:, max_idx[0], max_idx[1]]

    # Get the repetition index corresponding to the repetition whose R^2 score is closest to the best mean R^2 score.
    best_repetition_idx = np.argmin(np.abs(best_repetitions - best_mean_r2))

    # Set the corresponding widget values in the session state.
    st.session_state[st_key_prefix + 'repetition_num'] = best_repetition_idx
    st.session_state[st_key_prefix + 'repetition_num_models'] = best_num_models
    st.session_state[st_key_prefix + 'repetition_model_type'] = best_model_type


# Render UI for selecting repeat ensemble model for analysis.
def select_repeat_ensemble_model():

    # Get the required data from the session state.
    model_fitting_results = st.session_state[f'{st_key_prefix_perform_modeling}model_fitting_results']
    data_for_repeat_modeling_plots = st.session_state[f'{st_key_prefix_perform_modeling}data_for_repeat_modeling_plots']

    # Get the options for the following widgets.
    num_repetitions = data_for_repeat_modeling_plots['Ensemble R^2 score']['All repetitions'].shape[0]
    repetition_nums_models, repetition_model_types = data_for_repeat_modeling_plots['Ensemble R^2 score']['columns']

    # Allow the user to select which model type to study.
    key = st_key_prefix + 'repetition_model_type'
    if key not in st.session_state:
        st.session_state[key] = repetition_model_types[0]
    repetition_model_type = st.selectbox('Select the model type to study:', repetition_model_types, key=key)

    # Allow the user to select which number of component models to study.
    key = st_key_prefix + 'repetition_num_models'
    if key not in st.session_state:
        st.session_state[key] = repetition_nums_models[0]
    repetition_num_models = st.selectbox('Select the number of component models to study:', repetition_nums_models, key=key)

    # Allow the user to select which repetition to study.
    key = st_key_prefix + 'repetition_num'
    if key not in st.session_state:
        st.session_state[key] = 0
    repetition_num = st.number_input('Select the repetition to study:', min_value=0, max_value=num_repetitions-1, key=key)

    # Output the ensemble R^2 score for the selected model type, number of component models, and repetition.
    ensemble_r2_score = data_for_repeat_modeling_plots['Ensemble R^2 score']['All repetitions'][repetition_num, repetition_nums_models.index(repetition_num_models), repetition_model_types.index(repetition_model_type)]
    st.write(f'R^2 score for selected ensemble model: {ensemble_r2_score:.3f}')

    # Have a button that auto-chooses the best model type, number of component models, and repetition based on the ensemble R^2 scores.
    st.button('Auto-choose best settings', key=st_key_prefix + 'repeat_modeling_auto_select_button', on_click=set_best_model_repetition, help='Select the best model type, number of component models, and repetition based on the ensemble R^2 scores.', args=(data_for_repeat_modeling_plots,))

    # Choose the selected ensemble model for analysis.
    st.button('Select model for analysis', key=st_key_prefix + 'repeat_modeling_select_model_for_analysis_button', on_click=lambda: st.session_state.update({
        st_key_prefix + 'chosen_ensemble_model_for_analysis': model_fitting_results['repetitions_list'][repetition_num][repetition_num_models][repetition_model_type],
        st_key_prefix + 'train_test_generalization_indices': model_fitting_results['train_test_generalization_indices'][repetition_num],
        st_key_prefix + 'actually_used_settings_for_select_model_for_analysis_button': {
            'repetition_num': repetition_num,
            'repetition_num_models': repetition_num_models,
            'repetition_model_type': repetition_model_type,
            'model_fitting_results': copy.deepcopy(st.session_state[st_key_prefix_perform_modeling + 'actually_used_settings_for_perform_modeling_button']),
        }}), help='Select the ensemble model for analysis.')


# # Define a function to highlight the row where the index matches the target value.
# def highlight_selected_row(row, repetition_model_type):
#     return ['background-color: yellow' if row.name == repetition_model_type else '' for _ in row]


# Main function.
def main():

    # Check if modeling has been performed.
    if not st_key_prefix_perform_modeling + 'actually_used_settings_for_perform_modeling_button' in st.session_state:
        st.warning('No modeling data found. Please perform modeling.')
        return
    
    # Define some columns.
    main_columns = st.columns([1/3, 2/3])
    
    # Allow the user to select a repeat ensemble model for analysis.
    with main_columns[0]:
        select_repeat_ensemble_model()

    # Check if the user has selected an (repeat) ensemble model for analysis.
    if st_key_prefix + 'actually_used_settings_for_select_model_for_analysis_button' not in st.session_state:
        st.info('Please select a model for analysis.')
        return
    
    # Get the metrics for all the model types but otherwise as selected via repetition and number of component models.
    chosen_model = st.session_state[st_key_prefix + 'chosen_ensemble_model_for_analysis']

    # Flatten the dictionary, excluding a list of results keys.
    keys_to_exclude = ['Ensemble model', 'Ensemble residuals']
    if st.session_state[st_key_prefix + 'actually_used_settings_for_select_model_for_analysis_button']['repetition_num_models'] == 1:
        keys_to_exclude += ['Test - train R^2 scores']
        keys_to_exclude += [key for key in chosen_model if key.endswith(' (test sets)')]
    ser_ensemble_models = pd.DataFrame({key: value for key, value in chosen_model.items() if key not in keys_to_exclude})

    # Display the summary for each model with the selected row highlighted.
    st.write(ser_ensemble_models)

    # Display a plotly histogram of the ensemble residuals while allowing the user to select the number of bins.
    with st.columns(2)[0]:
        key = st_key_prefix + 'histogram_ensemble_residuals_num_bins'
        if key not in st.session_state:
            st.session_state[key] = 20
        num_bins = st.slider('Number of bins for histogram of ensemble residuals:', min_value=5, max_value=100, step=5, key=key)
    df_residuals = pd.DataFrame({'Ensemble residuals': chosen_model['Ensemble residuals']})
    fig = plotting.plot_ensemble_residuals_histogram(df_residuals, num_bins=num_bins)
    st.plotly_chart(fig, use_container_width=True)


# Run the main function.
if __name__ == "__main__":
    main()
