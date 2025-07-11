# Import necessary libraries.
import streamlit as st
import plotting

# Set the prefix to the session state keys defined in this and other scripts.
st_key_prefix = 'visualize_model_metrics.py__'
st_key_prefix_perform_modeling = 'perform_modeling.py__'


# Plot repeat modeling results.
def repeat_modeling_results_plot(data_for_repeat_modeling_plots, repeat_modeling_results='R^2 score', title_suffix='', dims=[0, 1, 2], last_axis_index=9999):
    
    # Allow user to select the plot type.
    repeat_modeling_results_lower = repeat_modeling_results.lower()
    key = st_key_prefix + repeat_modeling_results + '_plot_type'
    if key not in st.session_state:
        st.session_state[key] = 'line'
    repeat_modeling_results_plot_type = st.radio(
        label=f"Select plot type for {repeat_modeling_results_lower}:",
        options=['line', 'box', 'violin'],
        horizontal=True,
        key=key,
        help=f"Select the plot type for the {repeat_modeling_results_lower}. The line plot shows the mean with shaded min/max area. The box and violin plots show the distribution of the {repeat_modeling_results_lower} for each number of models.",
    )

    # Render the plotly plot.
    st.plotly_chart(plotting.plot_ensemble_metrics(data_for_repeat_modeling_plots[repeat_modeling_results], xaxis_title="Number of models", yaxis_title=repeat_modeling_results, title=f"{repeat_modeling_results} vs. number of models{title_suffix}", plot_type=repeat_modeling_results_plot_type, dims=dims, last_axis_index=last_axis_index))


# Main function.
def main():

    # Load the modeling data.
    if st_key_prefix_perform_modeling + 'actually_used_settings_for_perform_modeling_button' not in st.session_state:
        st.warning('No modeling data found. Please perform modeling.')
        return

    # Get the required data from the session state.
    data_for_repeat_modeling_plots = st.session_state[f'{st_key_prefix_perform_modeling}data_for_repeat_modeling_plots']

    # Make two main columns.
    main_columns = st.columns(2)

    # In the first column, place the two per-model plots.
    with main_columns[0]:

        # Write a header.
        st.header('Per model type')

        # Extract the available model types.
        model_types = data_for_repeat_modeling_plots['R^2 score']['columns'][1]

        # Create columns for model selection widgets.
        widget_columns = st.columns(3, vertical_alignment='bottom')

        # In the first column, create a selectbox.
        with widget_columns[0]:
            key = st_key_prefix + 'model_type'
            if (key not in st.session_state) or (st.session_state[key] not in model_types):
                st.session_state[key] = model_types[0]
            model_type = st.selectbox(
                label="Select model type:",
                options=model_types,
                key=key,
                help="Select the model type to visualize.",
            )

        # In the second column, create a Previous button.
        with widget_columns[1]:
            st.button('Previous model type', use_container_width=True, on_click=(lambda: st.session_state.update({(st_key_prefix + 'model_type'): model_types[model_types.index(st.session_state[st_key_prefix + 'model_type']) - 1]})), disabled=(model_types.index(st.session_state[key]) == 0))

        # In the third column, create a Next button.
        with widget_columns[2]:
            st.button('Next model type', use_container_width=True, on_click=(lambda: st.session_state.update({(st_key_prefix + 'model_type'): model_types[model_types.index(st.session_state[st_key_prefix + 'model_type']) + 1]})), disabled=(model_types.index(st.session_state[key]) == (len(model_types) - 1)))

        # Obtain the actual metric used for the repeat modeling.
        actual_metric = st.session_state[f'{st_key_prefix_perform_modeling}actually_used_settings_for_perform_modeling_button']['metric']

        # Draw the R^2 plot for a selected plot type.
        repeat_modeling_results_plot(data_for_repeat_modeling_plots, repeat_modeling_results='R^2 score', title_suffix=f' for {model_type} and actual metric {actual_metric}', dims=[0, 1, 3, 2], last_axis_index=model_types.index(model_type))

        # Draw the ensemble mean squared error decomposition plot for a selected model type.
        repeat_modeling_results_plot(data_for_repeat_modeling_plots, repeat_modeling_results='Ensemble mean squared error decomposition', title_suffix=f' for {model_type}', dims=[0, 1, 3, 2], last_axis_index=model_types.index(model_type))

    # In the second column, place the two all-model plots.
    with main_columns[1]:

        # Write a header.
        st.header('All model types')

        # Draw the ensemble R^2 score plot for a selected model type.
        repeat_modeling_results_plot(data_for_repeat_modeling_plots, repeat_modeling_results='Ensemble R^2 score', title_suffix=f' for actual metric {actual_metric}')

        # Draw the ensemble error plot for a selected model type.
        repeat_modeling_results_plot(data_for_repeat_modeling_plots, repeat_modeling_results='Ensemble error')


# Run the main function.
if __name__ == "__main__":
    main()
