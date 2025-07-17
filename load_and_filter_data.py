# Import relevant libraries.
import streamlit as st
import polars_data_loading as pdl
import copy

# Set the prefix to the session state keys defined in this and other scripts.
st_key_prefix = 'load_and_filter_data.py__'
st_key_prefix_app = 'app.py__'
st_key_prefix_polars_data_loading = 'polars_data_loading.py__'


# Load a pinned dataset.
def load_pinned_dataset(pinned_dataset_name, preview_size, num_unique_perc, max_num_multiselect_options):
    print('Running load_pinned_dataset.')
    st.session_state[st_key_prefix + 'lf'] = pdl.load_pinned_data(pinned_dataset_name)
    st.session_state[st_key_prefix + 'lazyframe_info'] = pdl.get_lazyframe_info(
        pinned_dataset_name,
        st.session_state[st_key_prefix + 'lf'],
        preview_size=preview_size,
        num_unique_perc=num_unique_perc,
        max_num_multiselect_options=max_num_multiselect_options
        )
    st.session_state[st_key_prefix + 'loaded_dataset_name'] = pinned_dataset_name + ' (pinned)'
    st.success('Pinned dataset loaded.')


# Load an uploaded dataset.
def load_uploaded_dataset(uploaded_dataset, uploaded_dataset_name, preview_size, num_unique_perc, max_num_multiselect_options):
    st.session_state[st_key_prefix + 'lf'] = pdl.load_uploaded_data(uploaded_dataset)
    st.session_state[st_key_prefix + 'lazyframe_info'] = pdl.get_lazyframe_info(
        uploaded_dataset_name,
        st.session_state[st_key_prefix + 'lf'],
        preview_size=preview_size,
        num_unique_perc=num_unique_perc,
        max_num_multiselect_options=max_num_multiselect_options
        )
    st.session_state[st_key_prefix + 'loaded_dataset_name'] = uploaded_dataset_name + ' (uploaded)'
    st.success('Uploaded dataset loaded.')


# Create app functionality to load a "pinned" dataset.
def pinned_dataset_loading():

    # Select a pinned dataset.
    available_pinned_datasets = pdl.get_available_pinned_datasets()
    key = st_key_prefix + 'pinned_dataset_name'
    if key not in st.session_state:
        st.session_state[key] = available_pinned_datasets[0] if available_pinned_datasets else None
    pinned_dataset_name = st.selectbox(':a: Select registered dataset...', options=available_pinned_datasets, key=key)

    # Load the selected pinned dataset.
    key = st_key_prefix + 'actually_used_settings_for_load_pinned_dataset_button'
    button_columns = st.columns(2)
    with button_columns[0]:
        if st.button(':arrows_counterclockwise: Load registered dataset', use_container_width=True, on_click=pdl.conditionally_reset_session_state, args=('pinned',), disabled=(pinned_dataset_name is None)):
            load_pinned_dataset_settings = {
                'pinned_dataset_name': pinned_dataset_name,
                'preview_size': st.session_state[f'{st_key_prefix_app}app_settings']['get_lazyframe_info']['preview_size'],
                'num_unique_perc': st.session_state[f'{st_key_prefix_app}app_settings']['get_lazyframe_info']['num_unique_perc'],
                'max_num_multiselect_options': st.session_state[f'{st_key_prefix_app}app_settings']['get_lazyframe_info']['max_num_multiselect_options']
            }
            st.session_state[key] = load_pinned_dataset_settings
            load_pinned_dataset(**load_pinned_dataset_settings)
    current_settings = str(st.session_state[key]) if key in st.session_state else None
    with button_columns[1]:
        with st.popover('Currently loaded registered data', use_container_width=True):
            st.write(current_settings)


# Create app functinality to upload a dataset.
def uploaded_dataset_loading():

    # Upload a dataset.
    uploaded_dataset = st.file_uploader('... OR, :b: Upload dataset:', type=['csv', 'xls', 'xlsx'])
    if uploaded_dataset is not None:
        uploaded_dataset_name = uploaded_dataset.name
    else:
        uploaded_dataset_name = None
    st.session_state[st_key_prefix + 'uploaded_dataset_name'] = uploaded_dataset_name

    # Load the uploaded dataset.
    key = st_key_prefix + 'actually_used_settings_for_load_uploaded_dataset_button'
    button_columns = st.columns(2)
    with button_columns[0]:
        if st.button(':arrows_counterclockwise: Load uploaded dataset', use_container_width=True, on_click=pdl.conditionally_reset_session_state, args=('uploaded',), disabled=(uploaded_dataset is None)):
            load_uploaded_dataset_settings = {
                'uploaded_dataset_name': uploaded_dataset_name,
                'preview_size': st.session_state[f'{st_key_prefix_app}app_settings']['get_lazyframe_info']['preview_size'],
                'num_unique_perc': st.session_state[f'{st_key_prefix_app}app_settings']['get_lazyframe_info']['num_unique_perc'],
                'max_num_multiselect_options': st.session_state[f'{st_key_prefix_app}app_settings']['get_lazyframe_info']['max_num_multiselect_options']
            }
            st.session_state[key] = load_uploaded_dataset_settings
            load_uploaded_dataset(uploaded_dataset, **load_uploaded_dataset_settings)
    current_settings = str(st.session_state[key]) if key in st.session_state else None
    with button_columns[1]:
        with st.popover('Currently loaded uploaded data', use_container_width=True):
            st.write(current_settings)


# Save the frame for analysis.
def save_frame_for_analysis(lf, frame_type_to_use, filter_dict, loaded_dataset_name, dataframe_generating_mechanism):
    args_tuple = (frame_type_to_use, filter_dict, loaded_dataset_name, lf)
    if dataframe_generating_mechanism == 'From dataset':
        st.session_state[st_key_prefix + 'filtered_frame'] = args_tuple
    elif dataframe_generating_mechanism == 'From memory':
        frame = pdl.generate_frame_of_certain_type(*args_tuple)
        st.session_state[st_key_prefix + 'filtered_frame'] = frame
    st.success('Data registered for analysis. Please proceed to the next page.')


# Filtering UI functionality. I had separated it out to here because I was temporarily decorating with st.fragment.
def filtering(lazyframe_info, loaded_dataset_name, lf):

    # Set up main columns.
    main_cols = st.columns([1/3, 2/3], border=True)

    # In the first column, set up the filtering options.
    with main_cols[0]:
        st.header(':two: Filter the data')

        # Get shortcuts to the columns in the dataset.
        string_columns = lazyframe_info['column_types']['string']
        numeric_columns = lazyframe_info['column_types']['numeric']
        multiselect_columns = lazyframe_info['column_types']['multiselect']
        hashable_columns = lazyframe_info['column_types']['hashable']

        # Select columns on which to perform filtering.
        key = st_key_prefix + 'filtering_columns'
        if key not in st.session_state:
            st.session_state[key] = []
        filtering_columns = st.multiselect('Select columns on which to perform filtering:', options=hashable_columns, key=key, on_change=pdl.manage_filter_dict, args=(loaded_dataset_name, lf, string_columns, numeric_columns, multiselect_columns))

        # Ensure the filter dictionary is initialized.
        if (f'{st_key_prefix_polars_data_loading}filter_dict' not in st.session_state) or (f'{st_key_prefix_polars_data_loading}full_options_dict' not in st.session_state):
            pdl.manage_filter_dict(loaded_dataset_name, lf, string_columns, numeric_columns, multiselect_columns)

        # Render the corresponding filter widgets.
        pdl.generate_filter_widgets(st.session_state[f'{st_key_prefix_polars_data_loading}filter_dict'], st.session_state[f'{st_key_prefix_polars_data_loading}full_options_dict'], filtering_columns, string_columns, numeric_columns, multiselect_columns, loaded_dataset_name, lf)

    # In the second column, display the filtered dataset and perform downstream analysis.
    with main_cols[1]:
        st.header('Filtered dataset')

        # Output the filtered dataset properties.
        filtered_lazyframe_info = pdl.get_filtered_lazyframe_info(
            st.session_state[f'{st_key_prefix_polars_data_loading}filter_dict'],
            loaded_dataset_name,
            lf,
            preview_size=st.session_state[f'{st_key_prefix_app}app_settings']['get_filtered_lazyframe_info']['preview_size']
            )
        st.markdown(f'''
                    Properties:
                    - Rows: `{filtered_lazyframe_info['shape'][0]:,}`
                    - Columns: `{filtered_lazyframe_info['shape'][1]:,}`
                    ''')
        
        # Output the filtered dataset preview.
        with st.expander('Preview:', expanded=False):
            st.write(filtered_lazyframe_info['df_preview'])

        # Set a row limit at which to hesitate running downstream analysis.
        key = st_key_prefix + 'row_limit_to_hesitate'
        if key not in st.session_state:
            st.session_state[key] = st.session_state[f'{st_key_prefix_app}app_settings']['general']['analysis_hesitation_row_limit_value']
        row_limit_to_hesitate = st.number_input(
            'Row limit at which to hesitate registering the filtered data for analysis:',
            min_value=0,
            step=st.session_state[f'{st_key_prefix_app}app_settings']['general']['analysis_hesitation_row_limit_step'],
            key=key,
            help="If the number of rows in the filtered dataset exceeds this limit, you'll be asked to confirm that you want to register the data for analysis. You likely don't want to run analysis on a huge dataset!",
            )

        # If the number of rows in the filtered dataset exceeds the limit, warn the user and ask if they really want to proceed with downstream analysis.
        many_rows = filtered_lazyframe_info['shape'][0] >= row_limit_to_hesitate
        if many_rows:
            st.warning(f'The number of rows in the filtered dataset ({filtered_lazyframe_info["shape"][0]}) exceeds the limit of {row_limit_to_hesitate} rows. Consider filtering further for downstream analysis.')
            key = st_key_prefix + 'register_data_for_analysis'
            if key not in st.session_state:
                st.session_state[key] = 'No'
            register_data_for_analysis = st.radio('Actually register data for analysis:', ['Yes', 'No'], key=key)

        # If they don't want to perform downstream analysis, then stop here.
        if not ((many_rows and register_data_for_analysis == 'Yes') or not many_rows):
            st.warning('No registration of data for analysis will be performed.')
            return

        # Header for downstream analysis.
        st.subheader(':three: Register filtered data for analysis')

        # Allow user to select which frame type to use: polars dataframe or lazyframe.
        key = st_key_prefix + 'frame_type_to_use'
        if key not in st.session_state:
            st.session_state[key] = 'Eager'
        # frame_type_to_use = st.radio('Frame type to use:', ['Eager', 'Lazy', 'Pandas'], key=key)
        frame_type_to_use = st.session_state[key]

        # Allow the user to select whether they want the resulting dataframe from the dataset itself or from memory.
        key = st_key_prefix + 'dataframe_generating_mechanism'
        if key not in st.session_state:
            st.session_state[key] = 'From memory'
        # dataframe_generating_mechanism = st.radio('Mechanism for generating frame:', ['From dataset', 'From memory'], key=key)
        dataframe_generating_mechanism = st.session_state[key]

        # Return the dataframe using the desired mechanism.
        key = st_key_prefix + 'actually_used_settings_for_save_frame_for_analysis_button'
        if st.button('Save frame for analysis', type="primary"):
            save_frame_for_analysis_settings = {
                'frame_type_to_use': frame_type_to_use,
                'filter_dict': copy.deepcopy(st.session_state[f'{st_key_prefix_polars_data_loading}filter_dict']),
                'loaded_dataset_name': loaded_dataset_name,
                'dataframe_generating_mechanism': dataframe_generating_mechanism
            }
            st.session_state[key] = save_frame_for_analysis_settings
            save_frame_for_analysis(lf, **save_frame_for_analysis_settings)
        current_settings = str(st.session_state[key]) if key in st.session_state else None
        with st.popover('Currently saved frame'):
            st.write(current_settings)


# Main function definition.
def main():

    # Page information.
    st.write("This page allows you to load a dataset and filter it down to the rows containing the data that you wish to model. (:three: steps total.)")

    # Data loading options.
    data_loading_cols = st.columns([1/3, 2/3], border=False)
    with data_loading_cols[0]:
        with st.columns(1, border=True)[0]:
            st.header(':one: Load data')

            # Functionality to load a pinned dataset.
            pinned_dataset_loading()

            # Functionality to upload a dataset.
            uploaded_dataset_loading()

            # Ensure data have been loaded.
            if st_key_prefix + 'lf' not in st.session_state:
                st.warning('Please load a registered dataset or upload one.')
                return

            # Retrieve the selected dataset.
            lf = st.session_state[st_key_prefix + 'lf']
            lazyframe_info = st.session_state[st_key_prefix + 'lazyframe_info']
            loaded_dataset_name = st.session_state[st_key_prefix + 'loaded_dataset_name']  # Remember loaded_dataset_name will have " (pinned)" or " (uploaded)" appended to it so no need to change this variable name. Also, this helps to elucidate which data is actually loaded, including whether it is pinned or uploaded.

    with data_loading_cols[1]:
        with st.columns(1, border=True)[0]:
            st.header('Loaded dataset')

            # Display the loaded dataset properties.
            st.markdown(f'''
                        Properties:
                        - Name: `{loaded_dataset_name}`
                        - Rows: `{lazyframe_info['shape'][0]:,}`
                        - Columns: `{lazyframe_info['shape'][1]:,}`
                        ''')
            
            # Dataset preview.
            with st.expander('Preview:', expanded=False):
                st.write(lazyframe_info['df_preview'])

    # Run the "Filters" and "Filtered dataset" sections. Running as a fragment to avoid modifying anything that is logically prior such as the sorting of the columns in the loaded dataset preview.
    filtering(lazyframe_info, loaded_dataset_name, lf)


# Run the main function
if __name__ == '__main__':
    main()
