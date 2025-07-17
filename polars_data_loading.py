# Import relevant libraries.
import streamlit as st
import os
import numpy as np
import subprocess
import yaml
import polars as pl
import pandas as pd

# Set the prefix to the session state keys defined in this and other scripts.
st_key_prefix = 'polars_data_loading.py__'
st_key_prefix_load_and_filter_data = 'load_and_filter_data.py__'


# Load the app settings from the settings.yaml file. The point is that this library is totally generalizable and specific use cases can be set using the settings.yaml file. I'm not caching this because it has settings I want to go into st.set_page_config() but if I cache it then the st.cache_data decorate is actually the first Streamlit command called so I get an error saying that st.set_page_config() must be called first.
def load_app_settings(settings_filename='settings.yaml'):
    with open(settings_filename, 'r') as f:
        return yaml.safe_load(f)


# Check if the platform is NIDAP.
@st.cache_data
def platform_is_nidap():
    print('Running platform_is_nidap()...')
    return np.any(['foundry-artifacts' in x for x in subprocess.run('conda config --show channels', shell=True, capture_output=True).stdout.decode().split('\n')[1:-1]])


# Get the available datasets in the current directory or in the Foundry environment.
@st.cache_data
def get_available_pinned_datasets():
    print('Running get_available_pinned_datasets()...')
    if not platform_is_nidap():
        csv_datasets_dir = os.path.join(os.getcwd(), 'csv_datasets')
        if os.path.isdir(csv_datasets_dir):
            return os.listdir(csv_datasets_dir)
        else:
            return []
    else:
        with open(os.path.join(os.getcwd(), '.foundry', 'aliases.yml')) as f:
            return list(yaml.safe_load(f).keys())


# Load (into a polars lazyframe) the data according to the platform we're on.
# This is slow but we're not caching it because we're handling this in the main function where we may want to reload the data. Also, lazyframes are unhashable so can't use the @st.cache_data decorator anyway.
def load_pinned_data(pinned_dataset_name):
    print(f'Loading pinned dataset: {pinned_dataset_name}')
    if not platform_is_nidap():
        return pl.scan_csv(os.path.join(os.getcwd(), 'csv_datasets', pinned_dataset_name))
    else:
        from foundry.transforms import Dataset
        return Dataset.get(pinned_dataset_name).read_table(format="lazy-polars")


# Load uploaded data into a polars lazyframe. uploaded_dataset is an UploadedFile object, which is a subclass of BytesIO. No need to do this differently depending on whether we're on NIDAP. But note the file can be csv, xls, or xlsx.
# This is slow but we're not caching it because we're handling this in the main function where we may want to reload the data. Also, lazyframes are unhashable so can't use the @st.cache_data decorator anyway.
def load_uploaded_data(uploaded_dataset):
    uploaded_dataset_name = uploaded_dataset.name
    file_extension = os.path.splitext(uploaded_dataset_name)[1].lower()
    print(f'Loading uploaded dataset: {uploaded_dataset_name}')
    if file_extension == '.csv':
        # return pl.scan_csv(uploaded_dataset)  # This doesn't work with st.cache_data because buffers such as uploaded_dataset are unhashable. Hashing must occur despite the prepending underscore (in def get_lazyframe_info(loaded_dataset_name, _lf, preview_size=300, num_unique_perc=5, max_num_multiselect_options=100)) because the function *result* must be hashed when using st.cache_data and the some of the results of that function depend on _lf. pl.read_csv(uploaded_dataset).lazy() seems to fix the error: "polars.exceptions.ComputeError: the enum variant ScanSources::Buffers cannot be serialized". This makes sufficient sense per a discussion wtih ChatGPT.
        return pl.read_csv(uploaded_dataset).lazy()
    elif file_extension == '.xls' or file_extension == '.xlsx':
        return pl.read_excel(uploaded_dataset).lazy()
    else:  # This should never occur because the check is done in st.file_uploader().
        raise ValueError('Unsupported file type. Please upload a CSV or Excel file.')


# If a new dataset is requested to be reloaded, clear the state of the app.
# ingestion_source is either 'pinned' or 'uploaded'.
def conditionally_reset_session_state(ingestion_source):

    # We always want to clear the cache whether or not the dataset has changed since many of the cached functions depend on the dataset and if we hit the button then we want to load the latest version of the dataset.
    st.cache_data.clear()

    # Store the session state keys for the currently selected dataset and the currently loaded one.    
    selected_dataset_key = f'{st_key_prefix_load_and_filter_data}{ingestion_source}_dataset_name'
    loaded_dataset_key = f'{st_key_prefix_load_and_filter_data}loaded_dataset_name'

    # There's only a need to reset anything in the session state if data has ever even been loaded in the first place.
    if loaded_dataset_key in st.session_state:

        # If the desired dataset to load has completely changed (as opposed to just reloading in the same dataset), reset the session state since, e.g., we want to change the filters.
        if (st.session_state[selected_dataset_key] + f' ({ingestion_source})') != st.session_state[loaded_dataset_key]:
            print('Clearing the session state because the desired dataset is different from the one that\'s already loaded.')
            for key in st.session_state.keys():
                if key not in [selected_dataset_key, loaded_dataset_key]:
                    del st.session_state[key]

        # If the dataset is the same, we want to keep the filters, i.e., much of what is in the session state. But there are some things we want to reset, such as the filter *options* (which may change with the dataset), though not the filters themselves. Not deleting 'lazyframe_info' and 'lf' because if the button is pressed then they will be overwritten anyway, especially since the cache has been cleared.
        else:
            print('Only resetting the filter options in the session state.')
            keys_to_delete = [st_key_prefix + 'full_options_dict']
            for key in keys_to_delete:
                if key in st.session_state:
                    del st.session_state[key]


# Get information about the main lazyframe.
@st.cache_data
def get_lazyframe_info(loaded_dataset_name, _lf, preview_size=300, num_unique_perc=5, max_num_multiselect_options=100):

    # Print what we're doing.
    print(f'Running get_lazyframe_info() for dataset: {loaded_dataset_name}...')

    # Define column datatypes
    non_hashable_types = (pl.List, pl.Object)  # Even though in theory pl.Datetime is hashable, I previously had it here probably to exclude it from consideration of uniqueness below due to an error at some point. Now putting it in time_types below and letting it be hashable.
    numeric_types = (pl.Float64, pl.Int64, pl.Int32)
    string_types = (pl.Utf8, pl.Boolean)  # Probably treating booleans as a string since they can act like categoricals which we often think of as strings. I know this isn't precise but it shouldn't hurt anything.
    time_types = (pl.Time, pl.Date, pl.Datetime)

    # Get and the schema.
    schema = _lf.collect_schema()

    # Get column and row information
    all_column_names = schema.names()
    num_columns = len(all_column_names)
    num_rows = _lf.select(pl.len()).collect().item()
    numeric_columns = [col for col, dtype in schema.items() if isinstance(dtype, numeric_types)]
    string_columns = [col for col, dtype in schema.items() if isinstance(dtype, string_types)]
    time_columns = [col for col, dtype in schema.items() if isinstance(dtype, time_types)]
    non_hashable_columns = [col for col, dtype in schema.items() if isinstance(dtype, non_hashable_types)]
    hashable_columns = [column for column in all_column_names if column not in non_hashable_columns]

    # --------
    # Determine columns with few unique values, which will be the columns represented by a multiselect widget. Copied with modification from lazyframe_operations.get_columns_with_few_unique_values().

    # Determine the cutoff for the number of unique values to determine whether the corresponding field/column will be of multiselect type.
    max_num_options = int(num_unique_perc / 100 * num_rows)
    max_num_options = min(max_num_options, max_num_multiselect_options)
    
    # Run a quick first pass at determining which columns have few unique values. Basically, efficiently eliminate columns right off the bat.
    not_so_unique_expr = pl.col(hashable_columns).n_unique() < (max_num_options + 1)  # polars expression saying whether the number of unique values in the columns is less than a cutoff = max_num_options + 1
    whether_not_all_unique_df = _lf.head(max_num_options + 1).select(not_so_unique_expr).collect()  # application of this expression saying whether the number of unique values in the first *cutoff* columns is less than the cutoff. If a column returns false, it means that all the first *cutoff* values are unique. If true, then not all of the first *cutoff* values are unique; there are some duplicates in the first *cutoff* rows.
    columns_with_few_unique_values_first_pass = [ser.name for ser in whether_not_all_unique_df if ser.item()]  # obtain the columns that have any sort of potential to, in full, have fewer than the cutoff number of unique values

    # Further filter to columns that have at most max_num_options unique values in the whole dataset. This is the slower operation as it runs on the full dataset, not just on the .head(). These should be the columns used with multiselect widgets.
    multiselect_columns = [
        col
        for col in columns_with_few_unique_values_first_pass
        if _lf.select(pl.col(col).n_unique()).collect().item() <= max_num_options
    ]

    # Filter out the multiselect columns from the hashable columns. Note that these multiselect columns are of course hashable.
    numeric_columns = [column for column in numeric_columns if column not in multiselect_columns]
    string_columns = [column for column in string_columns if column not in multiselect_columns]
    time_columns = [column for column in time_columns if column not in multiselect_columns]

    # --------

    # Print the columns of the three main types.
    print(f'Numeric columns: {numeric_columns}')
    print(f'String columns: {string_columns}')
    print(f'Time columns: {time_columns}')
    print(f'Multiselect columns: {multiselect_columns}')

    # Throw a warning if some columns are non-hashable (so you can't run unique()).
    if non_hashable_columns:
        print(f'WARNING: Some columns have been excluded from the hashable_column_names return value: {non_hashable_columns}')
    else:
        print(f'All columns are hashable.')
        
    # Check that we've determined all the columns of every type.
    assert len(numeric_columns) + len(string_columns) + len(time_columns) + len(multiselect_columns) + len(non_hashable_columns) == num_columns, f'ERROR: The sum of the number of numeric, string, multiselect, and non-hashable columns does not equal the total number of columns.'

    # Obtain a dataset preview
    df_preview = _lf.head(preview_size)

    # Return all these metadata.
    return {'shape': (num_rows, num_columns), 'all_column_names': all_column_names, 'column_types': {'string': string_columns, 'numeric': numeric_columns, 'time': time_columns, 'multiselect': multiselect_columns, 'non_hashable': non_hashable_columns, 'hashable': hashable_columns}, 'schema': schema, 'df_preview': df_preview}


# Get information about a filtered lazyframe.
@st.cache_data
def get_filtered_lazyframe_info_after_filter_dict_sorting(filter_dict, loaded_dataset_name, _lf, preview_size=300):

    # Print what we're doing.
    print(f'Running get_filtered_lazyframe_info_after_filter_dict_sorting() for dataset {loaded_dataset_name} and filter_dict {filter_dict}...')

    # Get the filter expressions from the filter dictionary.
    filter_expressions = filter_dict_to_expression_list(filter_dict)

    # Perform the filtering (really just generates another lazyframe).
    if filter_expressions:
        lf_filtered = _lf.filter(filter_expressions)
    else:
        lf_filtered = _lf

    # Get the schema
    schema = lf_filtered.collect_schema()

    # Get column and row information
    num_columns = len(schema)
    num_rows = lf_filtered.select(pl.len()).collect().item()

    # Obtain a dataset preview
    df_preview = lf_filtered.head(preview_size)

    # Return all these metadata.
    return {'shape': (num_rows, num_columns), 'schema': schema, 'df_preview': df_preview}


# Ensure the filter_dict going into get_filtered_lazyframe_info_after_filter_dict_sorting is sorted so that caching remains robust (I imagine if the dictionary order is different, it won't read from the cache).
def get_filtered_lazyframe_info(filter_dict_unsorted, loaded_dataset_name, lf, preview_size=300):

    # Sort the filter dictionary since it will be an input to a cached function and we don't want different orders to essentially change the hash.
    filter_dict = dict(sorted(filter_dict_unsorted.items()))

    # Determine the lazyframe information. This is the only place this function is called.
    return get_filtered_lazyframe_info_after_filter_dict_sorting(filter_dict, loaded_dataset_name, lf, preview_size=preview_size)


# Get an eager version of the filtered dataframe.
@st.cache_data
def get_filtered_dataframe_after_filter_dict_sorting(filter_dict, loaded_dataset_name, _lf):

    # Print what we're doing.
    print(f'Running get_filtered_dataframe_after_filter_dict_sorting() for dataset {loaded_dataset_name} and filter_dict {filter_dict}...')

    # Get the filter expressions from the filter dictionary.
    filter_expressions = filter_dict_to_expression_list(filter_dict)

    # Perform the filtering (really just generates another lazyframe).
    if filter_expressions:
        lf_filtered = _lf.filter(filter_expressions)
    else:
        lf_filtered = _lf

    # Return the filtered dataframe.
    return lf_filtered.collect()


# Ensure the filter_dict going into get_filtered_dataframe_after_filter_dict_sorting is sorted so that caching remains robust (I imagine if the dictionary order is different, it won't read from the cache).
def get_filtered_dataframe(filter_dict_unsorted, loaded_dataset_name, lf):

    # Sort the filter dictionary since it will be an input to a cached function and we don't want different orders to essentially change the hash.
    filter_dict = dict(sorted(filter_dict_unsorted.items()))

    # Get the resulting eager dataframe. This is the only place this function is called.
    return get_filtered_dataframe_after_filter_dict_sorting(filter_dict, loaded_dataset_name, lf)


# Get the filtered lazyframe.
def get_filtered_lazyframe(filter_dict, lf):

    # Get the filter expressions from the filter dictionary.
    filter_expressions = filter_dict_to_expression_list(filter_dict)

    # Perform the filtering (really just generates another lazyframe).
    if filter_expressions:
        lf_filtered = lf.filter(filter_expressions)
    else:
        lf_filtered = lf

    # Return the filtered lazyframe.
    return lf_filtered


# Generate a polars expression list of filters from a dictionary of filters.
def filter_dict_to_expression_list(filter_dict):

    # Initialize the list of expressions.
    expression_list = []

    # Loop through every item (key-value pair) in the filter dictionary.
    for column, current_settings in filter_dict.items():

        # Store the filter type.
        filter_type = current_settings['type']

        # Assume we're going to add the current filter to the list of expressions.
        add_current_filter = True

        # Generate the appropriate filter expression based on the filter type.
        if filter_type == 'multiselect':
            values, nulls = current_settings['values'], current_settings['nulls']
            if values is not None and nulls is not None:
                if nulls:
                    # Combine is_in() for non-null values with is_null() using logical OR (|).
                    expression = pl.col(column).is_in(values) | pl.col(column).is_null()
                else:
                    expression = pl.col(column).is_in(values)
            else:
                add_current_filter = False
        elif filter_type == 'string':
            regex, nulls = current_settings['regex'], current_settings['nulls']
            if regex is not None and nulls is not None:
                if nulls:
                    # Combine str.contains() with is_null() using logical OR (|).
                    expression = pl.col(column).str.contains(regex) | pl.col(column).is_null()
                else:
                    # Only use str.contains() if nulls is False.
                    expression = pl.col(column).str.contains(regex)
            else:
                add_current_filter = False
        elif filter_type == 'numeric':
            min_val, max_val, nulls = current_settings['min'], current_settings['max'], current_settings['nulls']
            if min_val is not None and max_val is not None and nulls is not None:
                if nulls:
                    # Combine is_between() with is_null() using logical OR (|).
                    expression = pl.col(column).is_between(min_val, max_val) | pl.col(column).is_null()
                else:
                    # Only use is_between() if nulls is False.
                    expression = pl.col(column).is_between(min_val, max_val)
            else:
                add_current_filter = False

        # Append the expression to the list of expressions.
        if add_current_filter:
            expression_list.append(expression)

    # Return the list of expressions.
    return expression_list


# Ensure consistency between the selected filter columns and the filter dictionary and session state.
# This is mainly called back from modification of the filtering columns.
def manage_filter_dict(loaded_dataset_name, lf, string_columns, numeric_columns, multiselect_columns):

    # If the filter dictionary hasn't yet been created, create it.
    key = st_key_prefix + 'filter_dict'
    if key not in st.session_state:
        st.session_state[key] = {}

    # If the full options dictionary hasn't yet been created, create it.
    key = st_key_prefix + 'full_options_dict'
    if key not in st.session_state:
        st.session_state[key] = {}

    # From the multiselect widget that determines which columns to filter on, return the value of the widget.
    filtering_columns = st.session_state[f'{st_key_prefix_load_and_filter_data}filtering_columns']

    # Get a *copy* of the keys of the filter dictionary.
    filter_dict_keys = list(st.session_state[st_key_prefix + 'filter_dict'].keys())

    # If a column is in the filter dictionary but not in the filtering columns (the latter of which is ground truth), delete it from the filter dictionary and delete the associated keys from the session state.
    for column in filter_dict_keys:
        if column not in filtering_columns:

            # Delete the corresponding entry from the filter dictionary.
            del st.session_state[st_key_prefix + 'filter_dict'][column]

            # Delete any associated session state keys.
            keys_to_delete = [
                st_key_prefix + f'{column}_filtering_string',
                st_key_prefix + f'{column}_minimum_value_filter',
                st_key_prefix + f'{column}_maximum_value_filter',
                st_key_prefix + f'{column}_choices_filter',
                st_key_prefix + f'{column}_filtering_string_reset_button',
                st_key_prefix + f'{column}_minmax_values_filter_reset_button',
                st_key_prefix + f'{column}_choices_filter_reset_button',
                st_key_prefix + f'{column}_choices_filter_apply_button',
                st_key_prefix + f'{column}_minmax_values_filter_apply_button',
                st_key_prefix + f'{column}_filtering_string_apply_button',
                st_key_prefix + f'{column}_minmax_values_filter_push_to_selection_button',
                st_key_prefix + f'{column}_choices_filter_push_to_selection_button',
                st_key_prefix + f'{column}_multiselect_allow_null_values',
                st_key_prefix + f'{column}_string_allow_null_values',
                st_key_prefix + f'{column}_numeric_allow_null_values',
                st_key_prefix + f'{column}_filtering_string_push_to_selection_button',
                ]
            for key in keys_to_delete:
                if key in st.session_state:
                    del st.session_state[key]

    # If a column is in the filtering columns but not in the filter dictionary, add it to the filter dictionary.
    for column in filtering_columns:

        # If the column is desired for filtering but doesn't already have a filter in the filter dictionary, i.e., is brand new...
        # Note that default values set here should result in a filter with all full-dataset values included, e.g., '' for string filters, min/max of the dataset for numeric filters, and all possible values in the dataset for multiselect filters. Also, null = True. These defaults are also set in generate_filter_widgets() (see the reset buttons).
        if column not in filter_dict_keys:
            
            # If the column is a string column, add it to the filter dictionary with a default regex.
            if column in string_columns:
                column_options = determine_column_options({column: {'type': 'string', 'regex': None, 'nulls': None}}, loaded_dataset_name, lf)
                st.session_state[st_key_prefix + 'filter_dict'][column] = {'type': 'string', 'regex': '', 'nulls': True}

            # If the column is a numeric column, add it to the filter dictionary with default min and max values.
            elif column in numeric_columns:
                column_options = determine_column_options({column: {'type': 'numeric', 'min': None, 'max': None, 'nulls': None}}, loaded_dataset_name, lf)
                st.session_state[st_key_prefix + 'filter_dict'][column] = {'type': 'numeric', 'min': column_options[column]['min'], 'max': column_options[column]['max'], 'nulls': True}

            # If the column is a multiselect column, add it to the filter dictionary with default values.
            elif column in multiselect_columns:
                column_options = determine_column_options({column: {'type': 'multiselect', 'values': None, 'nulls': None}}, loaded_dataset_name, lf)
                st.session_state[st_key_prefix + 'filter_dict'][column] = {'type': 'multiselect', 'values': column_options[column]['values'], 'nulls': True}

            # Save the full dataset options for the current column.
            if column not in st.session_state[st_key_prefix + 'full_options_dict']:
                st.session_state[st_key_prefix + 'full_options_dict'][column] = column_options[column]

    # Repopulate full_options_dict if it's been reset using the reload button and there's at least one filter.
    if st.session_state[st_key_prefix + 'full_options_dict'] == {} and st.session_state[st_key_prefix + 'filter_dict'] != {}:
        for column in filtering_columns:
            if column in string_columns:
                column_options = determine_column_options({column: {'type': 'string', 'regex': None, 'nulls': None}}, loaded_dataset_name, lf)
            elif column in numeric_columns:
                column_options = determine_column_options({column: {'type': 'numeric', 'min': None, 'max': None, 'nulls': None}}, loaded_dataset_name, lf)
            elif column in multiselect_columns:
                column_options = determine_column_options({column: {'type': 'multiselect', 'values': None, 'nulls': None}}, loaded_dataset_name, lf)
            st.session_state[st_key_prefix + 'full_options_dict'][column] = column_options[column]


# Update the filter in the filter dictionary specific to the column whose filter settings were just modified.
# This is called back from the individual filter widgets.
def update_filter_entry(column, filter_type, specific_values=None):

    # For string filters, update the filter dictionary with the new regex.
    if filter_type == 'string':
        if specific_values is None:  # otherwise, set regex to the default value of '', not the widget value
            regex = st.session_state[st_key_prefix + f'{column}_filtering_string']
            nulls = st.session_state[st_key_prefix + f'{column}_string_allow_null_values']
        else:
            regex, nulls = specific_values
        st.session_state[st_key_prefix + 'filter_dict'][column] = {'type': filter_type, 'regex': regex, 'nulls': nulls}

    # For numeric filters, update the filter dictionary with the new min and max values. Check that max >= min; otherwise, swap them.
    elif filter_type == 'numeric':
        if specific_values is None:
            min_val = st.session_state[st_key_prefix + f'{column}_minimum_value_filter']
            max_val = st.session_state[st_key_prefix + f'{column}_maximum_value_filter']
            nulls = st.session_state[st_key_prefix + f'{column}_numeric_allow_null_values']
            if min_val > max_val:
                min_val, max_val = max_val, min_val
                st.session_state[st_key_prefix + f'{column}_minimum_value_filter'] = min_val
                st.session_state[st_key_prefix + f'{column}_maximum_value_filter'] = max_val
        else:
            min_val, max_val, nulls = specific_values
        st.session_state[st_key_prefix + 'filter_dict'][column] = {'type': filter_type, 'min': min_val, 'max': max_val, 'nulls': nulls}

    # For multiselect filters, update the filter dictionary with the new values.
    elif filter_type == 'multiselect':
        if specific_values is None:
            values = st.session_state[st_key_prefix + f'{column}_choices_filter']
            nulls = st.session_state[st_key_prefix + f'{column}_multiselect_allow_null_values']
        else:
            values, nulls = specific_values
        st.session_state[st_key_prefix + 'filter_dict'][column] = {'type': filter_type, 'values': values, 'nulls': nulls}


# From a *filtered* lazyframe, get the corresponding possible column options.
# This should only ever be called from determine_widget_parameters() below.
@st.cache_data
def determine_column_options_after_filter_dict_sorting(filter_dict, loaded_dataset_name, _lf):

    # Print what we're doing.
    print(f'Running determine_column_options_after_filter_dict_sorting() for dataset {loaded_dataset_name} and filter_dict {filter_dict}...')

    # Get the filter expressions from the filter dictionary.
    filter_expressions = filter_dict_to_expression_list(filter_dict)

    # Perform the filtering (really just generates another lazyframe).
    if filter_expressions:
        lf_filtered = _lf.filter(filter_expressions)
    else:
        lf_filtered = _lf

    # Initialize the holder of the widget inputs.
    column_options = {}
    
    # For every column on which we want to perform filtering...
    for column, current_settings in filter_dict.items():
        
        # Get the unique values in the current column.
        if current_settings['type'] == 'multiselect':
            values = lf_filtered.select(
                pl.col(column).unique()
                ).collect().to_series().to_list()
            nulls = None in values
            values = [value for value in values if value is not None] if nulls else values  # Remove None values from the list of unique values since we handle nulls separately.
            column_options[column] = {
                'values': values,
                'nulls': nulls,
            }

        # Don't do anything for string filters as those widgets have no inputs (they're just text boxes).
        elif current_settings['type'] == 'string':
            nulls = lf_filtered.select(
                pl.col(column).is_null().any()
                ).collect().item()
            column_options[column] = {
                'regex': '',
                'nulls': nulls,
            }

        # For numeric filters, get the min and max values of the current column.
        elif current_settings['type'] == 'numeric':
            min_max_values = lf_filtered.select(
                pl.col(column).min().alias('min'),
                pl.col(column).max().alias('max')
                ).collect()  # If this ever crashes, maybe do the min() and the max() in separate lines. If that crashes, allocate more memory. If that crashes, we're probably in trouble.
            nulls = lf_filtered.select(
                pl.col(column).is_null().any()
                ).collect().item()
            column_options[column] = {
                'min': min_max_values['min'][0],
                'max': min_max_values['max'][0],
                'nulls': nulls,
            }

    # Return the widget inputs dictionary.
    return column_options


# Ensure the filter_dict going into determine_widget_parameters_after_filter_dict_sorting is sorted so that caching remains robust (I imagine if the dictionary order is different, it won't read from the cache).
def determine_column_options(filter_dict_unsorted, loaded_dataset_name, lf):

    # Sort the filter dictionary since it will be an input to a cached function and we don't want different orders to essentially change the hash.
    filter_dict = dict(sorted(filter_dict_unsorted.items()))

    # Determine the widget parameters. This is the only place this function is called.
    return determine_column_options_after_filter_dict_sorting(filter_dict, loaded_dataset_name, lf)


# Render filtering widgets for the selected columns for filtering.
def generate_filter_widgets(filter_dict, full_column_options, filtering_columns, string_columns, numeric_columns, multiselect_columns, loaded_dataset_name, lf):

    # From the filter dictionary, get the widget parameters for every filter.
    filtered_column_options = determine_column_options(filter_dict, loaded_dataset_name, lf)

    # For each of the columns desired for filtering...
    for column in filtering_columns:

        # Display the column name as a subheader.
        st.markdown(f"<h3 style='color: orange;'>{column}</h3>", unsafe_allow_html=True)

        # If the column is a string column, display a text box for the filtering string.
        if column in string_columns:

            # Obtain the min/max range for the current set of filters.
            regex_filtered = filtered_column_options[column]['regex']
            nulls_filtered = filtered_column_options[column]['nulls']

            # Get the current regex value from the filter dictionary and render the text box for the filtering string.
            key = st_key_prefix + f'{column}_filtering_string'
            st.session_state[key] = filter_dict[column]['regex']
            st.text_input(f'Regex filter:', key=key, on_change=update_filter_entry, args=(column, 'string'))

            # Add checkbox for null values.
            key = st_key_prefix + f'{column}_string_allow_null_values'
            st.session_state[key] = filter_dict[column]['nulls']
            st.checkbox('Allow null values', key=key, on_change=update_filter_entry, args=(column, 'string'))

            # Dummy button to apply changes (does nothing, but users may like this).
            st.button('Apply filter', key=st_key_prefix + f'{column}_filtering_string_apply_button')

            # Display the min/max values for the filtered column.
            with st.expander('Regex/null values for filtered column:', expanded=False):
                nulls_str = 'Nulls present: ' + ('Yes' if nulls_filtered else 'No')
                st.write(f'Regex: {regex_filtered}, {nulls_str}')
                st.button('⬆️ Push to selection', key=st_key_prefix + f'{column}_filtering_string_push_to_selection_button', on_click=update_filter_entry, args=(column, 'string'), kwargs={'specific_values': (regex_filtered, nulls_filtered)})

            # Add a button to reset the widget settings.
            st.button('Reset filter', key=(key + '_reset_button'), on_click=update_filter_entry, args=(column, 'string'), kwargs={'specific_values': ('', True)})

        # If the column is a numeric column, display number input widgets for the minimum and maximum values.
        elif column in numeric_columns:

            # Obtain the min/max range for the current set of filters.
            min_value_filtered = filtered_column_options[column]['min']
            max_value_filtered = filtered_column_options[column]['max']
            nulls_filtered = filtered_column_options[column]['nulls']

            # Obtain the min/max range for the full dataset.
            min_value_full = full_column_options[column]['min']
            max_value_full = full_column_options[column]['max']

            # Display their values on screen.
            st.write(f'Min/max values for full column: [{min_value_full}, {max_value_full}]')

            # Get the current min value from the filter dictionary and render the number input widget for the minimum value.
            key = st_key_prefix + f'{column}_minimum_value_filter'
            st.session_state[key] = filter_dict[column]['min']
            st.number_input(f'Minimum value filter:', key=key, min_value=min_value_full, max_value=max_value_full, on_change=update_filter_entry, args=(column, 'numeric'))

            # Get the current max value from the filter dictionary and render the number input widget for the maximum value.
            key = st_key_prefix + f'{column}_maximum_value_filter'
            st.session_state[key] = filter_dict[column]['max']
            st.number_input(f'Maximum value filter:', key=key, min_value=min_value_full, max_value=max_value_full, on_change=update_filter_entry, args=(column, 'numeric'))

            # Add checkbox for null values.
            key = st_key_prefix + f'{column}_numeric_allow_null_values'
            st.session_state[key] = filter_dict[column]['nulls']
            st.checkbox('Allow null values', key=key, on_change=update_filter_entry, args=(column, 'numeric'))

            # Dummy button to apply changes (does nothing, but users may like this).
            st.button('Apply filter', key=st_key_prefix + f'{column}_minmax_values_filter_apply_button')

            # Display the min/max values for the filtered column.
            with st.expander('Min/max/null values for filtered column:', expanded=False):
                nulls_str = 'Nulls present: ' + ('Yes' if nulls_filtered else 'No')
                st.write(f'Min/max: [{min_value_filtered}, {max_value_filtered}], {nulls_str}')
                st.button('⬆️ Push to selection', key=st_key_prefix + f'{column}_minmax_values_filter_push_to_selection_button', on_click=update_filter_entry, args=(column, 'numeric'), kwargs={'specific_values': (min_value_filtered, max_value_filtered, nulls_filtered)})

            # Add a button to reset the widget settings.
            st.button('Reset filter', key=st_key_prefix + f'{column}_minmax_values_filter_reset_button', on_click=update_filter_entry, args=(column, 'numeric'), kwargs={'specific_values': (min_value_full, max_value_full, True)})

        # If the column is a multiselect column, display a multiselect widget for the filtering values.
        elif column in multiselect_columns:

            # Obtain the options for the current set of filters.
            values_filtered = filtered_column_options[column]['values']
            nulls_filtered = filtered_column_options[column]['nulls']

            # Obtain the options for the full dataset.
            values_full = full_column_options[column]['values']

            # Get the current values from the filter dictionary and render the multiselect widget for the filtering values.
            key = st_key_prefix + f'{column}_choices_filter'
            st.session_state[key] = filter_dict[column]['values']
            st.multiselect(f'Choices filter:', values_full, key=key, on_change=update_filter_entry, args=(column, 'multiselect'))

            # Add checkbox for null values.
            key = st_key_prefix + f'{column}_multiselect_allow_null_values'
            st.session_state[key] = filter_dict[column]['nulls']
            st.checkbox('Allow null values', key=key, on_change=update_filter_entry, args=(column, 'multiselect'))

            # Dummy button to apply changes (does nothing, but users may like this).
            st.button('Apply filter', key=st_key_prefix + f'{column}_choices_filter_apply_button')

            # Display the values for the filtered column.
            with st.expander('Values for filtered column:', expanded=False):
                nulls_str = 'Nulls present: ' + ('Yes' if nulls_filtered else 'No')
                st.dataframe(
                    pd.DataFrame({'Filtered values': [values_filtered]}),
                    hide_index=True)
                st.write(nulls_str)
                st.button('⬆️ Push to selection', key=(key + '_push_to_selection_button'), on_click=update_filter_entry, args=(column, 'multiselect'), kwargs={'specific_values': (values_filtered, nulls_filtered)})

            # Add a button to reset the widget settings.
            st.button('Reset filter', key=(key + '_reset_button'), on_click=update_filter_entry, args=(column, 'multiselect'), kwargs={'specific_values': (values_full, True)})


# Based on the desired frame type, generate the dataframe or lazyframe. Can't cache this because input and potentially output both include non-hashable object (lazyframe).
def generate_frame_of_certain_type(frame_type_to_use, filter_dict, loaded_dataset_name, lf):
    print(f'Generating {frame_type_to_use.lower()} filtered frame...')
    if frame_type_to_use == 'Eager':
        return get_filtered_dataframe(filter_dict, loaded_dataset_name, lf)
    elif frame_type_to_use == 'Lazy':
        return get_filtered_lazyframe(filter_dict, lf)
    elif frame_type_to_use == 'Pandas':
        return get_filtered_dataframe(filter_dict, loaded_dataset_name, lf).to_pandas(use_pyarrow_extension_array=True)


# If filtered_frame is a tuple, use it as arguments to generate the frame. Otherwise, use it as the frame itself. Also cannot be cached because filtered_frame could contain/be a lazyframe and it could also return one.
def generate_or_return_frame(filtered_frame):
    if isinstance(filtered_frame, tuple):  # filtered_frame tells us how to generate the frame via a bunch of settings
        print('Generating frame...')
        return generate_frame_of_certain_type(*filtered_frame)
    else:  # filtered_frame is the filtered frame itself
        print('Returning already generated frame...')
        return filtered_frame
