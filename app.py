# Import relevant libraries
import streamlit as st
import polars_data_loading as pdl
import load_and_filter_data
import perform_modeling
import visualize_model_metrics
import select_model
import study_feature_importance
import predict_on_new_data

# Set the prefix to the session state keys defined in this script.
st_key_prefix = 'app.py__'


# Define the main function.
def main():

    # Define the pages for the navigation bar.
    pg = st.navigation(
        {
            "Pages": [
                st.Page(load_and_filter_data.main, title="🔍 Load and filter data", default=True, url_path='load_and_filter_data'),
                st.Page(perform_modeling.main, title="🧠 Perform modeling", url_path='perform_modeling'),
                st.Page(visualize_model_metrics.main, title="📊 Visualize model metrics", url_path='visualize_model_metrics'),
                st.Page(select_model.main, title="🎯 Select model", url_path='select_model'),
                st.Page(study_feature_importance.main, title="🧬 Study feature importance", url_path='study_feature_importance'),
                st.Page(predict_on_new_data.main, title="🔮 Predict on new data", url_path='predict_on_new_data'),
                ],
        }
    )

    # For widget persistence between pages, we need always copy the session state to itself.
    for key in st.session_state:
        if (not key.endswith('_button')) and (not key.endswith('__do_not_persist')):
            st.session_state[key] = st.session_state[key]

    # This is needed for the st.dataframe_editor() class (https://github.com/andrew-weisman/streamlit-dataframe-editor) but is also useful for seeing where we are and where we've been.
    st.session_state['current_page_name'] = pg.url_path if pg.url_path != '' else 'Home'
    if 'previous_page_name' not in st.session_state:
        st.session_state['previous_page_name'] = st.session_state['current_page_name']

    # Load app settings. Only call it once but can't use caching per the comment in the pdl.load_app_settings() function.
    key = st_key_prefix + 'app_settings'
    if key not in st.session_state:
        st.session_state[key] = pdl.load_app_settings()

    # Set page configuration.
    app_title = st.session_state[key]['general']['app_title']
    st.set_page_config(
        page_title=app_title,
        page_icon=st.session_state[key]['general']['page_icon'],
        layout='wide'
        )
    
    # Display the CBIIT logo.
    st.logo("cbiit_logo_color2.png", size="large", link="https://datascience.cancer.gov")
    
    # Display the title of the page.
    st.title(app_title + ': ' + pg.title)

    # Display the page.
    pg.run()

    # Update the previous page location.
    st.session_state['previous_page_name'] = st.session_state['current_page_name']


# Run the main function.
if __name__ == "__main__":
    main()
