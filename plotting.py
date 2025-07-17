import plotly.graph_objs as go
import plotly.colors
import plotly.express as px


# Index the last axis of a numpy array if it has more dimensions than expected.
def index_last_axis(arr, correct_ndim, my_index=9999):
    actual_ndim = arr.ndim
    if actual_ndim == correct_ndim:
        return arr
    elif actual_ndim == correct_ndim + 1:
        return arr[..., my_index]
    else:  # error
        raise ValueError("Incompatible array shape and correct_ndim setting.")


def plot_ensemble_metrics(data_dict, xaxis_title="Number of models", yaxis_title="Ensemble error", title="Ensemble error vs. number of models", plot_type='line', dims=[0, 1, 2], last_axis_index=9999):
    """
    Plots mean lines with shaded min/max areas for each series in a 2D numpy array.
    Args:
        data_dict: dict with keys 'Mean', 'Min', 'Max', 'columns'.
            - 'Mean', 'Min', 'Max': 2D numpy arrays of shape (num_x, num_series)
            - 'columns': [x_values, series_names]
    """

    # Create a copy of the data dictionary to avoid modifying the original.
    data_dict = data_dict.copy()

    # Generalize the functionality in the following blocks to larger-dimensional arrays.
    smaller_shifted_dims = [i - 1 for i in dims[1:]]
    data_dict['columns'] = [data_dict['columns'][i] for i in smaller_shifted_dims[0:2]]
    data_dict['Mean'] = index_last_axis(data_dict['Mean'].transpose(tuple(smaller_shifted_dims)), 2, last_axis_index)
    data_dict['Std'] = index_last_axis(data_dict['Std'].transpose(tuple(smaller_shifted_dims)), 2, last_axis_index)
    data_dict['Min'] = index_last_axis(data_dict['Min'].transpose(tuple(smaller_shifted_dims)), 2, last_axis_index)
    data_dict['Max'] = index_last_axis(data_dict['Max'].transpose(tuple(smaller_shifted_dims)), 2, last_axis_index)
    data_dict['All repetitions'] = index_last_axis(data_dict['All repetitions'].transpose(tuple(dims)), 3, last_axis_index)

    # Extract the data from the dictionary.
    mean_ = data_dict['Mean']
    min_ = data_dict['Min']
    max_ = data_dict['Max']
    all_ = data_dict['All repetitions']
    x_values = data_dict['columns'][0]
    series_names = data_dict['columns'][1]

    # Use Plotly's qualitative color palette.
    palette = plotly.colors.qualitative.Plotly
    n_colors = len(palette)

    # Create a figure.
    fig = go.Figure()

    # Loop through each series.
    for iseries, series_name in enumerate(series_names):

        # Get a color for the current series.
        color = palette[iseries % n_colors]
        rgb = plotly.colors.hex_to_rgb(color)

        # Line plot with shaded min/max area.
        if plot_type == 'line':

            # Shaded area between min and max.
            fig.add_trace(
                go.Scatter(
                    x=list(x_values) + list(x_values)[::-1],
                    y=list(max_[:, iseries]) + list(min_[:, iseries])[::-1],
                    fill='toself',
                    fillcolor=f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.15)',  # lighter shade
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False,
                    name=series_name,
                    legendgroup=series_name,
                )
            )

            # Mean line.
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=mean_[:, iseries],
                    mode='lines+markers',
                    name=series_name,
                    line=dict(color=color, width=2),
                    legendgroup=series_name,
                )
            )

        # Box plot for each x value.
        elif plot_type == 'box':
            for ix, x in enumerate(x_values):
                fig.add_trace(
                    go.Box(
                        y=all_[:, ix, iseries],
                        x=[x] * all_.shape[0],
                        name=series_name,
                        marker_color=color,
                        boxmean=True,
                        showlegend=(ix == 0),  # Only show legend once per series
                        legendgroup=series_name,
                    )
                )

        # Violin plot for each x value.
        elif plot_type == 'violin':
            for ix, x in enumerate(x_values):
                fig.add_trace(
                    go.Violin(
                        y=all_[:, ix, iseries],
                        x=[x] * all_.shape[0],
                        name=series_name,
                        line_color=color,
                        meanline_visible=True,
                        showlegend=(ix == 0),  # Only show legend once per series
                        legendgroup=series_name,
                    )
                )

    # Add the titles.
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )

    # Return the figure.
    return fig


# Plot a plotly bar chart for feature importance.
def plot_feature_importance(df, x='feature', y='mean_abs_shap', title='Mean absolute SHAP value per feature', labels={'mean_abs_shap': 'Mean(|SHAP value|)', 'feature': 'Feature'}):
    fig = px.bar(df, x=x, y=y, title=title, labels=labels)
    fig.update_layout(xaxis={'categoryorder': 'total descending'})
    return fig


def plot_ensemble_residuals_histogram(df_residuals, num_bins=20, title='Histogram of ensemble residuals'):
    """
    Plots a histogram of ensemble residuals using Plotly Express.
    Args:
        df_residuals: pandas DataFrame with a column 'Ensemble residuals'.
        num_bins: Number of bins for the histogram.
        title: Title for the plot.
    Returns:
        Plotly Figure object.
    """
    fig = px.histogram(
        df_residuals,
        x='Ensemble residuals',
        nbins=num_bins,
        title=title,
        labels={'Ensemble residuals': 'Ensemble residuals', 'count': 'Count'}
    )
    return fig


def plot_predictions_with_error_bars(
    df,
    x_col,
    prediction_methods,
    method_to_col_map,
    method_to_low_col_map,
    method_to_high_col_map,
    title="Predictions with Error Bars",
    xaxis_title=None,
    yaxis_title="Prediction",
    jitter_strength=0.15,
):
    """
    Plots predictions with error bars for multiple methods, supporting numeric or categorical x-axis.
    Args:
        df: DataFrame containing features and prediction columns.
        x_col: Column name for x-axis (feature).
        prediction_methods: List of method names (series).
        method_to_col_map: Dict mapping method name to prediction column.
        method_to_low_col_map: Dict mapping method name to low range column.
        method_to_high_col_map: Dict mapping method name to high range column.
        title: Plot title.
        xaxis_title: X-axis label.
        yaxis_title: Y-axis label.
        jitter_strength: Amount of jitter for categorical x-axis (default 0.15).
    Returns:
        Plotly Figure object.
    """
    import numpy as np
    import plotly.graph_objs as go
    import plotly.colors

    palette = plotly.colors.qualitative.Plotly
    n_colors = len(palette)
    fig = go.Figure()

    is_categorical = not np.issubdtype(df[x_col].dtype, np.number)
    if is_categorical:
        # Map categories to numbers for plotting
        categories = list(df[x_col].unique())
        cat_to_num = {cat: i for i, cat in enumerate(categories)}
        x_vals = df[x_col].map(cat_to_num)
        x_tickvals = list(range(len(categories)))
        x_ticktext = categories
    else:
        x_vals = df[x_col]
        x_tickvals = None
        x_ticktext = None

    for i, method in enumerate(prediction_methods):
        color = palette[i % n_colors]
        pred_col = method_to_col_map[method]
        low_col = method_to_low_col_map[method]
        high_col = method_to_high_col_map[method]

        # Jitter for categorical x-axis to separate series
        if is_categorical:
            jitter = (i - (len(prediction_methods)-1)/2) * jitter_strength
            x_plot = x_vals + jitter
        else:
            x_plot = x_vals

        fig.add_trace(
            go.Scatter(
                x=x_plot,
                y=df[pred_col],
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=df[high_col] - df[pred_col],
                    arrayminus=df[pred_col] - df[low_col],
                    thickness=1.5,
                    width=6,
                    color=color,
                ),
                mode='markers',
                marker=dict(color=color, size=10, line=dict(width=1, color='black')),
                name=method,
                legendgroup=method,
                text=df[x_col] if is_categorical else None,
                hovertemplate=(
                    f"{x_col}: %{{x}}<br>Prediction: %{{y}}<br>Low: %{{customdata[0]}}<br>High: %{{customdata[1]}}<extra>{method}</extra>"
                ),
                customdata=np.stack([df[low_col], df[high_col]], axis=-1),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title or x_col,
        yaxis_title=yaxis_title,
        legend_title="Prediction method:",
        xaxis=dict(
            tickmode='array' if is_categorical else 'auto',
            tickvals=x_tickvals if is_categorical else None,
            ticktext=x_ticktext if is_categorical else None,
        ),
        template="plotly_white",
    )
    return fig
