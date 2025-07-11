# Code Workspaces Prototype for DCEG HALO Metadata Viewer

The following is old.

* [Home on NIDAP](https://nidap.nih.gov/workspace/compass/view/ri.compass.main.folder.ed368b90-5cfe-4f21-9058-096353982b20)
* [Secondary home on GitHub](https://github.com/ncats/dceg-halo-metadata-viewer/tree/develop)

Features (updated 1/13/25 EOD):

- Data:
  - Choose x and y columns.
  - Choose a column and a subset of its values for filtering rows.
  - Display final number of rows and columns in filtered dataset.
  - Optionally reload the data from the NIDAP dataset.
- Scatter plot:
  - Select marker size.
  - Calculate best fit line (including r^2).
  - Select marker color for the first filtering group.
  - Tooltip information when hovering over datapoints.
  - Fully interactive plot: save image, zoom, pan, auto fit, maximize, select/hide filter groups.
  - Select points by point, box, or lasso (double-click to deselect points).
  - Display data table of selected points.
  - Highlight rows for selected points in full output of filtered data table.
  - Download any data table to CSV.
- Other:
  - Placeholder for violin etc. plots.
