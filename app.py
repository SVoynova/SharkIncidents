import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import qualitative


# ─────────────────────────────────────────────────────────
# 0) CREATE GLOBALVARIABLES 
# ─────────────────────────────────────────────────────────

# Define a consistent color palette for all charts
color_palette = {
    "Fatal Incidents": "#E74C3C",  # Light red for fatal incidents (dangerous/severe)
    "Non-Fatal Incidents": "#3498DB",  # Light blue for non-fatal incidents (calmer)
    "Provoked": "#F5B041",  # Orange for provoked incidents
    "Unprovoked": "#58D68D",  # Green for unprovoked incidents
    "Unknown": "#7F8C8D"  # Neutral gray for unknown
}


# ─────────────────────────────────────────────────────────
# 1) LOAD DATA (TWO DIFFERENT DATAFRAMES)
# ─────────────────────────────────────────────────────────

# A) DATAFRAME "data" for Map/Bar/Heatmap
data = pd.read_csv("data/injurydat.csv", encoding='utf-8', low_memory=False)

# Clean lat/lon
data["Latitude"]  = pd.to_numeric(data["Latitude"], errors="coerce")
data["Longitude"] = pd.to_numeric(data["Longitude"], errors="coerce")

# Convert to lowercase where relevant

cols_to_lower = [
    "Victim.injury", "Shark.common.name", "Provoked.unprovoked",
    "Victim.activity", "Injury.severity", "Victim.gender", "State"
]
for col in cols_to_lower:
    if col in data.columns:
        data[col] = data[col].astype(str).str.lower()

# Filter out rows where Victim.gender is not "male" or "female"
data = data[data['Victim.gender'].isin(['male', 'female'])]

# Create 'Fatal' flags if we have Victim.injury
if 'Victim.injury' in data.columns:
    data['Fatal'] = data['Victim.injury'].apply(lambda x: 1 if x == 'fatal' else 0)
    data['Non_Fatal'] = data['Fatal'].apply(lambda x: 1 if x == 0 else 0)

# Clean up 'Incident.year'
if 'Incident.year' in data.columns:
    data['Incident.year'] = pd.to_numeric(data['Incident.year'], errors='coerce')
    data.dropna(subset=['Incident.year'], inplace=True)
    data['Year'] = data['Incident.year'].astype(int)


# B) DATAFRAME "data2" for SPLOM
data2 = pd.read_excel(
    "data/Data2.xlsx", 
    sheet_name=0
)

# Fill missing for key columns
data2['Provoked/unprovoked'] = data2['Provoked/unprovoked'].fillna('Unknown')
data2['State'] = data2['State'].fillna('Unknown')
data2['Shark.common.name'] = data2['Shark.common.name'].fillna('Unknown')
data2['Victim.age'] = pd.to_numeric(data2['Victim.age'], errors='coerce')


# Convert numeric columns & fill NAs (or drop them, up to you)
numeric_cols_sp = [
    'Incident.year', 'Shark.length.m', 'Victim.age',
    'Depth.of.incident.m', 'Water.temperature.°C', 'Distance.to.shore.m'
]
for col in numeric_cols_sp:
    if col in data2.columns:
        data2[col] = pd.to_numeric(data2[col], errors='coerce')
        data2[col] = data2[col].fillna(0)

# Create special SPLOM attributes
selected_attributes = [
    'Incident.year', 'Shark.length.m', 'Victim.age',
    'Depth.of.incident.m', 'Water.temperature.°C', 'Distance.to.shore.m'
]
attribute_labels = {
    'Incident.year': 'Year',
    'Shark.length.m': 'Shark Length (m)',
    'Victim.age': 'Victim Age',
    'Depth.of.incident.m': 'Depth (m)',
    'Water.temperature.°C': 'Temp (°C)',
    'Distance.to.shore.m': 'Dist to Shore (m)'
}

# Build state/shark dropdown options for the SPLOM
sp_total_cases = len(data2)
sp_state_counts = data2['State'].value_counts()
sp_shark_counts = data2['Shark.common.name'].value_counts()

sp_state_options = (
    [{'label': f"Total ({sp_total_cases} cases)", 'value': 'Total'}] +
    [{'label': f"{state} ({count} cases)", 'value': state} for state, count in sp_state_counts.items()]
)
sp_shark_options = (
    [{'label': f"Total ({sp_total_cases} cases)", 'value': 'Total'}] +
    [{'label': f"{shark} ({count} cases)", 'value': shark} for shark, count in sp_shark_counts.items()]
)

data2['Provoked/unprovoked'] = data2['Provoked/unprovoked'].fillna('Unknown')
data2['Date'] = pd.to_datetime(data2['Incident.year'].astype(str) + '-01', errors='coerce')



# Normalize 'State' column in the main dataset
data['State'] = data['State'].str.strip().str.lower()

# Normalize 'State' column in the SPLOM dataset
data2['State'] = data2['State'].str.strip().str.lower()

# Update the dropdown options to match the normalized state names
state_options = [
    {"label": state.capitalize(), "value": state}
    for state in sorted(data2['State'].dropna().unique())
]


# ─────────────────────────────────────────────────────────
# 2) PREPARE THE DATA SUBSETS FOR EXISTING CHARTS
# ─────────────────────────────────────────────────────────

# A) Map with Category Dropdown
map_data = data.dropna(subset=["Latitude", "Longitude"]).copy()

dropdown_options = [
    {"label": "Shark Name",         "value": "Shark.common.name"},
    {"label": "Provoked Incident",  "value": "Provoked.unprovoked"},
    {"label": "Victim Activity",    "value": "Victim.activity"},
    {"label": "Injury Severity",    "value": "Injury.severity"},
    {"label": "Victim Gender",      "value": "Victim.gender"},
]
all_columns = [opt["value"] for opt in dropdown_options]

# Columns to show in map hover text
cols_to_show = [
    "Victim.injury",
    "Shark.common.name",
    "Provoked.unprovoked",
    "Victim.activity",
    "Injury.severity",
    "Victim.gender"
]

master_traces = []
col_to_trace_indices = {col: [] for col in all_columns}
color_palette = qualitative.Plotly
trace_index = 0

for col in all_columns:
    if col not in map_data.columns:
        continue
    unique_vals = map_data[col].dropna().unique()
    unique_vals = [x for x in unique_vals if x != "nan"]

    # If removing 'male' from "Injury.severity"
    if col == "Injury.severity":
        unique_vals = [x for x in unique_vals if x != "male"]

    for i, val in enumerate(unique_vals):
        subset = map_data[map_data[col] == val].copy()
        subset["hover_text"] = subset.apply(
            lambda row: "<br>".join(
                f"{c}: {row[c]}" for c in cols_to_show if c in subset.columns
            ),
            axis=1
        )
        new_trace = go.Scattermapbox(
            lat=subset["Latitude"],
            lon=subset["Longitude"],
            mode="markers",
            marker=dict(size=8, color=color_palette[i % len(color_palette)]),
            name=str(val),
            visible=False,
            showlegend=False,
            text=subset["hover_text"],
            hoverinfo="text"
        )
        master_traces.append(new_trace)
        col_to_trace_indices[col].append(trace_index)
        trace_index += 1

# Default: show the first category
default_col = dropdown_options[0]["value"]
initial_visibility = [False] * len(master_traces)
initial_showlegend = [False] * len(master_traces)
for idx in col_to_trace_indices[default_col]:
    initial_visibility[idx] = True
    initial_showlegend[idx] = True

for i, tr in enumerate(master_traces):
    tr.visible = initial_visibility[i]
    tr.showlegend = initial_showlegend[i]

map_fig = go.Figure(
    data=master_traces,
    layout=go.Layout(
        mapbox=dict(
            style="open-street-map",
            zoom=3,
            center=dict(lat=-25, lon=135)
        ),
        height=500,
        title="Filtered Incident Map",
        legend=dict(title=dict(text=dropdown_options[0]["label"]))
    )
)

# # Create the map dropdown
# buttons = []
# for opt in dropdown_options:
#     col_name = opt["value"]
#     label = opt["label"]
#     new_visibility = [False] * len(master_traces)
#     new_showlegend = [False] * len(master_traces)
#     for idx in col_to_trace_indices[col_name]:
#         new_visibility[idx] = True
#         new_showlegend[idx] = True
#     buttons.append(
#         dict(
#             label=label,
#             method="update",
#             args=[
#                 {"visible": new_visibility, "showlegend": new_showlegend},
#                 {"legend": {"title": {"text": label}}}
#             ]
#         )
#     )

# map_fig.update_layout(
#     updatemenus=[
#         dict(
#             type="dropdown",
#             buttons=buttons,
#             direction="down",
#             x=0.02,
#             y=1.02,
#             showactive=True
#         )
#     ]
# )

# B) Ranked Bar Chart Data
# (Needs columns: State, Year, Incident.day)
if {'State', 'Year', 'Incident.day'}.issubset(data.columns):
    ranked_data = data.groupby(['State', 'Year'], dropna=False).agg(
        Total_Incidents=('Incident.day', 'count'),
        Fatal_Incidents=('Fatal', 'sum'),
        Non_Fatal_Incidents=('Non_Fatal', 'sum')
    ).reset_index()
else:
    ranked_data = pd.DataFrame(columns=['State','Year','Total_Incidents','Fatal_Incidents','Non_Fatal_Incidents'])

# C) Heatmap & State-Specific Map
if {'Victim.activity', 'State', 'Incident.day'}.issubset(data.columns):
    heatmap_data = data.groupby(['Victim.activity', 'State'], dropna=False).agg(
        Total_Incidents=('Incident.day', 'count'),
        Fatal_Incidents=('Fatal', 'sum')
    ).reset_index()
    heatmap_data['Non_Fatal_Incidents'] = (
        heatmap_data['Total_Incidents'] - heatmap_data['Fatal_Incidents']
    )
else:
    heatmap_data = pd.DataFrame(columns=['Victim.activity','State','Total_Incidents','Fatal_Incidents','Non_Fatal_Incidents'])


# ─────────────────────────────────────────────────────────
# 3) BUILD THE DASH APP LAYOUT
# ─────────────────────────────────────────────────────────
app = dash.Dash(__name__)

app.layout = html.Div([
    # Title
    html.H1("Shark Incident Analysis Dashboard", style={'textAlign': 'center'}),

    # Top Section with 2x2 Grid (Maps and Charts)
    html.Div([
        # Left Column
        html.Div([
            # Filtered Incident Map Section
            html.Div([
                html.H3("Filtered Incident Map", style={'textAlign': 'center'}),
                dcc.Dropdown(
                    id="map-filter-dropdown",
                    options=dropdown_options,
                    value=dropdown_options[0]["value"],  # Default selection
                    placeholder="Select Filter",
                    clearable=False,
                    style={'marginBottom': '125px', 'width': '90%', 'margin': '0 auto'}
                ),
                dcc.Graph(id="map-graph", style={'height': '4   0vh', 'width': '100%'})
            ], style={'padding': '30px', 'boxSizing': 'border-box'}),

            # State Incident Map Section
            html.Div([
                html.H3("State Incident Distribution", style={'textAlign': 'center'}),
                dcc.Dropdown(
                    id="state-selector",
                    options=[
                        {"label": state, "value": state}
                        for state in sorted(heatmap_data['State'].dropna().unique())
                    ],
                    placeholder="Select a state",
                    clearable=False,
                    searchable=True,
                    style={'marginBottom': '25px', 'width': '90%', 'margin': '0 auto'}
                ),
                dcc.Graph(id="state-incident-map", style={'height': '50vh', 'width': '100%'})
            ], style={'padding': '30px', 'boxSizing': 'border-box'})
        ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        # Right Column
        html.Div([
            # Ranked Bar Chart Section
            html.Div([
                html.H3("Ranked Bar Chart of Shark Incidents", style={'textAlign': 'center'}),
                html.Label("Select Year Range:", style={'marginTop': '10px'}),
                html.Div([
                    dcc.Input(
                        id="start-year",
                        type="number",
                        placeholder="Start Year",
                        min=ranked_data['Year'].min() if not ranked_data.empty else 1900,
                        max=ranked_data['Year'].max() if not ranked_data.empty else 2025,
                        step=1,
                        value=ranked_data['Year'].min() if not ranked_data.empty else 1900,
                    ),
                    dcc.Input(
                        id="end-year",
                        type="number",
                        placeholder="End Year",
                        min=ranked_data['Year'].min() if not ranked_data.empty else 1900,
                        max=ranked_data['Year'].max() if not ranked_data.empty else 2025,
                        step=1,
                        value=ranked_data['Year'].max() if not ranked_data.empty else 2025,
                    ),
                ], style={'display': 'flex', 'gap': '10px', 'justifyContent': 'center'}),
                html.Label("Select Chart Mode:", style={'marginTop': '15px'}),
                dcc.RadioItems(
                    id="chart-mode",
                    options=[
                        {"label": "Grouped Bar (Fatal vs. Non-Fatal)", "value": "grouped"},
                        {"label": "Fatal Incidents Only", "value": "fatal"},
                        {"label": "Non-Fatal Incidents Only", "value": "non_fatal"},
                        {"label": "Stacked Bar (Percentages)", "value": "percentage_stacked"}
                    ],
                    value="grouped",
                    inline=True,
                    style={'textAlign': 'center'}
                ),
                html.Label("Toggle Logarithmic Scale:", style={'marginTop': '10px'}),
                dcc.Checklist(
                    id="log-scale",
                    options=[{"label": "Log Scale", "value": "log"}],
                    value=[],
                    style={'textAlign': 'center'}
                ),
                dcc.Graph(id="bar-chart", style={'height': '40vh', 'width': '100%'})
            ], style={'padding': '20px', 'boxSizing': 'border-box'}),

            # Activity Location Risk Profiles
            html.Div([
                html.H3("Activity-Location Risk Profiles", style={'textAlign': 'center'}),
                html.Label("Select Incident Metric:", style={'marginTop': '15px'}),
                dcc.RadioItems(
                    id="metric-selector",
                    options=[
                        {"label": "Total Incidents", "value": "Total_Incidents"},
                        {"label": "Fatal Incidents", "value": "Fatal_Incidents"},
                        {"label": "Non-Fatal Incidents", "value": "Non_Fatal_Incidents"}
                    ],
                    value="Total_Incidents",
                    inline=True,
                    style={'textAlign': 'center'}
                ),
                dcc.Graph(id="activity-location-heatmap", style={'height': '40vh', 'width': '100%'})
            ], style={'padding': '20px', 'boxSizing': 'border-box'})
        ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ], style={
        'display': 'flex',
        'flexDirection': 'row',
        'alignItems': 'stretch',
        'justifyContent': 'space-between',
        'width': '100%',
        'boxSizing': 'border-box'
    }),

    # SPLOM Section
    html.Div([
        html.H3("Filtered Scatterplot Matrix (SPLOM)", style={'textAlign': 'center'}),
        dcc.Graph(id='sp_matrix', style={'height': '80vh', 'width': '100%'}),
        html.Div([
            html.Div([
                html.Label("Filter by State:"),
                dcc.Dropdown(
                    id='state-filter',
                    options=sp_state_options,
                    value='Total',
                    clearable=True,
                    placeholder="Select a State"
                )
            ], style={'width': '45%', 'display': 'inline-block'}),

            html.Div([
                html.Label("Filter by Shark Type:"),
                dcc.Dropdown(
                    id='shark-type-filter',
                    options=state_options,
                    value='Total',
                    clearable=True,
                    placeholder="Select a Shark Type"
                )
            ], style={'width': '45%', 'display': 'inline-block'}),

            html.Div([
                html.Label("Filter by Victim Age:"),
                dcc.RangeSlider(
                    id='age-slider',
                    min=int(data2['Victim.age'].min()) if 'Victim.age' in data2.columns else 0,
                    max=int(data2['Victim.age'].max()) if 'Victim.age' in data2.columns else 100,
                    step=1,
                    marks={i: str(i) for i in range(0, 101, 10)},
                    value=[
                        int(data2['Victim.age'].min()) if 'Victim.age' in data2.columns else 0,
                        int(data2['Victim.age'].max()) if 'Victim.age' in data2.columns else 100
                    ]
                )
            ], style={'marginTop': '20px'})
        ], style={'padding': '20px', 'borderTop': '1px solid #ccc', 'marginTop': '10px'})
    ], style={'marginTop': '30px'}),
    
    # (E) zoltan chart
    # First row: Line Chart and Pie Chart
    html.Div([
        # Line Chart
        html.Div([dcc.Graph(id='line-chart')], style={"width": "80%", "height": "95%", "display": "inline-block"}),

        # Pie Chart
        html.Div([
            html.H4("Attack Type Distribution", style={"textAlign": "center"}),
            dcc.Graph(id='pie-chart')
        ], style={"width": "18%", "height": "95%", "display": "inline-block", "verticalAlign": "top"})
    ]),

    # Second row: Date slider
    html.Div([
        dcc.RangeSlider(
            id='date-slider',
            min=data2['Date'].dt.year.min(),
            max=data2['Date'].dt.year.max(),
            step=1,
            marks={year: str(year) for year in range(data2['Date'].dt.year.min(), data2['Date'].dt.year.max() + 1, 10)},
            value=[data2['Date'].dt.year.min(), data2['Date'].dt.year.max()]
        )
    ], style={"height": "5%", "padding": "20px"})
])


# ─────────────────────────────────────────────────────────
# 4) DEFINE CALLBACKS
# ─────────────────────────────────────────────────────────

# Define the callback for the filtered incident map
@app.callback(
    Output("map-graph", "figure"),  # Output: the map figure
    Input("map-filter-dropdown", "value")  # Input: dropdown value
)
def update_filtered_map(selected_column):
    # Ensure the selected column exists
    if selected_column not in map_data.columns:
        # Return an empty map if the column is invalid
        return px.scatter_mapbox(
            title="Invalid Column Selection"
        ).update_layout(height=500)

    # Adjust visibility of traces based on the selected column
    updated_visibility = [False] * len(master_traces)
    updated_showlegend = [False] * len(master_traces)
    for idx in col_to_trace_indices[selected_column]:
        updated_visibility[idx] = True
        updated_showlegend[idx] = True

    # Update figure layout and visibility
    updated_map_fig = go.Figure(
        data=master_traces,  # Use existing traces
        layout=go.Layout(
            mapbox=dict(
                style="open-street-map",
                zoom=3,
                center=dict(lat=-25, lon=135)
            ),
            height=700,
            title='',
            legend=dict(title=dict(text=selected_column.replace('.', ' ').capitalize()))
        )
    )

    # Apply visibility and legend updates to each trace
    for i, trace in enumerate(updated_map_fig.data):
        trace.visible = updated_visibility[i]
        trace.showlegend = updated_showlegend[i]

    # Define the new color scheme
    color_map = {
        "provoked": "#F5B041",  # Orange for provoked incidents
        "unprovoked": "#58D68D",  # Green for unprovoked incidents
        "unknown": "#7F8C8D"  # Neutral gray for unknown
    }

    # Apply colors based on the Provoked/Unprovoked classification
    for trace in updated_map_fig.data:
        if trace.name in color_map:
            trace.marker.color = color_map[trace.name]

    return updated_map_fig


# (A) Bar Chart
@app.callback(
    Output("bar-chart", "figure"),
    [
        Input("chart-mode", "value"),
        Input("log-scale", "value"),
        Input("start-year", "value"),
        Input("end-year", "value"),
    ]
)
def update_chart(chart_mode, log_scale, start_year, end_year):
    if len(ranked_data) == 0:
        return px.bar(title="No data for bar chart")

    if start_year is None:
        start_year = ranked_data['Year'].min()
    if end_year is None:
        end_year = ranked_data['Year'].max()
    if start_year > end_year:
        return px.bar(title="Error: Start Year cannot be greater than End Year")

    filtered_data = ranked_data[
        (ranked_data['Year'] >= start_year) & (ranked_data['Year'] <= end_year)
    ]

    if chart_mode == "grouped":
        grouped_data = filtered_data.groupby('State', as_index=False).agg(
            Fatal_Incidents=('Fatal_Incidents', 'sum'),
            Non_Fatal_Incidents=('Non_Fatal_Incidents', 'sum'),
            Total_Incidents=('Total_Incidents', 'sum')
        ).sort_values(by='Total_Incidents', ascending=False)

        melted = pd.melt(
            grouped_data,
            id_vars=['State', 'Total_Incidents'],
            value_vars=['Fatal_Incidents', 'Non_Fatal_Incidents'],
            var_name='Incident_Type',
            value_name='Incident_Count'
        )
        type_mapping = {
            "Fatal_Incidents": "Fatal Incidents",
            "Non_Fatal_Incidents": "Non-Fatal Incidents"
        }
        melted['Incident_Type'] = melted['Incident_Type'].map(type_mapping)

        fig = px.bar(
            melted,
            x="State",
            y="Incident_Count",
            color="Incident_Type",
            barmode="group",
            title=f"Grouped Bar Chart: Fatal vs. Non-Fatal Incidents ({start_year}-{end_year})",
            labels={
                "State": "State",
                "Incident_Count": "Number of Incidents",
                "Incident_Type": "Type of Incident"
            },
            color_discrete_map={
                "Fatal Incidents": "#E74C3C",
                "Non-Fatal Incidents": "#3498DB"
            }
        )
        fig.update_traces(
            hovertemplate=(
                "<b>State:</b> %{x}<br>"
                "<b>Incident Type:</b> %{customdata[0]}<br>"
                "<b>Number of Incidents:</b> %{y}<br>"
                "<b>Total Incidents in State:</b> %{customdata[1]}<extra></extra>"
            ),
            customdata=melted[['Incident_Type', 'Total_Incidents']].values
        )
        fig.update_yaxes(range=[0, 450])

    elif chart_mode == "fatal":
        fatal_data = filtered_data.groupby('State').agg(
            Fatal_Incidents=('Fatal_Incidents', 'sum')
        ).reset_index().sort_values(by='Fatal_Incidents', ascending=False)
        fig = px.bar(
            fatal_data,
            x="State",
            y="Fatal_Incidents",
            title=f"Bar Chart: Fatal Incidents Only ({start_year}-{end_year})",
            labels={"State": "State", "Fatal_Incidents": "Number of Fatal Incidents"},
            color_discrete_sequence=["#E74C3C"]
        )
        fig.update_yaxes(range=[0, 450])

    elif chart_mode == "non_fatal":
        non_fatal_data = filtered_data.groupby('State').agg(
            Non_Fatal_Incidents=('Non_Fatal_Incidents', 'sum')
        ).reset_index().sort_values(by='Non_Fatal_Incidents', ascending=False)
        fig = px.bar(
            non_fatal_data,
            x="State",
            y="Non_Fatal_Incidents",
            title=f"Bar Chart: Non-Fatal Incidents Only ({start_year}-{end_year})",
            labels={"State": "State", "Non_Fatal_Incidents": "Number of Non-Fatal Incidents"},
            color_discrete_sequence=["#3498DB"]
        )
        fig.update_yaxes(range=[0, 450])

    elif chart_mode == "percentage_stacked":
        percentage_data = filtered_data.groupby('State', as_index=False).agg(
            Fatal_Incidents=('Fatal_Incidents', 'sum'),
            Non_Fatal_Incidents=('Non_Fatal_Incidents', 'sum'),
            Total_Incidents=('Total_Incidents', 'sum')
        )
        percentage_data['Fatal_Percentage'] = (
            percentage_data['Fatal_Incidents'] / percentage_data['Total_Incidents'] * 100
        )
        percentage_data['Non_Fatal_Percentage'] = (
            percentage_data['Non_Fatal_Incidents'] / percentage_data['Total_Incidents'] * 100
        )

        melted = pd.melt(
            percentage_data,
            id_vars=['State', 'Total_Incidents'],
            value_vars=['Fatal_Percentage', 'Non_Fatal_Percentage'],
            var_name='Incident_Type',
            value_name='Percentage'
        )
        type_mapping = {
            "Fatal_Percentage": "Fatal Incidents",
            "Non_Fatal_Percentage": "Non-Fatal Incidents"
        }
        melted['Incident_Type'] = melted['Incident_Type'].map(type_mapping)

        fig = px.bar(
            melted,
            x="State",
            y="Percentage",
            color="Incident_Type",
            barmode="stack",
            title=f"Stacked Bar Chart: Percentage of Fatal vs. Non-Fatal Incidents ({start_year}-{end_year})",
            labels={"Percentage": "Percentage (%)"},
            color_discrete_map={
                "Fatal Incidents": "#E74C3C",
                "Non-Fatal Incidents": "#3498DB"
            }
        )
        fig.update_traces(
            hovertemplate="<b>Percentage:</b> %{y:.2f}%<br>",
            customdata=melted[['Incident_Type', 'Total_Incidents']].to_numpy(),
            showlegend=False
        )
        fig.update_yaxes(range=[0, 100])

    # Optional log scale
    if "log" in log_scale:
        fig.update_yaxes(type="log")

    return fig


# (B) Heatmap Callback
@app.callback(
    Output("activity-location-heatmap", "figure"),
    [Input("metric-selector", "value")]
)
def update_heatmap(metric):
    if heatmap_data.empty:
        return px.bar(title="No data for heatmap")

    fig = px.density_heatmap(
        heatmap_data,
        x="Victim.activity",
        y="State",
        z=metric,
        color_continuous_scale="Viridis",  # Apply Viridis color scale
        title="Activity-Location Heatmap",
        labels={
            "Victim.activity": "Activity",
            "State": "State",
            metric: "Incident Count"
        }
    )
    fig.update_layout(
        xaxis_tickangle=45,
        height=600,
        title_x=0.5,
        margin=dict(l=50, r=50, t=50, b=100),
        coloraxis_colorbar=dict(
            title="Incident Count",  # Add colorbar title
            ticks="outside"  # Place ticks outside
        )
    )
    return fig


# (C) Per-State Incident Map Callback
@app.callback(
    Output("state-incident-map", "figure"),
    [Input("state-selector", "value"),
     Input("metric-selector", "value")]
)
def update_state_incident_map(selected_state, selected_metric):
    if not selected_state:
        return px.scatter_mapbox(
            title="Select a State to View Incidents",
            height=600
        )

    state_data = data[data['State'] == selected_state]
    if state_data.empty:
        return px.scatter_mapbox(
            title=f"No Data Available for {selected_state}",
            height=600
        )

    # Differentiate logic based on which input triggered the callback
    ctx = dash.callback_context
    if not ctx.triggered:
        triggered_input = None
    else:
        triggered_input = ctx.triggered[0]["prop_id"].split(".")[0]

    # Default map logic for state-selector
    if triggered_input == "state-selector" or triggered_input is None:
        fig = px.scatter_mapbox(
            state_data,
            lat="Latitude",
            lon="Longitude",
            color="Fatal",
            hover_data={"Victim.activity": True, "Victim.injury": True},
            title=f"Incident Distribution in {selected_state}",
            labels={"Fatal": "Fatal Incidents"},
            color_continuous_scale=["green", "red"]
        )
    else:  # Handle updates based on metric-selector
        fig = px.scatter_mapbox(
            state_data,
            lat="Latitude",
            lon="Longitude",
            size=selected_metric,
            hover_data={"Victim.activity": True, "Victim.injury": True},
            title=f"Incident Distribution in {selected_state}",
            labels={selected_metric: "Selected Metric"},
            color_continuous_scale="Blues"
        )

    fig.update_layout(
        mapbox_style="open-street-map",
        height=600,
        title_x=0.5,
        margin=dict(l=50, r=50, t=50, b=100)
    )
    return fig




# (D) SPLOM Callback (Uses data2)
@app.callback(
    Output('sp_matrix', 'figure'),
    [
        Input('state-filter', 'value'),
        Input('shark-type-filter', 'value'),
        Input('age-slider', 'value')
    ]
)
def update_sp_matrix(selected_state, selected_shark, selected_age_range):
    # Start from the second dataset
    df_sp = data2.copy()

    # Ensure Victim.age is numeric
    if 'Victim.age' in df_sp.columns:
        df_sp['Victim.age'] = pd.to_numeric(df_sp['Victim.age'], errors='coerce')

    # Filter by State
    if 'State' in df_sp.columns and selected_state != 'Total':
        df_sp = df_sp[df_sp['State'] == selected_state]

    # Filter by Shark Type
    if 'Shark.common.name' in df_sp.columns and selected_shark != 'Total':
        df_sp = df_sp[df_sp['Shark.common.name'] == selected_shark]

    # Filter by Victim Age
    if 'Victim.age' in df_sp.columns:
        df_sp = df_sp[
            (df_sp['Victim.age'] >= selected_age_range[0]) &
            (df_sp['Victim.age'] <= selected_age_range[1])
        ]

    # Handle missing 'Provoked/unprovoked' column
    if 'Provoked/unprovoked' not in df_sp.columns:
        df_sp['Provoked/unprovoked'] = 'Unknown'

    # Build SPLOM
    fig = px.scatter_matrix(
        df_sp,
        dimensions=[col for col in selected_attributes if col in df_sp.columns],
        color='Provoked/unprovoked',
        labels=attribute_labels,
        title="Filtered Scatterplot Matrix (SPLOM)",
        color_discrete_map={
            'Provoked': '#F5B041', 
            'Unprovoked': '#58D68D',
            'Unknown': '#7F8C8D'
        }
    )

    # Example axis formatting
    for axis in fig.layout:
        if axis.startswith('xaxis') or axis.startswith('yaxis'):
            fig.layout[axis].tickformat = ".0f"

    fig.update_layout(template="plotly_white", height=650, width=1200)
    return fig

# (E) chart from zoltan

@app.callback(
    [Output('line-chart', 'figure'),
     Output('pie-chart', 'figure')],
    [Input('date-slider', 'value')]
)
def update_charts(selected_range):
    # Filter the data based on the selected range
    start_date = pd.Timestamp(selected_range[0], 1, 1)
    end_date = pd.Timestamp(selected_range[1], 12, 31)
    filtered_data = data2[(data2['Date'] >= start_date) & (data2['Date'] <= end_date)]

    # Line Chart preprocessing
    yearly_data = filtered_data.groupby(['Date', 'Provoked/unprovoked']).size().reset_index(name='Count')
    line_fig = go.Figure()
    color_map = {
        'provoked':  "#F5B041",  # Orange for provoked incidents
        'unprovoked': "#58D68D",  # Green for unprovoked incidents
        'Unknown': "#7F8C8D"  # Neutral gray for unknown
    }
    for incident_type in yearly_data['Provoked/unprovoked'].unique():
        incident_data = yearly_data[yearly_data['Provoked/unprovoked'] == incident_type]
        line_fig.add_trace(
            go.Scatter(
                x=incident_data['Date'],
                y=incident_data['Count'],
                mode='lines+markers',
                name=incident_type,
                line=dict(color=color_map.get(incident_type, '#000000'))  # Default to black if missing

            )
        )
    line_fig.update_layout(
        title="Yearly Shark Incident Trends",
        xaxis_title="Date",
        yaxis_title="Number of Incidents",
        template="plotly_white"
    )

    # Pie Chart preprocessing
    pie_data = filtered_data['Provoked/unprovoked'].value_counts().reset_index()
    pie_data.columns = ['Provoked/unprovoked', 'Count']
    pie_fig = px.pie(
        pie_data,
        values='Count',
        names='Provoked/unprovoked',
        color='Provoked/unprovoked',  # Match color here
        color_discrete_map=color_map,  # Apply the custom color map
        hole=0.3  # Makes it a donut chart for a cleaner look
    )
    pie_fig.update_layout(margin=dict(t=15, b=15, l=0, r=0))

    return line_fig, pie_fig


# ─────────────────────────────────────────────────────────
# 5) RUN THE APP
# ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)