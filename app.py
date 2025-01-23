import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import qualitative

# ─────────────────────────────────────────────────────────
# 1) LOAD & PREP THE DATA (ONCE)
# ─────────────────────────────────────────────────────────
data = pd.read_csv("data/injurydat.csv", encoding='utf-8', low_memory=False)

# Clean lat/lon
data["Latitude"]  = pd.to_numeric(data["Latitude"], errors="coerce")
data["Longitude"] = pd.to_numeric(data["Longitude"], errors="coerce")

# Convert relevant columns to lowercase
cols_to_lower = [
    "Victim.injury", "Shark.common.name", "Provoked.unprovoked",
    "Victim.activity", "Injury.severity", "Victim.gender"
]
for col in cols_to_lower:
    if col in data.columns:
        data[col] = data[col].astype(str).str.lower()

# Because some of our analyses need 'Fatal' flags, define them here:
data['Fatal'] = data['Victim.injury'].apply(lambda x: 1 if str(x).lower() == 'fatal' else 0)
data['Non_Fatal'] = data['Fatal'].apply(lambda x: 1 if x == 0 else 0)

# Clean the 'Incident.year' column
data['Incident.year'] = pd.to_numeric(data['Incident.year'], errors='coerce')
data = data.dropna(subset=['Incident.year'])
data['Year'] = data['Incident.year'].astype(int)

# ─────────────────────────────────────────────────────────
# 1A) DATA FOR THE MAP
# ─────────────────────────────────────────────────────────
# Drop rows with no lat/lon
map_data = data.dropna(subset=["Latitude", "Longitude"]).copy()

# ─────────────────────────────────────────────────────────
# 1B) DATA FOR THE BAR CHART
# ─────────────────────────────────────────────────────────
ranked_data = data.groupby(['State', 'Year']).agg(
    Total_Incidents=('Incident.day', 'count'),
    Fatal_Incidents=('Fatal', 'sum'),
    Non_Fatal_Incidents=('Non_Fatal', 'sum')
).reset_index()

# ─────────────────────────────────────────────────────────
# 1C) DATA FOR THE HEATMAP
# ─────────────────────────────────────────────────────────
heatmap_data = data.groupby(['Victim.activity', 'State']).agg(
    Total_Incidents=('Incident.day', 'count'),
    Fatal_Incidents=('Fatal', 'sum')
).reset_index()

heatmap_data['Non_Fatal_Incidents'] = (
    heatmap_data['Total_Incidents'] - heatmap_data['Fatal_Incidents']
)

# ─────────────────────────────────────────────────────────
# 2) BUILD THE MAP FIGURE (MASTER TRACES + DROPDOWN)
# ─────────────────────────────────────────────────────────
dropdown_options = [
    {"label": "Shark Name",         "value": "Shark.common.name"},
    {"label": "Provoked Incident",  "value": "Provoked.unprovoked"},
    {"label": "Victim Activity",    "value": "Victim.activity"},
    {"label": "Injury Severity",    "value": "Injury.severity"},
    {"label": "Victim Gender",      "value": "Victim.gender"},
]
all_columns = [opt["value"] for opt in dropdown_options]

# Columns to show in the hover text
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

    # Specifically remove 'male' from "Injury.severity", if needed
    if col == "Injury.severity":
        unique_vals = [x for x in unique_vals if x != "male"]

    for i, val in enumerate(unique_vals):
        subset = map_data[map_data[col] == val].copy()

        # Build a hover text that only includes your chosen columns
        subset["hover_text"] = subset.apply(
            lambda row: "<br>".join([
                f"{c}: {row[c]}" for c in cols_to_show if c in subset.columns
            ]),
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

# Default: show the first dropdown column
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
        height=700,
        title="Filtered Incident Map",
        legend=dict(title=dict(text=dropdown_options[0]["label"]))
    )
)

# Create the dropdown buttons
buttons = []
for opt in dropdown_options:
    col_name = opt["value"]
    label = opt["label"]

    new_visibility = [False] * len(master_traces)
    new_showlegend = [False] * len(master_traces)
    for idx in col_to_trace_indices[col_name]:
        new_visibility[idx] = True
        new_showlegend[idx] = True

    buttons.append(
        dict(
            label=label,
            method="update",
            args=[
                {"visible": new_visibility, "showlegend": new_showlegend},
                {"legend": {"title": {"text": label}}}
            ]
        )
    )

map_fig.update_layout(
    updatemenus=[
        dict(
            type="dropdown",
            buttons=buttons,
            direction="down",
            x=0.02,
            y=1.02,
            showactive=True
        )
    ]
)

# ─────────────────────────────────────────────────────────
# 3) BUILD THE DASH APP
# ─────────────────────────────────────────────────────────
app = dash.Dash(__name__)

app.layout = html.Div([
    # ✨ MAP with Dropdown Filter
    html.H1("Shark Incidents Map", style={'textAlign': 'center'}),
    dcc.Graph(id="map-graph", figure=map_fig),

    # ✨ BAR CHART
    html.H1("Ranked Bar Chart of Shark Incidents by State", style={'textAlign': 'center'}),
    dcc.Graph(id="bar-chart"),
    html.Label("Select Year Range:"),
    html.Div([
        dcc.Input(
            id="start-year",
            type="number",
            placeholder="Start Year",
            min=ranked_data['Year'].min(),
            max=ranked_data['Year'].max(),
            step=1,
            value=ranked_data['Year'].min(),
        ),
        dcc.Input(
            id="end-year",
            type="number",
            placeholder="End Year",
            min=ranked_data['Year'].min(),
            max=ranked_data['Year'].max(),
            step=1,
            value=ranked_data['Year'].max(),
        ),
    ], style={'display': 'flex', 'gap': '10px'}),
    html.Label("Select Chart Mode:"),
    dcc.RadioItems(
        id="chart-mode",
        options=[
            {"label": "Grouped Bar (Fatal vs. Non-Fatal)", "value": "grouped"},
            {"label": "Fatal Incidents Only", "value": "fatal"},
            {"label": "Non-Fatal Incidents Only", "value": "non_fatal"},
            {"label": "Stacked Bar (Percentages)", "value": "percentage_stacked"}
        ],
        value="grouped",
        inline=True
    ),
    html.Label("Toggle Logarithmic Scale:"),
    dcc.Checklist(
        id="log-scale",
        options=[{"label": "Log Scale", "value": "log"}],
        value=[]
    ),

    html.Hr(),

    # ✨ HEATMAP + STATE SELECTION
    html.H1("Activity-Location Risk Profiles", style={'textAlign': 'center'}),

    html.Label("Select Incident Metric:"),
    dcc.RadioItems(
        id="metric-selector",
        options=[
            {"label": "Total Incidents", "value": "Total_Incidents"},
            {"label": "Fatal Incidents", "value": "Fatal_Incidents"},
            {"label": "Non-Fatal Incidents", "value": "Non_Fatal_Incidents"}
        ],
        value="Total_Incidents",
        inline=True
    ),
    dcc.Graph(id="activity-location-heatmap"),

    html.Hr(),

    html.Label("Select a State to View Spatial Distribution:"),
    dcc.Dropdown(
        id="state-selector",
        options=[
            {"label": state, "value": state} 
            for state in sorted(heatmap_data['State'].dropna().unique())
        ],
        placeholder="Select a state",
        searchable=True
    ),
    dcc.Graph(id="state-incident-map")
])

# ─────────────────────────────────────────────────────────
# 4) BAR CHART CALLBACK
# ─────────────────────────────────────────────────────────
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
                "Fatal Incidents": "lightcoral",
                "Non-Fatal Incidents": "lightblue"
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
            color_discrete_sequence=["lightcoral"]
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
            color_discrete_sequence=["lightblue"]
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
                "Fatal Incidents": "lightcoral",
                "Non-Fatal Incidents": "lightblue"
            }
        )
        fig.update_traces(
            hovertemplate="<b>Percentage:</b> %{y:.2f}%<br>",
            customdata=melted[['Incident_Type', 'Total_Incidents']].to_numpy(),
            showlegend=False
        )
        fig.update_yaxes(range=[0, 100])

    # Optional: Log scale
    if "log" in log_scale:
        fig.update_yaxes(type="log")

    return fig

# ─────────────────────────────────────────────────────────
# 5) HEATMAP CALLBACK
# ─────────────────────────────────────────────────────────
@app.callback(
    Output("activity-location-heatmap", "figure"),
    [Input("metric-selector", "value")]
)
def update_heatmap(metric):
    fig = px.density_heatmap(
        heatmap_data,
        x="Victim.activity",
        y="State",
        z=metric,
        color_continuous_scale="RdYlGn",
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
        margin=dict(l=50, r=50, t=50, b=100)
    )
    return fig

# ─────────────────────────────────────────────────────────
# 6) STATE-SPECIFIC MAP CALLBACK
# ─────────────────────────────────────────────────────────
@app.callback(
    Output("state-incident-map", "figure"),
    [Input("state-selector", "value")]
)
def update_state_map(selected_state):
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
    fig.update_layout(
        mapbox_style="open-street-map",
        height=600,
        title_x=0.5,
        margin=dict(l=50, r=50, t=50, b=100)
    )
    return fig

# ─────────────────────────────────────────────────────────
# 7) RUN THE APP
# ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
