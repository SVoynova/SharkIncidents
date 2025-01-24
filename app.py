import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import qualitative


# ─────────────────────────────────────────────────────────
# 0) CREATE GLOBAL VARIABLES 
# ─────────────────────────────────────────────────────────

# Define a consistent color palette for charts
color_palette = {
    "Fatal Incidents": "#E74C3C",
    "Non-Fatal Incidents": "#3498DB",
    "Provoked": "#F5B041",
    "Unprovoked": "#58D68D",
    "Unknown": "#7F8C8D"
}


# ─────────────────────────────────────────────────────────
# 1) LOAD DATA (TWO DIFFERENT DATAFRAMES)
# ─────────────────────────────────────────────────────────

# A) DATAFRAME "data" for Map/Bar/Heatmap
data = pd.read_csv("data/injurydat.csv", encoding='utf-8', low_memory=False)

# Clean lat/lon
data["Latitude"] = pd.to_numeric(data["Latitude"], errors="coerce")
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

# Create 'Fatal' flags
if 'Victim.injury' in data.columns:
    data['Fatal'] = data['Victim.injury'].apply(lambda x: 1 if x == 'fatal' else 0)
    data['Non_Fatal'] = data['Fatal'].apply(lambda x: 1 if x == 0 else 0)

# Clean up 'Incident.year'
if 'Incident.year' in data.columns:
    data['Incident.year'] = pd.to_numeric(data['Incident.year'], errors='coerce')
    data.dropna(subset=['Incident.year'], inplace=True)
    data['Year'] = data['Incident.year'].astype(int)

# Normalize 'State'
data['State'] = data['State'].str.strip().str.lower()


# B) DATAFRAME "data2" for SPLOM
data2 = pd.read_excel("data/Data2.xlsx", sheet_name=0)

# Fill missing for key columns
data2['Provoked/unprovoked'] = data2['Provoked/unprovoked'].fillna('Unknown')
data2['State'] = data2['State'].fillna('Unknown')
data2['Shark.common.name'] = data2['Shark.common.name'].fillna('Unknown')
data2['Victim.age'] = pd.to_numeric(data2['Victim.age'], errors='coerce')

# Convert numeric columns & fill NAs
numeric_cols_sp = [
    'Incident.year', 'Shark.length.m', 'Victim.age',
    'Depth.of.incident.m', 'Water.temperature.°C', 'Distance.to.shore.m'
]
for col in numeric_cols_sp:
    if col in data2.columns:
        data2[col] = pd.to_numeric(data2[col], errors='coerce').fillna(0)

data2['Provoked/unprovoked'] = data2['Provoked/unprovoked'].fillna('Unknown')
data2['State'] = data2['State'].str.strip().str.lower()
data2['Date'] = pd.to_datetime(data2['Incident.year'].astype(str) + '-01', errors='coerce')


# ─────────────────────────────────────────────────────────
# 2) PREPARE SUBSETS FOR CHARTS
# ─────────────────────────────────────────────────────────

# (A) Ranking data for bar chart
if {'State', 'Year', 'Incident.day'}.issubset(data.columns):
    ranked_data = data.groupby(['State', 'Year'], dropna=False).agg(
        Total_Incidents=('Incident.day', 'count'),
        Fatal_Incidents=('Fatal', 'sum'),
        Non_Fatal_Incidents=('Non_Fatal', 'sum')
    ).reset_index()
else:
    ranked_data = pd.DataFrame(columns=['State','Year','Total_Incidents','Fatal_Incidents','Non_Fatal_Incidents'])

# (B) Heatmap for activity-location
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


# For SPLOM
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


# ─────────────────────────────────────────────────────────
# 3) DASH APP LAYOUT
# ─────────────────────────────────────────────────────────

app = dash.Dash(__name__)

# Dropdown for the map & heatmap
dropdown_options = [
    {"label": "Shark Name",         "value": "Shark.common.name"},
    {"label": "Provoked Incident",  "value": "Provoked.unprovoked"},
    {"label": "Victim Activity",    "value": "Victim.activity"},
    {"label": "Injury Severity",    "value": "Injury.severity"},
    {"label": "Victim Gender",      "value": "Victim.gender"},
]

app.layout = html.Div([
    html.H1("Shark Incident Analysis Dashboard 🦈", style={'textAlign': 'center'}),

    # Top Section (Map & Heatmap in left column, bar chart & activity heatmap in right column)
    html.Div([
        html.Div([
            # Filtered Incident Map
            html.Div([
                html.H3("Filtered Incident Map", style={'textAlign': 'center'}),
                dcc.Dropdown(
                    id="map-filter-dropdown",
                    options=dropdown_options,
                    value=dropdown_options[0]["value"],
                    clearable=False,
                    style={'marginBottom': '25px', 'width': '90%', 'margin': '0 auto'}
                ),
                dcc.Graph(id="map-graph", style={'height': '40vh', 'width': '100%'})
            ], style={'padding': '20px'}),

            # Heatmap of High-Risk Areas
            html.Div([
                html.H3("Heatmap of High-Risk Areas 🔥", style={'textAlign': 'center'}),
                dcc.Graph(id="heatmap-graph", style={'height': '50vh', 'width': '100%'})
            ], style={'padding': '20px'})

        ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        html.Div([
            # Bar Chart
            html.Div([
                html.H3("Ranked Bar Chart of Shark Incidents", style={'textAlign': 'center'}),
                html.Label("Select Year Range:"),
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
                    inline=True
                ),
                html.Label("Toggle Logarithmic Scale:"),
                dcc.Checklist(
                    id="log-scale",
                    options=[{"label": "Log Scale", "value": "log"}],
                    value=[]
                ),
                dcc.Graph(id="bar-chart", style={'height': '40vh', 'width': '100%'})
            ], style={'padding': '20px'}),

            # Activity-Location Heatmap
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
                    inline=True
                ),
                dcc.Graph(id="activity-location-heatmap", style={'height': '40vh', 'width': '100%'})
            ], style={'padding': '20px'})
        ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ]),

    # SPLOM
    html.Div([
        html.H3("Filtered Scatterplot Matrix (SPLOM) 💠", style={'textAlign': 'center'}),
        dcc.Graph(id='sp_matrix', style={'height': '80vh', 'width': '100%'}),
        html.Div([
            html.Div([
                html.Label("Filter by State:"),
                dcc.Dropdown(
                    id='state-filter',
                    options=[{"label": "Total", "value": "Total"}] + [
                        {"label": st, "value": st} for st in sorted(data2['State'].unique()) if st
                    ],
                    value='Total',
                    clearable=True
                )
            ], style={'width': '45%', 'display': 'inline-block'}),

            html.Div([
                html.Label("Filter by Shark Type:"),
                dcc.Dropdown(
                    id='shark-type-filter',
                    options=[{"label": "Total", "value": "Total"}] + [
                        {"label": sh, "value": sh} for sh in sorted(data2['Shark.common.name'].unique()) if sh
                    ],
                    value='Total',
                    clearable=True
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

    # Additional row with line & pie charts
    html.Div([
        html.Div([dcc.Graph(id='line-chart')], style={"width": "80%", "display": "inline-block"}),
        html.Div([
            html.H4("Attack Type Distribution", style={"textAlign": "center"}),
            dcc.Graph(id='pie-chart')
        ], style={"width": "18%", "display": "inline-block", "verticalAlign": "top"})
    ], style={'marginTop': '30px'}),

    html.Div([
        dcc.RangeSlider(
            id='date-slider',
            min=data2['Date'].dt.year.min(),
            max=data2['Date'].dt.year.max(),
            step=1,
            marks={year: str(year) for year in range(data2['Date'].dt.year.min(), data2['Date'].dt.year.max() + 1, 10)},
            value=[data2['Date'].dt.year.min(), data2['Date'].dt.year.max()]
        )
    ], style={"padding": "20px"})
])


# ─────────────────────────────────────────────────────────
# 4) DEFINE CALLBACKS
# ─────────────────────────────────────────────────────────

# (A) Single callback for Map + Heatmap
@app.callback(
    [Output("map-graph", "figure"),
     Output("heatmap-graph", "figure")],
    Input("map-filter-dropdown", "value")
)
def update_map_and_heatmap(selected_column):
    """
    Builds both the multi-category scatter map AND the density heatmap
    from the same filtered subset of 'data'.
    """
    # 1) Filter out rows without lat/lon
    filtered = data.dropna(subset=["Latitude", "Longitude"]).copy()

    # 2) If the selected column is "Injury.severity," remove "male" entries, etc.
    #    (This replicates your special logic from earlier.)
    if selected_column == "Injury.severity":
        filtered = filtered[filtered["Injury.severity"] != "male"]

    # 3) Build unique categories for the map
    if selected_column not in filtered.columns:
        # If the user picks something invalid, show empty
        empty_fig1 = px.scatter_mapbox(title="Invalid Column Selection").update_layout(height=400)
        empty_fig2 = px.scatter_mapbox(title="No Heatmap").update_layout(height=400)
        return empty_fig1, empty_fig2

    # Drop any row that doesn't have the chosen column
    filtered = filtered.dropna(subset=[selected_column])
    if filtered.empty:
        empty_fig1 = px.scatter_mapbox(title="No Data After Filtering").update_layout(height=400)
        empty_fig2 = px.scatter_mapbox(title="No Heatmap").update_layout(height=400)
        return empty_fig1, empty_fig2

    unique_vals = filtered[selected_column].unique()
    unique_vals = [x for x in unique_vals if x != "nan"]

    # 4) Build the multi-trace Scattermapbox
    traces = []
    color_list = qualitative.Plotly  # e.g. 10 colors from Plotly's palette
    for i, val in enumerate(unique_vals):
        sub_df = filtered[filtered[selected_column] == val]
        hover_text = sub_df.apply(
            lambda row: f"Shark: {row['Shark.common.name']},\n"
                        f"Provoked?: {row['Provoked.unprovoked']},\n"
                        f"Severity: {row['Injury.severity']},\n"
                        f"Gender: {row['Victim.gender']}", axis=1
        )
        trace = go.Scattermapbox(
            lat=sub_df["Latitude"],
            lon=sub_df["Longitude"],
            mode="markers",
            marker=dict(size=8, color=color_list[i % len(color_list)]),
            name=str(val),
            text=hover_text,
            hoverinfo="text"
        )
        traces.append(trace)

    map_fig = go.Figure(
        data=traces,
        layout=go.Layout(
            mapbox=dict(
                style="open-street-map",
                zoom=3,
                center=dict(lat=-25, lon=135)
            ),
            height=500,
            title=f"Filtered Incident Map",
            legend=dict(title=f"{selected_column}")
        )
    )

    # 5) Build the density heatmap using the *same subset*
    heatmap_fig = px.density_mapbox(
        filtered,
        lat='Latitude',
        lon='Longitude',
        radius=10,
        center=dict(lat=filtered['Latitude'].mean(), lon=filtered['Longitude'].mean()),
        zoom=3,
        mapbox_style="open-street-map",
        height=500,
        title="Heatmap of High-Risk Areas"
    )

    return map_fig, heatmap_fig


# (B) Ranked Bar Chart
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
    if ranked_data.empty:
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
            title=f"Grouped Bar: Fatal vs. Non-Fatal ({start_year}-{end_year})",
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
        fig.update_yaxes(range=[0, 450])

    elif chart_mode == "fatal":
        fatal_data = filtered_data.groupby('State').agg(
            Fatal_Incidents=('Fatal_Incidents', 'sum')
        ).reset_index().sort_values(by='Fatal_Incidents', ascending=False)
        fig = px.bar(
            fatal_data,
            x="State",
            y="Fatal_Incidents",
            title=f"Fatal Incidents Only ({start_year}-{end_year})",
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
            title=f"Non-Fatal Incidents Only ({start_year}-{end_year})",
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
            title=f"Stacked Percent Fatal vs. Non-Fatal ({start_year}-{end_year})",
            labels={"Percentage": "Percentage (%)"},
            color_discrete_map={
                "Fatal Incidents": "#E74C3C",
                "Non-Fatal Incidents": "#3498DB"
            }
        )
        fig.update_yaxes(range=[0, 100])

    if "log" in log_scale:
        fig.update_yaxes(type="log")

    return fig


# (C) Activity-Location Heatmap
@app.callback(
    Output("activity-location-heatmap", "figure"),
    [Input("metric-selector", "value")]
)
def update_activity_location_heatmap(metric):
    if heatmap_data.empty:
        return px.bar(title="No data for heatmap")

    fig = px.density_heatmap(
        heatmap_data,
        x="Victim.activity",
        y="State",
        z=metric,
        color_continuous_scale="Viridis",
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
            title="Incident Count",
            ticks="outside"
        )
    )
    return fig


# (D) SPLOM
@app.callback(
    Output('sp_matrix', 'figure'),
    [
        Input('state-filter', 'value'),
        Input('shark-type-filter', 'value'),
        Input('age-slider', 'value')
    ]
)
def update_sp_matrix(selected_state, selected_shark, selected_age_range):
    df_sp = data2.copy()
    if 'Victim.age' in df_sp.columns:
        df_sp['Victim.age'] = pd.to_numeric(df_sp['Victim.age'], errors='coerce')

    # Filter by State
    if selected_state != 'Total':
        df_sp = df_sp[df_sp['State'] == selected_state]

    # Filter by Shark
    if selected_shark != 'Total':
        df_sp = df_sp[df_sp['Shark.common.name'] == selected_shark]

    # Filter by Age
    if 'Victim.age' in df_sp.columns:
        df_sp = df_sp[(df_sp['Victim.age'] >= selected_age_range[0]) &
                      (df_sp['Victim.age'] <= selected_age_range[1])]

    # Ensure color column exists
    if 'Provoked/unprovoked' not in df_sp.columns:
        df_sp['Provoked/unprovoked'] = 'Unknown'

    fig = px.scatter_matrix(
        df_sp,
        dimensions=[col for col in selected_attributes if col in df_sp.columns],
        color='Provoked/unprovoked',
        labels=attribute_labels,
        title="Filtered Scatterplot Matrix (SPLOM)",
        color_discrete_map={
            'provoked':  "#F5B041",
            'unprovoked': "#58D68D",
            'Unknown': "#7F8C8D"
        }
    )

    for axis in fig.layout:
        if axis.startswith('xaxis') or axis.startswith('yaxis'):
            fig.layout[axis].tickformat = ".0f"

    fig.update_layout(template="plotly_white", height=650, width=1200)
    return fig


# (E) Line & Pie charts
@app.callback(
    [Output('line-chart', 'figure'), Output('pie-chart', 'figure')],
    [Input('date-slider', 'value')]
)
def update_line_and_pie(selected_range):
    start_year, end_year = selected_range
    start_date = pd.Timestamp(start_year, 1, 1)
    end_date = pd.Timestamp(end_year, 12, 31)
    filtered_data = data2[(data2['Date'] >= start_date) & (data2['Date'] <= end_date)]

    # Line chart
    yearly_data = filtered_data.groupby(['Date', 'Provoked/unprovoked']).size().reset_index(name='Count')
    line_fig = go.Figure()
    color_map = {
        'provoked':  "#F5B041",
        'unprovoked': "#58D68D",
        'Unknown': "#7F8C8D"
    }
    for incident_type in yearly_data['Provoked/unprovoked'].unique():
        inc_df = yearly_data[yearly_data['Provoked/unprovoked'] == incident_type]
        line_fig.add_trace(
            go.Scatter(
                x=inc_df['Date'],
                y=inc_df['Count'],
                mode='lines+markers',
                name=incident_type,
                line=dict(color=color_map.get(incident_type, '#333333'))
            )
        )
    line_fig.update_layout(
        title="Yearly Shark Incident Trends",
        xaxis_title="Date",
        yaxis_title="Number of Incidents",
        template="plotly_white"
    )

    # Pie chart
    pie_data = filtered_data['Provoked/unprovoked'].value_counts().reset_index()
    pie_data.columns = ['Provoked/unprovoked', 'Count']
    pie_fig = px.pie(
        pie_data,
        values='Count',
        names='Provoked/unprovoked',
        color='Provoked/unprovoked',
        color_discrete_map=color_map,
        hole=0.3
    )
    pie_fig.update_layout(margin=dict(t=15, b=15, l=0, r=0))

    return line_fig, pie_fig


# ─────────────────────────────────────────────────────────
# 5) RUN THE APP
# ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
