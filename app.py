import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px

# Load the dataset
data = pd.read_csv("data/injurydat.csv")

# Data preparation
data['Fatal'] = data['Victim.injury'].apply(lambda x: 1 if str(x).lower() == 'fatal' else 0)
data['Non_Fatal'] = data['Fatal'].apply(lambda x: 1 if x == 0 else 0)

# Clean the 'Incident.year' column
data['Incident.year'] = pd.to_numeric(data['Incident.year'], errors='coerce')  # Convert to numeric, force invalid values to NaN
data = data.dropna(subset=['Incident.year'])  # Drop rows with NaN in 'Incident.year'
data['Year'] = data['Incident.year'].astype(int)  # Convert to integer

# Aggregate data by state and year
ranked_data = data.groupby(['State', 'Year']).agg(
    Total_Incidents=('Incident.day', 'count'),
    Fatal_Incidents=('Fatal', 'sum'),
    Non_Fatal_Incidents=('Non_Fatal', 'sum')
).reset_index()

# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
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
    )
])

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
    # Validate and handle None inputs
    if start_year is None:
        start_year = ranked_data['Year'].min()
    if end_year is None:
        end_year = ranked_data['Year'].max()
    if start_year > end_year:
        return px.bar(title="Error: Start Year cannot be greater than End Year")

    # Filter data by selected year range
    filtered_data = ranked_data[
        (ranked_data['Year'] >= start_year) & (ranked_data['Year'] <= end_year)
    ]


    if chart_mode == "grouped":
        grouped_data = filtered_data.groupby('State', as_index=False).agg(
            Fatal_Incidents=('Fatal_Incidents', 'sum'),
            Non_Fatal_Incidents=('Non_Fatal_Incidents', 'sum'),
            Total_Incidents=('Total_Incidents', 'sum')
        )
        grouped_data = grouped_data.sort_values(by='Total_Incidents', ascending=False)
        grouped_data = pd.melt(
            grouped_data,
            id_vars=['State', 'Total_Incidents'],
            value_vars=['Fatal_Incidents', 'Non_Fatal_Incidents'],
            var_name='Incident_Type',
            value_name='Incident_Count'
        )
        
        # Map values for readability
        incident_type_mapping = {
            "Fatal_Incidents": "Fatal Incidents",
            "Non_Fatal_Incidents": "Non-Fatal Incidents"
        }
        grouped_data['Incident_Type'] = grouped_data['Incident_Type'].map(incident_type_mapping)

        # Create grouped bar chart
        fig = px.bar(
            grouped_data,
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

        # Pass custom data directly to the figure
        fig.update_traces(
            hovertemplate=(
                "<b>State:</b> %{x}<br>"
                "<b>Incident Type:</b> %{customdata[0]}<br>"
                "<b>Number of Incidents:</b> %{y}<br>"
                "<b>Total Incidents in State:</b> %{customdata[1]}<extra></extra>"
            ),
            customdata=grouped_data[['Incident_Type', 'Total_Incidents']].values  # Use .values for NumPy array
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
            Total_Incidents=('Total_Incidents', 'sum'))
        percentage_data['Fatal_Percentage'] = (percentage_data['Fatal_Incidents'] / percentage_data['Total_Incidents']) * 100
        percentage_data['Non_Fatal_Percentage'] = (percentage_data['Non_Fatal_Incidents'] / percentage_data['Total_Incidents']) * 100

        percentage_data_melted = pd.melt(
            percentage_data,
            id_vars=['State', 'Total_Incidents'],
            value_vars=['Fatal_Percentage', 'Non_Fatal_Percentage'],
            var_name='Incident_Type',
            value_name='Percentage'
        )

        # Map incident types for better readability
        incident_type_mapping = {
            "Fatal_Percentage": "Fatal Incidents",
            "Non_Fatal_Percentage": "Non-Fatal Incidents"
        }
        percentage_data_melted['Incident_Type'] = percentage_data_melted['Incident_Type'].map(incident_type_mapping)

        fig = px.bar(
            percentage_data_melted,
            x="State",
            y="Percentage",
            color="Incident_Type",
            barmode="stack",
            title=f"Stacked Bar Chart: Percentage of Fatal vs. Non-Fatal Incidents ({start_year}-{end_year})",
            labels={
                "Percentage": "Percentage (%)",
            },
            color_discrete_map={
                "Fatal Incidents": "lightcoral",
                "Non-Fatal Incidents": "lightblue"
            }
        )

        # Add custom hovertemplate for cleaner hover labels
        fig.update_traces(
            hovertemplate=(
                "<b>Percentage:</b> %{y:.2f}%<br>"  # Percentage
            ),
            customdata=percentage_data_melted[['Incident_Type', 'Total_Incidents']].to_numpy(),
            showlegend=False

        )

        fig.update_yaxes(range=[0, 100])  # Fix y-axis range to [0, 100]


    return fig


if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
