import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
from dash import dash_table
import os
import random
import datetime
import json
import numpy as np

# Expanded sample data to simulate realistic NASA seismic data
np.random.seed(42)  # For reproducibility
n_samples = 10000

time_values = np.linspace(0, 80000, n_samples)
# Simulating a seismic event with random noise
amplitude_values = np.random.normal(0, 1e-9, n_samples)
# Adding a significant event around the middle
amplitude_values[4000:4500] += np.linspace(0, 5e-9, 500)
# Adding post-seismic activity
decay = np.exp(-np.linspace(0, 5, 500))
amplitude_values[4500:5000] += decay * 5e-9

# Creating DataFrame for the dataset
df = pd.DataFrame({
    'time': time_values,
    'amplitude': amplitude_values
})

# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout of the app
app.layout = dbc.Container([
    html.Div([
        html.H1("Seismic Detection Platform", className="text-center my-4", style={"color": "#ffcc00", "font-family": "'Arial Black', sans-serif", "text-shadow": "2px 2px #003366", "padding": "10px", "borderBottom": "3px solid #003366"}),
        html.Div([
            html.Img(src="/.venv/37dfd7c7-6189-4f23-88b0-c94e64a11096.webp", style={"width": "150px", "height": "auto", "margin": "0 auto", "display": "block", "padding": "20px"}),
        ]),
        dcc.Tabs(id='tabs', className='custom-tabs', children=[
            dcc.Tab(label='Data Upload', className='custom-tab', selected_className='custom-tab--selected', children=[
                html.Div([
                    html.H5("Upload Seismic Data Files", className="my-3", style={"color": "#ffcc00", "font-family": "'Arial', sans-serif", "text-shadow": "1px 1px #003366", "backgroundColor": "#003366", "padding": "10px", "borderRadius": "10px"}),
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Drag and Drop or ', html.A('Select Files', className="text-primary", style={"color": "#ffcc00", "font-weight": "bold", "textDecoration": "underline"})
                        ]),
                        style={
                            'width': '100%', 'height': '60px', 'lineHeight': '60px',
                            'borderWidth': '1px', 'borderStyle': 'dashed',
                            'borderRadius': '15px', 'textAlign': 'center', 'margin': '10px',
                            'borderColor': '#003366', 'backgroundColor': '#e6f0ff', 'boxShadow': '5px 5px 10px #888888'
                        },
                        multiple=True
                    ),
                    html.Div(["Selected Data Source: NASA"], className="my-3", style={"color": "#ffcc00", "font-family": "'Arial', sans-serif", "font-weight": "bold", "backgroundColor": "#003366", "padding": "5px", "borderRadius": "5px"}),
                    dbc.Button("Data Upload", id='data-upload-button', color='warning', className='mt-3', style={"backgroundColor": "#ffcc00", "borderColor": "#003366", "color": "#003366", "boxShadow": "3px 3px 8px #666666", "transition": "background-color 0.3s ease", "hover": {"backgroundColor": "#e6b800"}}),
                ])
            ]),
            dcc.Tab(label='Real-Time Analysis', className='custom-tab', selected_className='custom-tab--selected', children=[
                html.Div([
                    html.H5("Real-Time Seismic Wave Analysis", className="my-3", style={"color": "#ffcc00", "font-family": "'Arial', sans-serif", "text-shadow": "1px 1px #003366", "borderLeft": "5px solid #ffcc00", "paddingLeft": "10px"}),
                    dcc.Graph(
                        id='seismic-graph',
                        figure={
                            'data': [
                                go.Scatter(x=df['time'], y=df['amplitude'], mode='lines', name='Amplitude')
                            ],
                            'layout': go.Layout(
                                title='Real-Time Seismic Wave Analysis',
                                xaxis={'title': 'Time (seconds)', 'titlefont': {'color': '#ffcc00'}, 'tickfont': {'color': '#ffcc00'}},
                                yaxis={'title': 'Amplitude (m/s)', 'titlefont': {'color': '#ffcc00'}, 'tickfont': {'color': '#ffcc00'}},
                                margin={'l': 40, 'b': 40, 't': 60, 'r': 10},
                                hovermode='closest',
                                plot_bgcolor='#003366',
                                paper_bgcolor='#003366',
                                font={'color': '#ffcc00'},
                                titlefont={'size': 22},
                                showlegend=True
                            )
                        }
                    ),
                    dbc.Button("Simulate Real-Time Data", id='simulate-data', color='success', className='mt-3', style={"backgroundColor": "#ffcc00", "borderColor": "#003366", "color": "#003366", "boxShadow": "3px 3px 8px #666666", "borderRadius": "5px", "hover": {"backgroundColor": "#e6b800"}}),
                    dcc.Interval(id='interval-component', interval=2000, n_intervals=0)
                ])
            ]),
            dcc.Tab(label='Seismic Event Detection', className='custom-tab', selected_className='custom-tab--selected', children=[
                html.Div([
                    html.H5("Seismic Event Detection Results", className="my-3", style={"color": "#ffcc00", "font-family": "'Arial', sans-serif", "text-shadow": "1px 1px #003366", "borderBottom": "3px solid #ffcc00", "paddingBottom": "10px"}),
                    dcc.Graph(
                        id='event-detection-graph',
                        figure={
                            'data': [
                                go.Scatter(x=df['time'], y=df['amplitude'], mode='lines', name='Filtered Velocity')
                            ],
                            'layout': go.Layout(
                                title='Seismic Event Detection',
                                xaxis={'title': 'Time (seconds)', 'titlefont': {'color': '#ffcc00'}, 'tickfont': {'color': '#ffcc00'}},
                                yaxis={'title': 'Filtered Velocity (m/s)', 'titlefont': {'color': '#ffcc00'}, 'tickfont': {'color': '#ffcc00'}},
                                margin={'l': 40, 'b': 40, 't': 60, 'r': 10},
                                hovermode='closest',
                                plot_bgcolor='#003366',
                                paper_bgcolor='#003366',
                                font={'color': '#ffcc00'},
                                titlefont={'size': 22}
                            )
                        },
                        style={'border': '2px solid #003366', 'borderRadius': '15px', 'padding': '10px', 'backgroundColor': '#e6f0ff', 'boxShadow': '5px 5px 10px #888888', "hover": {"borderColor": "#ffcc00"}}
                    ),
                    dbc.Button("Export Report", id='export-report', color='primary', className='mt-3', style={"backgroundColor": "#ffcc00", "borderColor": "#003366", "color": "#003366", "boxShadow": "3px 3px 8px #666666", "transition": "transform 0.3s ease", "hover": {"transform": "scale(1.1)"}}),
                    dcc.Download(id='download-dataframe-csv'),
                ])
            ]),
            dcc.Tab(label='Interactive Mars Globe', className='custom-tab', selected_className='custom-tab--selected', children=[
                html.Div([
                    html.H5("Explore the Interactive Mars Globe", className="my-3", style={"color": "#ffcc00", "font-family": "'Arial', sans-serif", "text-shadow": "1px 1px #003366", "borderRadius": "5px", "padding": "5px", "backgroundColor": "#003366"}),
                    html.Iframe(
                        srcDoc=open(os.path.join(os.getcwd(), 'interactive_mars_globe.html'), encoding='utf-8').read(),
                        style={"width": "100%", "height": "600px", "border": "2px solid #003366", "borderRadius": "15px", "backgroundColor": "#e6f0ff", "boxShadow": "5px 5px 10px #888888", "transition": "border-color 0.3s ease", "hover": {"borderColor": "#ffcc00"}}
                    )
                ])
            ]),
            dcc.Tab(label='Interactive Moon Globe', className='custom-tab', selected_className='custom-tab--selected', children=[
                html.Div([
                    html.H5("Explore the Interactive Moon Globe", className="my-3", style={"color": "#ffcc00", "font-family": "'Arial', sans-serif", "text-shadow": "1px 1px #003366", "borderRadius": "5px", "padding": "5px", "backgroundColor": "#003366"}),
                    html.Iframe(
                        srcDoc=open(os.path.join(os.getcwd(), 'interactive_moon_globe.html'), encoding='utf-8').read(),
                        style={"width": "100%", "height": "600px", "border": "2px solid #003366", "borderRadius": "15px", "backgroundColor": "#e6f0ff", "boxShadow": "5px 5px 10px #888888", "transition": "border-color 0.3s ease", "hover": {"borderColor": "#ffcc00"}}
                    )
                ])
            ]),
            dcc.Tab(label='Earthquake Statistics', className='custom-tab', selected_className='custom-tab--selected', children=[
                html.Div([
                    html.H5("Global Earthquake Statistics", className="my-3", style={"color": "#ffcc00", "font-family": "'Arial', sans-serif", "text-shadow": "1px 1px #003366", "borderBottom": "3px solid #ffcc00", "paddingBottom": "10px"}),
                    dcc.Graph(
                        id='earthquake-statistics-graph',
                        figure={
                            'data': [
                                go.Bar(x=['Magnitude 3-4', 'Magnitude 4-5', 'Magnitude 5-6', 'Magnitude 6+'], y=[200, 150, 50, 20], name='Earthquakes')
                            ],
                            'layout': go.Layout(
                                title='Global Earthquake Statistics',
                                xaxis={'title': 'Magnitude Range', 'titlefont': {'color': '#ffcc00'}, 'tickfont': {'color': '#ffcc00'}},
                                yaxis={'title': 'Number of Earthquakes', 'titlefont': {'color': '#ffcc00'}, 'tickfont': {'color': '#ffcc00'}},
                                margin={'l': 40, 'b': 40, 't': 60, 'r': 10},
                                hovermode='closest',
                                plot_bgcolor='#003366',
                                paper_bgcolor='#003366',
                                font={'color': '#ffcc00'},
                                titlefont={'size': 22}
                            )
                        },
                        style={'border': '2px solid #003366', 'borderRadius': '15px', 'padding': '10px', 'backgroundColor': '#e6f0ff', 'boxShadow': '5px 5px 10px #888888', 'hover': {'borderColor': '#ffcc00'}}
                    ),
                    dbc.Button("Refresh Statistics", id='refresh-statistics-button', color='info', className='mt-3', style={"backgroundColor": "#ffcc00", "borderColor": "#003366", "color": "#003366", "boxShadow": "3px 3px 8px #666666", "borderRadius": "5px", "hover": {"transform": "scale(1.1)"}})
                ])
            ]),
            dcc.Tab(label='Seismic Hazard Map', className='custom-tab', selected_className='custom-tab--selected', children=[
                html.Div([
                    html.H5("Seismic Hazard Mapping", className="my-3", style={"color": "#ffcc00", "font-family": "'Arial', sans-serif", "text-shadow": "1px 1px #003366", "borderBottom": "3px solid #ffcc00", "paddingBottom": "10px"}),
                    dcc.Graph(
                        id='hazard-map',
                        figure={
                            'data': [
                                go.Contour(z=np.random.rand(10, 10), colorscale='Viridis', name='Hazard Levels')
                            ],
                            'layout': go.Layout(
                                title='Seismic Hazard Map',
                                xaxis={'title': 'Longitude', 'titlefont': {'color': '#ffcc00'}, 'tickfont': {'color': '#ffcc00'}},
                                yaxis={'title': 'Latitude', 'titlefont': {'color': '#ffcc00'}, 'tickfont': {'color': '#ffcc00'}},
                                margin={'l': 40, 'b': 40, 't': 60, 'r': 10},
                                hovermode='closest',
                                plot_bgcolor='#003366',
                                paper_bgcolor='#003366',
                                font={'color': '#ffcc00'},
                                titlefont={'size': 22}
                            )
                        },
                        style={'border': '2px solid #003366', 'borderRadius': '15px', 'padding': '10px', 'backgroundColor': '#e6f0ff', 'boxShadow': '5px 5px 10px #888888', 'hover': {'borderColor': '#ffcc00'}}
                    ),
                    dbc.Button("Generate Hazard Map", id='generate-hazard-map', color='danger', className='mt-3', style={"backgroundColor": "#ffcc00", "borderColor": "#003366", "color": "#003366", "boxShadow": "3px 3px 8px #666666", "borderRadius": "5px", "hover": {"transform": "scale(1.1)"}})
                ])
            ]),
            dcc.Tab(label='Frequency Analysis', className='custom-tab', selected_className='custom-tab--selected', children=[
                html.Div([
                    html.H5("Frequency Domain Analysis", className="my-3", style={"color": "#ffcc00", "font-family": "'Arial', sans-serif", "text-shadow": "1px 1px #003366", "borderBottom": "3px solid #ffcc00", "paddingBottom": "10px"}),
                    dcc.Graph(
                        id='frequency-graph',
                        figure={
                            'data': [
                                go.Scatter(x=np.linspace(0, 20, 500), y=np.abs(np.fft.fft(np.sin(np.linspace(0, 20, 500)))), mode='lines', name='Frequency Spectrum')
                            ],
                            'layout': go.Layout(
                                title='Frequency Domain Analysis',
                                xaxis={'title': 'Frequency (Hz)', 'titlefont': {'color': '#ffcc00'}, 'tickfont': {'color': '#ffcc00'}, 'range': [0, 10]},
                                yaxis={'title': 'Amplitude', 'titlefont': {'color': '#ffcc00'}, 'tickfont': {'color': '#ffcc00'}, 'range': [0, 200]},
                                margin={'l': 40, 'b': 40, 't': 60, 'r': 10},
                                hovermode='closest',
                                plot_bgcolor='#003366',
                                paper_bgcolor='#003366',
                                font={'color': '#ffcc00'},
                                titlefont={'size': 22}
                            )
                        },
                        style={'border': '2px solid #003366', 'borderRadius': '15px', 'padding': '10px', 'backgroundColor': '#e6f0ff', 'boxShadow': '5px 5px 10px #888888', 'hover': {'borderColor': '#ffcc00'}}
                    ),
                    dbc.Button("Perform FFT Analysis", id='fft-button', color='primary', className='mt-3', style={"backgroundColor": "#ffcc00", "borderColor": "#003366", "color": "#003366", "boxShadow": "3px 3px 8px #666666", "borderRadius": "5px", "hover": {"transform": "scale(1.1)"}})
                ])
            ]),
            dcc.Tab(label='Anomaly Detection', className='custom-tab', selected_className='custom-tab--selected', children=[
                html.Div([
                    html.H5("Automatic Anomaly Detection", className="my-3", style={"color": "#ffcc00", "font-family": "'Arial', sans-serif", "text-shadow": "1px 1px #003366", "borderBottom": "3px solid #ffcc00", "paddingBottom": "10px"}),
                    dcc.Graph(
                        id='anomaly-detection-graph',
                        figure={
                            'data': [
                                go.Scatter(x=df['time'], y=df['amplitude'], mode='lines', name='Amplitude'),
                                go.Scatter(x=df[df['amplitude'] > 5e-9]['time'], y=df[df['amplitude'] > 5e-9]['amplitude'], mode='markers', name='Anomalies', marker=dict(color='red', size=10))
                            ],
                            'layout': go.Layout(
                                title='Anomaly Detection',
                                xaxis={'title': 'Time (seconds)', 'titlefont': {'color': '#ffcc00'}, 'tickfont': {'color': '#ffcc00'}},
                                yaxis={'title': 'Amplitude (m/s)', 'titlefont': {'color': '#ffcc00'}, 'tickfont': {'color': '#ffcc00'}},
                                margin={'l': 40, 'b': 40, 't': 60, 'r': 10},
                                hovermode='closest',
                                plot_bgcolor='#003366',
                                paper_bgcolor='#003366',
                                font={'color': '#ffcc00'},
                                titlefont={'size': 22}
                            )
                        },
                        style={'border': '2px solid #003366', 'borderRadius': '15px', 'padding': '10px', 'backgroundColor': '#e6f0ff', 'boxShadow': '5px 5px 10px #888888', 'hover': {'borderColor': '#ffcc00'}}
                    ),
                    dbc.Button("Detect Anomalies", id='detect-anomalies-button', color='warning', className='mt-3', style={"backgroundColor": "#ffcc00", "borderColor": "#003366", "color": "#003366", "boxShadow": "3px 3px 8px #666666", "borderRadius": "5px", "hover": {"transform": "scale(1.1)"}}),
                    html.Div(id='anomaly-detection-result', className='mt-3', style={'color': '#ff0000', 'font-family': "'Arial', sans-serif", 'font-weight': 'bold', 'borderBottom': '2px dashed #ff0000', 'paddingBottom': '5px'})
                ])
            ])
        ], colors={"border": "#003366", "primary": "#003366", "background": "#e6f0ff"})
    ], style={'backgroundColor': '#003366', 'padding': '20px', 'borderRadius': '15px', 'boxShadow': '5px 5px 15px #888888'})
], fluid=True)

# Callbacks to handle interactive behavior
@app.callback(
    Output('event-detection-graph', 'figure'),
    [Input('upload-data', 'contents')]
)
def update_event_detection(contents):
    # In real scenario, process the contents and update graph accordingly
    # Here we just simulate graph update based on uploaded data
    figure = {
        'data': [
            go.Scatter(x=df['time'], y=df['amplitude'], mode='lines', name='Filtered Velocity')
        ],
        'layout': go.Layout(
            title='Seismic Event Detection',
            xaxis={'title': 'Time (seconds)', 'titlefont': {'color': '#ffcc00'}, 'tickfont': {'color': '#ffcc00'}},
            yaxis={'title': 'Filtered Velocity (m/s)', 'titlefont': {'color': '#ffcc00'}, 'tickfont': {'color': '#ffcc00'}},
            margin={'l': 40, 'b': 40, 't': 60, 'r': 10},
            hovermode='closest',
            plot_bgcolor='#003366',
            paper_bgcolor='#003366',
            font={'color': '#ffcc00'},
            titlefont={'size': 22}
        )
    }
    return figure

@app.callback(
    Output('seismic-graph', 'figure'),
    [Input('simulate-data', 'n_clicks'), Input('interval-component', 'n_intervals')],
    prevent_initial_call=True
)
def simulate_real_time_data(n_clicks, n_intervals):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    new_time = df['time'].iloc[-1] + 50
    new_amplitude = random.uniform(3, 6) * 1e-9
    df.loc[len(df)] = [new_time, new_amplitude]

    figure = {
        'data': [
            go.Scatter(x=df['time'], y=df['amplitude'], mode='lines+markers', name='Amplitude')
        ],
        'layout': go.Layout(
            title='Real-Time Seismic Wave Analysis',
            xaxis={'title': 'Time (seconds)', 'titlefont': {'color': '#ffcc00'}, 'tickfont': {'color': '#ffcc00'}},
            yaxis={'title': 'Amplitude (m/s)', 'titlefont': {'color': '#ffcc00'}, 'tickfont': {'color': '#ffcc00'}},
            margin={'l': 40, 'b': 40, 't': 60, 'r': 10},
            hovermode='closest',
            plot_bgcolor='#003366',
            paper_bgcolor='#003366',
            font={'color': '#ffcc00'},
            titlefont={'size': 22}
        )
    }
    return figure

@app.callback(
    Output('frequency-graph', 'figure'),
    [Input('fft-button', 'n_clicks')],
    prevent_initial_call=True
)
def perform_fft_analysis(n_clicks):
    # Perform FFT analysis on amplitude data
    time = df['time']
    amplitude = df['amplitude']
    fft_values = np.fft.fft(amplitude)
    frequencies = np.fft.fftfreq(len(time), d=(time[1] - time[0]))

    figure = {
        'data': [
            go.Scatter(x=frequencies, y=np.abs(fft_values), mode='lines', name='Frequency Spectrum')
        ],
        'layout': go.Layout(
            title='Frequency Domain Analysis',
            xaxis={'title': 'Frequency (Hz)', 'titlefont': {'color': '#ffcc00'}, 'tickfont': {'color': '#ffcc00'}, 'range': [0, 10]},
            yaxis={'title': 'Amplitude', 'titlefont': {'color': '#ffcc00'}, 'tickfont': {'color': '#ffcc00'}, 'range': [0, 200]},
            margin={'l': 40, 'b': 40, 't': 60, 'r': 10},
            hovermode='closest',
            plot_bgcolor='#003366',
            paper_bgcolor='#003366',
            font={'color': '#ffcc00'},
            titlefont={'size': 22}
        )
    }
    return figure

@app.callback(
    [Output('anomaly-detection-graph', 'figure'), Output('anomaly-detection-result', 'children')],
    [Input('detect-anomalies-button', 'n_clicks')],
    prevent_initial_call=True
)
def detect_anomalies(n_clicks):
    # Detect anomalies using a simple threshold method
    threshold = 5e-9
    anomalies = df[df['amplitude'] > threshold]

    anomaly_result = "No anomalies detected" if anomalies.empty else f"Anomalies detected at times: {anomalies['time'].tolist()}"

    figure = {
        'data': [
            go.Scatter(x=df['time'], y=df['amplitude'], mode='lines', name='Amplitude'),
            go.Scatter(x=anomalies['time'], y=anomalies['amplitude'], mode='markers', name='Anomalies', marker=dict(color='red', size=10))
        ],
        'layout': go.Layout(
            title='Anomaly Detection',
            xaxis={'title': 'Time (seconds)', 'titlefont': {'color': '#ffcc00'}, 'tickfont': {'color': '#ffcc00'}},
            yaxis={'title': 'Amplitude (m/s)', 'titlefont': {'color': '#ffcc00'}, 'tickfont': {'color': '#ffcc00'}},
            margin={'l': 40, 'b': 40, 't': 60, 'r': 10},
            hovermode='closest',
            plot_bgcolor='#003366',
            paper_bgcolor='#003366',
            font={'color': '#ffcc00'},
            titlefont={'size': 22}
        )
    }
    return figure, anomaly_result

@app.callback(
    Output('download-dataframe-csv', 'data'),
    [Input('export-report', 'n_clicks')],
    prevent_initial_call=True
)
def download_csv(n_clicks):
    if n_clicks:
        # In a real-world scenario, the data would be processed into a report format
        return dcc.send_data_frame(df.to_csv, f"seismic_report_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")

if __name__ == '__main__':
    app.run_server(debug=True)