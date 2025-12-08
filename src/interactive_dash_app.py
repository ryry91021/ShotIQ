import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
import pandas as pd
import numpy as np


class InteractiveDashApp:
    """
    Dash-based interactive basketball court application.
    Users can click on the court to get shot probability predictions.
    Supports switching between different players.
    """

    def __init__(self, df, predictors_dict):
        """
        Initialize the Dash app.
        
        Args:
            df: DataFrame with shot data
            predictors_dict: Dictionary with player names as keys and trained predictors as values
        """
        self.df = df
        self.predictors_dict = predictors_dict
        self.available_players = sorted(list(predictors_dict.keys()))
        self.current_player = self.available_players[0] if self.available_players else None
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()

    def draw_court(self, fig):
        """
        Draws NBA court lines on the plotly figure.
        """
        # Court outline
        fig.add_shape(type="rect", x0=0, y0=0, x1=50, y1=47, line=dict(color="Black", width=2))

        # Hoop & Backboard
        fig.add_shape(type="circle", x0=24.25, y0=4, x1=25.75, y1=5.5, line_color="Black")
        fig.add_shape(type="line", x0=22, y0=4, x1=28, y1=4, line=dict(color="Black", width=2))

        # Paint (Key)
        fig.add_shape(type="rect", x0=17, y0=0, x1=33, y1=19, line=dict(color="Black", width=2))
        
        # Free Throw Circle
        fig.add_shape(type="circle", x0=19, y0=13, x1=31, y1=25, line=dict(color="Black", width=2))

        # 3-Point Line (Side lines)
        fig.add_shape(type="line", x0=3, y0=0, x1=3, y1=14, line=dict(color="Black", width=2))
        fig.add_shape(type="line", x0=47, y0=0, x1=47, y1=14, line=dict(color="Black", width=2))

    def create_base_figure(self, player_name):
        """
        Creates the base court figure with historical shot data for a given player.
        """
        # Filter data for the player
        player_data = self.df[self.df["player"].str.contains(player_name, case=False, na=False)]

        fig = go.Figure()

        # Draw court lines
        self.draw_court(fig)

        # Add a fully transparent rectangle covering the entire court for click detection
        # This shape is clickable and covers the whole court area
        fig.add_trace(go.Scatter(
            x=[0, 50, 50, 0, 0],
            y=[0, 0, 47, 47, 0],
            fill='toself',
            fillcolor='rgba(0,0,0,0)',
            line=dict(color='rgba(0,0,0,0)'),
            mode='lines',
            hovertemplate='Click to predict shot probability<extra></extra>',
            name='Court',
            showlegend=False
        ))

        # Add historical shots AFTER the transparent rectangle so they're on top
        fig.add_trace(go.Scatter(
            x=player_data['shotX'],
            y=player_data['shotY'],
            mode='markers',
            marker=dict(
                size=8,
                color=player_data['made'],
                colorscale=[[0, 'rgba(255, 0, 0, 0.6)'], [1, 'rgba(0, 255, 0, 0.6)']],
                showscale=False
            ),
            text=[f"Distance: {d:.1f}ft<br>Made: {m}" 
                  for d, m in zip(player_data['distance'], player_data['made'])],
            hovertemplate='<b>Historical Shot</b><br>%{text}<extra></extra>',
            name="Historical Shots",
            showlegend=True
        ))

        fig.update_layout(
            title=f"<b>{player_name}</b> - Click on court to predict shot probability",
            xaxis=dict(range=[0, 50], showgrid=False, visible=False),
            yaxis=dict(range=[0, 47], showgrid=False, visible=False),
            width=800,
            height=800,
            template="plotly_white",
            hovermode='closest',
            clickmode='event+select'
        )

        return fig

    def setup_layout(self):
        """
        Set up the Dash app layout.
        """
        self.app.layout = html.Div([
            html.Div([
                html.Div([
                    html.H1("Shot Probability Predictor", 
                            style={'textAlign': 'center', 'marginBottom': 10, 'marginRight': 20}),
                    html.Div([
                        html.Label("Select Player:", style={'fontWeight': 'bold', 'marginRight': 10}),
                        dcc.Dropdown(
                            id='player-selector',
                            options=[{'label': p, 'value': p} for p in self.available_players],
                            value=self.current_player,
                            style={'width': '200px'}
                        ),
                    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
                ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'marginBottom': 20}),
                html.P("Click anywhere on the court to predict the shot probability at that location.",
                       style={'textAlign': 'center', 'fontSize': 16, 'color': '#666'}),
            ]),
            
            html.Div([
                dcc.Graph(
                    id='court-graph',
                    figure=self.create_base_figure(self.current_player),
                    style={'display': 'inline-block', 'width': '65%', 'verticalAlign': 'top'}
                ),
                html.Div(
                    id='prediction-output',
                    style={
                        'display': 'inline-block',
                        'width': '33%',
                        'verticalAlign': 'top',
                        'padding': '20px',
                        'backgroundColor': '#f9f9f9',
                        'borderLeft': '1px solid #ddd',
                        'fontFamily': 'monospace',
                        'height': '800px',
                        'overflowY': 'auto'
                    },
                    children=[
                        html.H3("Prediction Results", style={'marginTop': 0}),
                        html.P("Click on the court to see predictions here.", style={'color': '#999'})
                    ]
                )
            ], style={'display': 'flex'}),

            dcc.Store(id='selected-player-store', data=self.current_player),
            dcc.Store(id='click-data-store', data=None)
        ], style={'fontFamily': 'Arial, sans-serif', 'padding': '20px'})

    def setup_callbacks(self):
        """
        Set up Dash callbacks for player selection and click events.
        """
        # Callback 1: Update the store when player selector changes
        @self.app.callback(
            Output('selected-player-store', 'data'),
            Input('player-selector', 'value'),
            prevent_initial_call=False
        )
        def update_selected_player(selected_player):
            """Store the selected player."""
            if selected_player not in self.available_players:
                selected_player = self.current_player
            return selected_player

        # Callback 2: Update the court figure when player changes (no clicks)
        @self.app.callback(
            Output('court-graph', 'figure'),
            Input('selected-player-store', 'data'),
            prevent_initial_call=False
        )
        def update_court_for_player(selected_player):
            """Update court display when player is selected."""
            if not selected_player or selected_player not in self.available_players:
                selected_player = self.current_player
            return self.create_base_figure(selected_player)

        # Callback 3: Handle court clicks and show predictions
        @self.app.callback(
            [Output('court-graph', 'figure'),
             Output('prediction-output', 'children')],
            Input('court-graph', 'clickData'),
            State('selected-player-store', 'data'),
            prevent_initial_call=True
        )
        def handle_court_click(clickData, selected_player):
            """Handle clicks on the court and show predictions."""
            if not selected_player or selected_player not in self.available_players:
                selected_player = self.current_player
            
            if not clickData or 'points' not in clickData or len(clickData['points']) == 0:
                return self.create_base_figure(selected_player), [
                    html.H3("Prediction Results", style={'marginTop': 0}),
                    html.P("Click on the court to see predictions here.", style={'color': '#999'})
                ]

            point = clickData['points'][0]
            shotX = point.get('x')
            shotY = point.get('y')

            if shotX is None or shotY is None:
                return self.create_base_figure(selected_player), [
                    html.H3("Prediction Results", style={'marginTop': 0}),
                    html.P("Click on a court location to get a prediction.", style={'color': '#999'})
                ]

            # Get prediction from the selected player's predictor
            shot_type = 3  # Default shot type
            try:
                predictor = self.predictors_dict[selected_player]
                prob = predictor.predict_probability(
                    shotX=float(shotX),
                    shotY=float(shotY),
                    shot_type=shot_type
                )
                
                # Calculate distance from hoop
                hoop_x, hoop_y = 25, 4.75
                distance = np.sqrt((float(shotX) - hoop_x)**2 + (float(shotY) - hoop_y)**2)
                
                # Create a fresh figure from the base
                fig = self.create_base_figure(selected_player)
                
                # Add a marker to the court where they clicked
                fig.add_trace(go.Scatter(
                    x=[shotX],
                    y=[shotY],
                    mode='markers',
                    marker=dict(size=15, color='blue', symbol='x', line=dict(color='blue', width=3)),
                    hovertemplate=f"<b>Predicted Shot</b><br>Probability: {prob*100:.1f}%<extra></extra>",
                    name='Predicted Shot',
                    showlegend=True
                ))
                
                # Create prediction display
                prediction_text = [
                    html.H3("Shot Prediction", style={'marginTop': 0, 'color': '#333'}),
                    html.Hr(),
                    html.Div([
                        html.P(f"Location: ({shotX:.1f}, {shotY:.1f})", style={'margin': '10px 0'}),
                        html.P(f"Distance from Hoop: {distance:.2f} ft", style={'margin': '10px 0'}),
                        html.Hr(),
                        html.H2(f"{prob*100:.1f}%", style={'color': '#2ecc71', 'marginBottom': 10, 'marginTop': 10}),
                        html.P(f"{selected_player} has a {prob*100:.1f}% chance of making this shot.",
                               style={'fontSize': 14, 'marginTop': 0}),
                    ], style={'backgroundColor': '#fff', 'padding': '15px', 'borderRadius': '5px', 'marginTop': '20px'})
                ]
                
                return fig, prediction_text
                
            except Exception as e:
                error_text = [
                    html.H3("Error", style={'marginTop': 0, 'color': '#e74c3c'}),
                    html.P(f"Could not make prediction: {str(e)}", style={'color': '#c0392b'})
                ]
                return self.create_base_figure(selected_player), error_text

    def run(self, debug=True, port=8050):
        """
        Run the Dash app server.
        
        Args:
            debug: Whether to run in debug mode (default: True)
            port: Port to run the server on (default: 8050)
        """
        print(f"\n🏀 Starting Shot Probability Predictor...")
        print(f"📊 Open your browser and go to: http://127.0.0.1:{port}")
        print(f"Available players: {', '.join(self.available_players)}\n")
        self.app.run(debug=debug, port=port)
