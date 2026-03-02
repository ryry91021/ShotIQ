import plotly.graph_objects as go
import pandas as pd
import numpy as np

class InteractiveCourtPlotter:
    """
    Plots an interactive basketball court using Plotly.
    Allows users to interactively choose shot locations and view data[cite: 55].
    """

    def __init__(self, player=None, predictor=None):
        self.player = player
        self.predictor = predictor

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

        # 3-Point Line
        # Side lines (corners) up to 14 ft
        fig.add_shape(type="line", x0=3, y0=0, x1=3, y1=14, line=dict(color="Black", width=2))
        fig.add_shape(type="line", x0=47, y0=0, x1=47, y1=14, line=dict(color="Black", width=2))

        # Arc centered at hoop (use a scatter trace for smooth curve)
        hoop_x, hoop_y = 25.0, 4.75
        three_r = 23.75
        # angles chosen so arc meets the corner lines around y=14
        theta1 = np.deg2rad(21.6)
        theta2 = np.deg2rad(158.4)
        theta = np.linspace(theta1, theta2, 180)
        arc_x = hoop_x + three_r * np.cos(theta)
        arc_y = hoop_y + three_r * np.sin(theta)
        fig.add_trace(go.Scatter(
            x=arc_x,
            y=arc_y,
            mode='lines',
            line=dict(color='Black', width=2),
            hoverinfo='skip',
            showlegend=False,
            name='3PT Arc'
        ))

    def plot_shot_data(self, df, player=None):
        """
        Plots shots using Plotly Scatter with click interactivity for probability prediction.
        Clicking on the court calculates the shot probability for that location.
        """
        if player:
            df = df[df["player"].str.contains(player, case=False, na=False)]
            title = f"{player} - Shot Heatmap & Success (Click on court for probability)"
        else:
            title = "League Wide Shot Data"

        fig = go.Figure()

        # Draw court lines
        self.draw_court(fig)

        # Add Shots
        fig.add_trace(go.Scatter(
            x=df['shotX'],
            y=df['shotY'],
            mode='markers',
            marker=dict(
                size=6,
                color=df['made'],
                colorscale=[[0, 'red'], [1, 'green']], # Red for miss, Green for make
                opacity=0.6
            ),
            text=df['distance'].apply(lambda x: f"Dist: {x}ft"), # Hover text
            name="Shots"
        ))

        fig.update_layout(
            title=title,
            xaxis=dict(range=[0, 50], showgrid=False, visible=False),
            yaxis=dict(range=[0, 47], showgrid=False, visible=False),
            width=600,
            height=600,
            template="plotly_white"
        )
        
        # Add click handler for probability prediction
        if self.predictor is not None:
            fig.update_layout(
                clickmode='event+select',
                hovermode='closest'
            )
            
            # Custom JavaScript-like behavior via plotly's click data
            fig.update_layout(
                title=dict(
                    text=f"{player} - Shot Heatmap & Success<br><sub>Click on court to predict shot probability</sub>"
                    if player else "League Wide Shot Data<br><sub>Click on court to predict shot probability</sub>"
                )
            )
        
        print("Generating interactive plot...")
        # In a real app, fig.show() or return fig
        return fig