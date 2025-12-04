import plotly.graph_objects as go
import pandas as pd

class InteractiveCourtPlotter:
    """
    Plots an interactive basketball court using Plotly.
    Allows users to interactively choose shot locations and view data[cite: 55].
    """

    def __init__(self, player=None):
        self.player = player

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

        # 3-Point Line (Simplified approximation for Plotly)
        # Side lines
        fig.add_shape(type="line", x0=3, y0=0, x1=3, y1=14, line=dict(color="Black", width=2))
        fig.add_shape(type="line", x0=47, y0=0, x1=47, y1=14, line=dict(color="Black", width=2))
        # Arc (Plotly doesn't have a simple 'arc' primitive like mpl, using SVG path or scatter would be precise
        # but for this demo we rely on the scatter data to define the shape visually)

    def plot_shot_data(self, df, player=None):
        """
        Plots shots using Plotly Scatter.
        """
        if player:
            df = df[df["player"].str.contains(player, case=False, na=False)]
            title = f"{player} - Shot Heatmap & Success"
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
        
        print("Generating interactive plot...")
        # In a real app, fig.show() or return fig
        return fig