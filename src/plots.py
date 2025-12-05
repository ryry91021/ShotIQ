from matplotlib.patches import Circle, Rectangle, Arc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np




class CourtPlotter:
    """
    Class to plot a basketball court using matplotlib.
    """

    def __init__(self, player=None, team=None):
        self.player = player
        self.team = team

    def __draw_court_feet(self, ax=None, color='black', lw=2, outer_lines=True):
        """
        NBA half court in feet.
        inputs: None
        output: matplotlib Axes object with court drawn
        """
        if ax is None:
            ax = plt.gca()

        # Key dims
        hoop_x, hoop_y = 25, 4.75
        rim_r = 0.75                 # 9" radius
        backboard_y = 4.0            # just in front of rim center
        paint_h = 19.0               # key depth
        ft_radius = 6.0
        three_r = 23.75
        corner_h = 14.0
        halfcourt_y = 47.0

        # --- Hoop / backboard ---
        hoop = Circle((hoop_x, hoop_y), rim_r, lw=lw, color=color, fill=False)
        backboard = Rectangle((hoop_x - 3, backboard_y), 6, 0.1, lw=lw, color=color)

        # --- Paint (key) ---
        outer_box = Rectangle((hoop_x - 8, 0), 16, paint_h, lw=lw, color=color, fill=False)
        inner_box = Rectangle((hoop_x - 6, 0), 12, paint_h, lw=lw, color=color, fill=False)

        # --- Free throw circle (TOP solid, BOTTOM dashed) ---
        free_throw_top = Arc((hoop_x, paint_h), 2*ft_radius, 2*ft_radius,
                            theta1=0, theta2=180, lw=lw, color=color)
        free_throw_bottom = Arc((hoop_x, paint_h), 2*ft_radius, 2*ft_radius,
                                theta1=180, theta2=360, lw=lw, color=color, linestyle='dashed')

        # --- Restricted area (TOP half) ---
        restricted = Arc((hoop_x, hoop_y), 8, 8, theta1=0, theta2=180, lw=lw, color=color)

        # --- Three-point line ---
        # Corners: vertical lines at x = 25 ± 22 up to 14 ft
        corner_left  = Rectangle((hoop_x - 22, 0), 0, corner_h, lw=lw, color=color)
        corner_right = Rectangle((hoop_x + 22, 0), 0, corner_h, lw=lw, color=color)
        # Arc centered at hoop; angles computed so it meets corner lines (~21.6° and 158.4°)
        three_arc = Arc((hoop_x, hoop_y), 2*three_r, 2*three_r,
                        theta1=21.6, theta2=158.4, lw=lw, color=color)

        # --- Half-court arcs ---
        center_outer = Arc((25, halfcourt_y), 12, 12, theta1=0, theta2=180, lw=lw, color=color)
        center_inner = Arc((25, halfcourt_y), 4, 4, theta1=0, theta2=180, lw=lw, color=color)

        elems = [hoop, backboard, outer_box, inner_box,
                free_throw_top, free_throw_bottom, restricted,
                corner_left, corner_right, three_arc,
                center_outer, center_inner]

        if outer_lines:
            boundary = Rectangle((0, 0), 50, 47, lw=lw, color=color, fill=False)
            elems.append(boundary)

        for e in elems:
            ax.add_patch(e)

        return ax
    
    def plot_shot_data(self, df, player=None, figsize=(12, 11)):
        """
        Plot makes and misses on a basketball court for a given player
        inputs:
            df - DataFrame with shot data; must include 'shotX', 'shotY', 'made' columns
            player - optional player name for title and selection
            figsize - size of the figure
            court_color - background color of the court
        output: matplotlib Axes object (basketball court plot)
        """
        sns.set_style("white")

        #Validation
        if not isinstance(player, str):
            raise ValueError("Player name must be provided as a string.")
        
        data=df[df["player"].str.contains(player, case=False, na=False)]
        fig, ax = plt.subplots(figsize=figsize)
        self.__draw_court_feet(ax, outer_lines=True)

        colors = np.where(data["made"] == 1, "green", "red")
        ax.scatter(data["shotX"], data["shotY"], c=colors, alpha=0.7, s=30, edgecolors="none")

        ax.set_xlim(0, 50)
        ax.set_ylim(0, 47)
        ax.set_title(f"{player} — Made (Green) vs Missed (Red) Shots", y=1.03, fontsize=16)
        ax.set_xlabel("Court Width (feet)")
        ax.set_ylabel("Court Length (feet)")
        ax.tick_params(labelbottom=False, labelleft=False)

        print("Displaying court...")
        plt.show()



