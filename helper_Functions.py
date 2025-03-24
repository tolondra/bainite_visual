import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def download_plots_as_pdf(stored_figures, change=None):
    """
    Download and save all plots as a PDF.

    Args:
        stored_figures (list): List to store the generated figures.
        change (optional): Placeholder parameter for future use.
    """
    with output_widget:
        # Clear previous output
        clear_output(wait=True)

        # Clear the stored_figures list to ensure we're starting fresh
        stored_figures.clear()

        # Generate and store all plots when the download button is clicked
        # Call all the plotting functions to generate the plots
        plot_scatter()  # Generate scatter plot

        plot_pareto_and_content_variations()  # Generate Pareto and content variations plot

        plot_pareto_and_variations()  # Generate Pareto and variations plot

        # Save all the stored figures to a PDF
        with PdfPages("all_plots.pdf") as pdf:
            for fig in stored_figures:
                pdf.savefig(fig)  # Save the figure to the PDF
                plt.close(fig)  # Close the figure to avoid conflicts

        st.write("All plots have been saved into 'all_plots.pdf'.")


def calculate_chem_comp_difference_to_next_exp_point(exp_filtered_df, pareto_df):
    """
    Calculate the difference in chemical composition to the nearest experimental point.

    Args:
        exp_filtered_df (DataFrame): The filtered experimental DataFrame.
        pareto_df (DataFrame): The Pareto front DataFrame.

    Returns:
        DataFrame: The updated Pareto front DataFrame with the chemical composition differences.
    """
    # List of chemical elements to calculate the difference
    chemicals = ["C", "Si", "Mn", "Mo", "V", "Al"]

    # Initialize a list to store the differences
    composition_differences = []

    # Loop over each point in the Pareto front
    for _, pareto_point in pareto_df.iterrows():
        pareto_chemical_values = pareto_point[
            chemicals
        ].values  # Get the chemical composition for Pareto point

        # Calculate the squared differences for each experimental data point
        exp_differences = []
        for _, exp_point in exp_filtered_df.iterrows():
            exp_chemical_values = exp_point[
                chemicals
            ].values  # Get the chemical composition for experimental point
            # Calculate the squared differences for all chemical elements
            diff = np.sqrt(np.sum((pareto_chemical_values - exp_chemical_values) ** 2))
            exp_differences.append(diff)

        # Find the minimum difference to the nearest experimental point
        min_composition_diff = min(exp_differences)
        composition_differences.append(min_composition_diff)

    # Create a copy of pareto_df to avoid SettingWithCopyWarning
    pareto_df = pareto_df.copy()
    pareto_df["Diff. in chem. comp. to next Exp. point"] = (
        composition_differences  # Safe assignment
    )

    return pareto_df


def calculate_rel_error(pareto_df):
    """
    Calculate the relative error for UEL and YS.

    Args:
        pareto_df (DataFrame): The Pareto front DataFrame.

    Returns:
        DataFrame: The updated Pareto front DataFrame with the relative error.
    """
    pareto_df = pareto_df.copy()
    pareto_df["Rel Error UEL/YS"] = (pareto_df["Sigma YS"] / pareto_df["YS"]) + (
        pareto_df["Sigma UEL"] / pareto_df["UEL"]
    )
    return pareto_df


def pareto_efficient_points(filtered_df):
    """
    Find the Pareto-efficient points in the filtered DataFrame.

    Args:
        filtered_df (DataFrame): DataFrame containing the 'UEL' and 'YS' columns.

    Returns:
        DataFrame: A DataFrame containing only the Pareto-efficient points.
    """
    # Extract the costs from the DataFrame (UEL and YS), and use -1 for maximization
    costs = filtered_df[["UEL", "YS"]].values * -1  # Multiply by -1 for maximization

    # Initialize a boolean array for tracking efficiency
    is_efficient = np.ones(costs.shape[0], dtype=bool)

    # Loop through each point and check for dominance
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Mark any point dominated by another point as not efficient
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True  # Keep the point itself as Pareto-efficient

    # Return only the efficient points from the filtered DataFrame
    return filtered_df[is_efficient]


def pareto_efficient_points_new(filtered_df):
    """
    More efficient Pareto-efficient computation using sorting.

    Args:
        filtered_df (DataFrame): DataFrame containing the 'UEL' and 'YS' columns.

    Returns:
        DataFrame: A DataFrame containing only the Pareto-efficient points.
    """
    # Extract the costs from the DataFrame (UEL and YS), and use -1 for maximization
    costs = filtered_df[["UEL", "YS"]].values * -1  # Multiply by -1 for maximization

    sorted_indices = np.lexsort(
        costs.T[::-1]
    )  # Sort by last objective, then second-to-last, etc.
    sorted_costs = costs[sorted_indices]
    is_efficient = np.ones(costs.shape[0], dtype=bool)

    for i in range(len(sorted_costs) - 1):
        if is_efficient[sorted_indices[i]]:
            # Eliminate dominated points
            is_efficient[sorted_indices[i + 1 :]] = np.any(
                sorted_costs[i + 1 :] < sorted_costs[i], axis=1
            )

    return filtered_df[is_efficient]


def filter_data(df, sliders, change=None):
    """
    Filter the DataFrame based on slider values.

    Args:
        df (DataFrame): The DataFrame to be filtered.
        sliders (dict): Dictionary of sliders with their min and max values.
        change (optional): Placeholder parameter for future use.

    Returns:
        DataFrame: The filtered DataFrame.
    """
    filtered_df = df

    for column, slider in sliders.items():
        if column in df.keys():
            min_value, max_value = slider
            filtered_df = filtered_df[
                (filtered_df[column] >= min_value) & (filtered_df[column] <= max_value)
            ]
    return filtered_df


def plot_scatter(
    filtered_df, exp_filtered_df, show_experimental_checkbox, show=True, change=None
):
    """
    Create and display a scatter plot of UEL vs YS.

    Args:
        filtered_df (DataFrame): The filtered DataFrame for simulated data.
        exp_filtered_df (DataFrame): The filtered DataFrame for experimental data.
        show_experimental_checkbox (bool): Flag to show experimental data.
        show (bool): Flag to display the plot.
        change (optional): Placeholder parameter for future use.

    Returns:
        Figure: The generated figure.
    """
    # Scatter plot of the filtered data (simulated data)
    fig = plt.figure(figsize=(10, 6))

    plt.scatter(
        filtered_df["YS"],
        filtered_df["UEL"],
        color="blue",
        alpha=0.5,
        label="Filtered Data",
    )

    # If the checkbox is checked, add experimental data
    if show_experimental_checkbox:
        # Plot the experimental data above the simulated data
        plt.scatter(
            exp_filtered_df["YS"],
            exp_filtered_df["UEL"],
            color="red",
            alpha=0.7,
            label="Experimental Data",
        )

    plt.title("Scatter Plot of UEL vs YS")
    plt.xlabel("YS")
    plt.ylabel("UEL")
    plt.grid(True)
    plt.legend(loc="upper right")
    if show:
        st.pyplot(fig)
    return plt.gcf()


def plot_pareto_and_content_variations(
    filtered_df,
    exp_filtered_df,
    show_experimental_checkbox,
    dropdown,
    show=True,
    change=None,
):
    """
    Create and display plots for Pareto front and chemical content variations.

    Args:
        filtered_df (DataFrame): The filtered DataFrame for simulated data.
        exp_filtered_df (DataFrame): The filtered DataFrame for experimental data.
        show_experimental_checkbox (bool): Flag to show experimental data.
        dropdown (str): Selected column for color-coding.
        show (bool): Flag to display the plot.
        change (optional): Placeholder parameter for future use.

    Returns:
        list: List of generated figures.
    """
    stored_figures = []

    # Calculate Pareto points for the filtered data
    pareto_df = pareto_efficient_points_new(filtered_df)
    color_col = dropdown

    if color_col == "Diff. in chem. comp. to next Exp. point":
        pareto_df = calculate_chem_comp_difference_to_next_exp_point(
            exp_filtered_df, pareto_df
        )
    elif color_col == "Rel Error UEL/YS":
        pareto_df = calculate_rel_error(pareto_df)

    # Set font size globally
    plt.rcParams.update(
        {
            "font.size": 14,  # General font size
            "axes.titlesize": 16,  # Title font size
            "axes.labelsize": 14,  # Axis label font size
            "xtick.labelsize": 12,  # X-tick font size
            "ytick.labelsize": 12,  # Y-tick font size
        }
    )

    # Create the figure with two or three subplots
    if show_experimental_checkbox:
        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, sharex=True, figsize=(10, 8), layout="constrained"
        )
    else:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, sharex=True, figsize=(10, 8), layout="constrained"
        )

    if not pareto_df.empty:
        pareto_sorted = pareto_df.sort_values(by="YS")
        # Pareto front plot
        sc = ax1.scatter(
            pareto_sorted["YS"],
            pareto_sorted["UEL"],
            c=pareto_sorted[color_col],
            cmap="viridis",
            alpha=0.7,
        )

        ax1.set_title("Pareto Front: UEL vs YS", x=1.3, y=0.5, pad=-14)
        ax1.set_ylabel("UEL")
        ax1.grid(True)
        fig.colorbar(sc, ax=ax1, label=color_col, location="left", aspect=5, pad=-0.05)

        # Chemical content variations along the Pareto front
        chemicals = ["C", "Si", "Mn", "Cr", "Mo", "V", "Al"]
        for chemical in chemicals:
            ax2.plot(
                pareto_sorted["YS"],
                pareto_sorted[chemical],
                label=f"{chemical}",
                linewidth=1.5,
            )

        if not show_experimental_checkbox:
            ax2.set_xlabel("YS")
        ax2.set_title(
            "Chemical Content Variations \n Along the Pareto Front",
            x=1.3,
            y=0.5,
            pad=-14,
        )
        ax2.set_ylabel("Content")
        ax2.grid(True)
        ax2.legend(bbox_to_anchor=(-0.2, 1))

    # If the checkbox is activated, show experimental data in the main plot
    if show_experimental_checkbox:
        # Show experimental data in the main plot
        ax1.scatter(
            exp_filtered_df["YS"],
            exp_filtered_df["UEL"],
            color="red",
            alpha=0.7,
            label="Experimental Data",
        )

    # If the checkbox is activated, show experimental chemical composition separately
    if show_experimental_checkbox:
        # Plot experimental chemical composition along the YS axis
        chemicals = ["C", "Si", "Mn", "Cr", "Mo", "V", "Al"]
        for chemical in chemicals:
            ax3.plot(
                exp_filtered_df["YS"],
                exp_filtered_df[chemical],
                label=f"{chemical}",
                linestyle="--",
            )

        ax3.set_title("Experimental Chemical \n Composition", x=1.3, y=0.5, pad=-14)
        ax3.set_xlabel("YS")
        ax3.set_ylabel("Content")
        ax3.grid(True)
        ax3.legend(bbox_to_anchor=(-0.2, 1))

    stored_figures.append(plt.gcf())
    if show:
        st.pyplot(fig)
    return stored_figures


def plot_pareto_and_variations(
    filtered_df,
    exp_filtered_df,
    show_experimental_checkbox,
    dropdown,
    show=True,
    change=None,
):
    """
    Create and display plots for Pareto front and variations.

    Args:
        filtered_df (DataFrame): The filtered DataFrame for simulated data.
        exp_filtered_df (DataFrame): The filtered DataFrame for experimental data.
        show_experimental_checkbox (bool): Flag to show experimental data.
        dropdown (str): Selected column for color-coding.
        show (bool): Flag to display the plot.
        change (optional): Placeholder parameter for future use.

    Returns:
        Figure: The generated figure.
    """
    # Calculate Pareto points only for the filtered data
    pareto_df = pareto_efficient_points(filtered_df)
    color_col = dropdown

    if color_col == "Diff. in chem. comp. to next Exp. point":
        pareto_df = calculate_chem_comp_difference_to_next_exp_point(
            exp_filtered_df, pareto_df
        )
    elif color_col == "Rel Error UEL/YS":
        pareto_df = calculate_rel_error(pareto_df)

    fig = plt.figure(figsize=(12, 8))
    plt.scatter(
        filtered_df["YS"],
        filtered_df["UEL"],
        color="lightgray",
        alpha=0.1,
        label="All Points",
    )
    plt.xlabel("YS")
    plt.ylabel("UEL")

    pareto_sorted = pareto_df.sort_values(by="YS")
    plt.plot(
        pareto_sorted["YS"],
        pareto_sorted["UEL"],
        color="black",
        linewidth=1,
        alpha=0.7,
        label="Pareto Line",
    )

    pareto_plus_sigma = pareto_df.copy()
    pareto_plus_sigma["YS"] += pareto_plus_sigma["Sigma YS"]
    pareto_plus_sigma["UEL"] += pareto_plus_sigma["Sigma UEL"]
    if not pareto_plus_sigma.empty:
        plus_sigma_sorted = pareto_plus_sigma.sort_values(by="YS")
        plt.plot(
            plus_sigma_sorted["YS"],
            plus_sigma_sorted["UEL"],
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label="Pareto + Sigma",
        )
    pareto_minus_sigma = pareto_df.copy()
    pareto_minus_sigma["YS"] -= pareto_minus_sigma["Sigma YS"]
    pareto_minus_sigma["UEL"] -= pareto_minus_sigma["Sigma UEL"]
    if not pareto_minus_sigma.empty:
        minus_sigma_sorted = pareto_minus_sigma.sort_values(by="YS")
        plt.plot(
            minus_sigma_sorted["YS"],
            minus_sigma_sorted["UEL"],
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label="Pareto - Sigma",
        )
    paretoForbar = pd.concat(
        [pareto_sorted, plus_sigma_sorted, minus_sigma_sorted],
        ignore_index=True,
        sort=True,
    )
    sc = plt.scatter(
        paretoForbar["YS"],
        paretoForbar["UEL"],
        c=paretoForbar[color_col],
        cmap="viridis",
        alpha=0.7,
        label="Pareto Line",
    )
    plt.colorbar(sc, label=color_col, aspect=10)
    if show_experimental_checkbox:
        plt.scatter(
            exp_filtered_df["YS"],
            exp_filtered_df["UEL"],
            color="red",
            alpha=0.7,
            label="Experimental Data",
        )
    if show:
        st.pyplot(fig)
    return plt.gcf()
