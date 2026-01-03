"""
Visualization Module
Creates basic, advanced, and interactive visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('ggplot')
sns.set_palette("husl")


def create_line_plot(df, x_col, y_col, title="Line Plot", save_path=None):
    """
    Create a line plot
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    x_col : str
        Column for x-axis
    y_col : str
        Column for y-axis
    title : str
        Plot title
    save_path : str
        Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df[x_col], df[y_col], marker='o', linewidth=2, markersize=4)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Line plot saved to {save_path}")
    
    plt.close()


def create_bar_chart(df, x_col, y_col=None, title="Bar Chart", save_path=None):
    """
    Create a bar chart
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    x_col : str
        Column for x-axis
    y_col : str
        Column for y-axis (None for count)
    title : str
        Plot title
    save_path : str
        Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    if y_col is None:
        data = df[x_col].value_counts().head(10)
        plt.bar(range(len(data)), data.values, color='steelblue')
        plt.xticks(range(len(data)), data.index, rotation=45, ha='right')
    else:
        data = df.groupby(x_col)[y_col].mean().sort_values(ascending=False).head(10)
        plt.bar(range(len(data)), data.values, color='steelblue')
        plt.xticks(range(len(data)), data.index, rotation=45, ha='right')
        plt.ylabel(y_col, fontsize=12)
    
    plt.xlabel(x_col, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Bar chart saved to {save_path}")
    
    plt.close()


def create_histogram(df, column, bins=30, title="Histogram", save_path=None):
    """
    Create a histogram
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    column : str
        Column to plot
    bins : int
        Number of bins
    title : str
        Plot title
    save_path : str
        Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.hist(df[column].dropna(), bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Histogram saved to {save_path}")
    
    plt.close()


def create_pair_plot(df, columns=None, save_path=None):
    """
    Create a pair plot
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    columns : list
        Columns to include
    save_path : str
        Path to save the plot
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()[:5]  # Limit to 5 columns
    
    print(f"Creating pair plot for columns: {columns}")
    
    sns.pairplot(df[columns], diag_kind='kde', plot_kws={'alpha': 0.6})
    plt.suptitle('Pair Plot', fontsize=16, fontweight='bold', y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Pair plot saved to {save_path}")
    
    plt.close()


def create_heatmap(df, columns=None, title="Correlation Heatmap", save_path=None):
    """
    Create a correlation heatmap
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    columns : list
        Columns to include
    title : str
        Plot title
    save_path : str
        Path to save the plot
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    plt.figure(figsize=(12, 10))
    corr_matrix = df[columns].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {save_path}")
    
    plt.close()


def create_violin_plot(df, x_col, y_col, title="Violin Plot", save_path=None):
    """
    Create a violin plot
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    x_col : str
        Categorical column
    y_col : str
        Numeric column
    title : str
        Plot title
    save_path : str
        Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df, x=x_col, y=y_col)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Violin plot saved to {save_path}")
    
    plt.close()


def create_probability_distribution(df, column, save_path=None):
    """
    Visualize probability distribution for a numeric feature
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    column : str
        Column to visualize
    save_path : str
        Path to save the plot
    """
    data = df[column].dropna()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram with density curve
    axes[0].hist(data, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].set_xlabel(column, fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title(f'Probability Distribution: {column}', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(data, dist="norm", plot=axes[1])
    axes[1].set_title(f'Q-Q Plot: {column}', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Probability distribution plot saved to {save_path}")
    
    plt.close()


def create_interactive_scatter(df, x_col, y_col, color_col=None, size_col=None, 
                               title="Interactive Scatter Plot", save_path=None):
    """
    Create an interactive scatter plot using Plotly
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    x_col : str
        Column for x-axis
    y_col : str
        Column for y-axis
    color_col : str
        Column for color coding
    size_col : str
        Column for size coding
    title : str
        Plot title
    save_path : str
        Path to save the HTML file
    """
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col,
                     hover_data=df.columns.tolist(), title=title,
                     labels={x_col: x_col.replace('_', ' ').title(),
                            y_col: y_col.replace('_', ' ').title()})
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        hovermode='closest',
        width=1000,
        height=600
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"Interactive scatter plot saved to {save_path}")
    
    return fig


def create_interactive_dashboard(df, save_path=None):
    """
    Create an interactive dashboard using Plotly
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    save_path : str
        Path to save the HTML file
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Price Distribution', 'Price by City', 'Price by Property Type', 'Price vs Size'),
        specs=[[{"type": "histogram"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Price distribution
    if 'price' in df.columns:
        fig.add_trace(
            go.Histogram(x=df['price'].dropna(), name='Price Distribution', nbinsx=30),
            row=1, col=1
        )
    
    # Price by city
    if 'city' in df.columns and 'price' in df.columns:
        city_price = df.groupby('city')['price'].mean().sort_values(ascending=False).head(10)
        fig.add_trace(
            go.Bar(x=city_price.index, y=city_price.values, name='Avg Price by City'),
            row=1, col=2
        )
    
    # Price by property type
    if 'type' in df.columns and 'price' in df.columns:
        type_price = df.groupby('type')['price'].mean().sort_values(ascending=False).head(10)
        fig.add_trace(
            go.Bar(x=type_price.index, y=type_price.values, name='Avg Price by Type'),
            row=2, col=1
        )
    
    # Price vs Size (if size_numeric exists)
    if 'size_numeric' in df.columns and 'price' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['size_numeric'], y=df['price'], mode='markers',
                      name='Price vs Size', marker=dict(size=5, opacity=0.6)),
            row=2, col=2
        )
    
    fig.update_layout(
        title_text="Real Estate Data Dashboard",
        height=800,
        showlegend=False
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"Interactive dashboard saved to {save_path}")
    
    return fig


if __name__ == "__main__":
    # Example usage
    data_path = "../data/real_estate_dataset.csv"
    df = pd.read_csv(data_path)
    
    # Basic visualizations
    if 'price' in df.columns:
        create_histogram(df, 'price', title="Price Distribution", save_path="../outputs/price_histogram.png")
    
    if 'city' in df.columns:
        create_bar_chart(df, 'city', title="Properties by City", save_path="../outputs/city_bar.png")
    
    # Advanced visualizations
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:5]
    if len(numeric_cols) >= 2:
        create_heatmap(df, columns=numeric_cols, save_path="../outputs/correlation_heatmap.png")
    
    # Interactive visualization
    if 'price' in df.columns and 'size_numeric' in df.columns:
        create_interactive_scatter(df, 'size_numeric', 'price', color_col='city',
                                  save_path="../outputs/interactive_scatter.html")

