"""
Visualization Module for Resume Scoring Agent
Handles charts, plots, and visual analytics
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import logging
from typing import Dict, List, Tuple, Any, Optional
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ResumeVisualizer:
    """Handles visualization and plotting for resume scoring analysis"""
    
    def __init__(self, color_scheme: str = "viridis"):
        """
        Initialize the visualizer
        
        Args:
            color_scheme: Color scheme for plots
        """
        self.color_scheme = color_scheme
        
        # Configure matplotlib
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        # Color schemes - Updated for better visual appeal
        self.colors = {
            'primary': '#1E3A8A',      # Professional Dark Blue
            'secondary': '#3B82F6',    # Bright Blue  
            'success': '#10B981',      # Emerald Green
            'warning': '#F59E0B',      # Amber Orange
            'background': '#F8FAFC'    # Light Gray Background
        }
    
    def create_similarity_bar_chart(self, 
                                  similarity_scores: Dict[str, float],
                                  top_n: int = 15,
                                  title: str = "Resume Similarity Scores") -> go.Figure:
        """
        Create a horizontal bar chart of similarity scores
        
        Args:
            similarity_scores: Dictionary mapping resume names to scores
            top_n: Number of top resumes to display
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        # Sort and get top N
        sorted_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        resumes, scores = zip(*sorted_scores) if sorted_scores else ([], [])
        
        # Create color scale based on scores
        colors = [self._get_score_color(score) for score in scores]
        
        fig = go.Figure(data=[
            go.Bar(
                y=resumes,
                x=scores,
                orientation='h',
                marker=dict(
                    color=colors,
                    line=dict(color='white', width=1)
                ),
                text=[f'{score:.3f}' for score in scores],
                textposition='inside',
                textfont=dict(color='white', size=12)
            )
        ])
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            xaxis_title="Similarity Score",
            yaxis_title="Resume",
            template="plotly_white",
            height=max(400, len(resumes) * 30),
            margin=dict(l=150, r=50, t=80, b=50)
        )
        
        return fig
    
    def create_score_distribution_plot(self, 
                                     similarity_scores: Dict[str, float],
                                     title: str = "Similarity Score Distribution") -> go.Figure:
        """
        Create a distribution plot of similarity scores
        
        Args:
            similarity_scores: Dictionary mapping resume names to scores
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        scores = list(similarity_scores.values())
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Histogram', 'Box Plot'),
            vertical_spacing=0.15
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=scores,
                nbinsx=20,
                name="Distribution",
                marker=dict(color=self.colors['primary'], opacity=0.7)
            ),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(
                x=scores,
                name="Score Range",
                marker=dict(color=self.colors['secondary'])
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            template="plotly_white",
            height=600,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Similarity Score", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        
        return fig
    
    def create_keyword_match_chart(self, 
                                 keyword_matches: Dict[str, Any],
                                 top_n: int = 15) -> go.Figure:
        """
        Create a chart showing keyword matches
        
        Args:
            keyword_matches: Keyword match analysis from scorer
            top_n: Number of top keywords to display
            
        Returns:
            Plotly figure object
        """
        detailed_matches = keyword_matches.get('detailed_matches', {})
        
        # Get top keywords by average TF-IDF score
        top_keywords = list(detailed_matches.keys())[:top_n]
        
        job_scores = [detailed_matches[kw]['job_tfidf'] for kw in top_keywords]
        resume_scores = [detailed_matches[kw]['resume_tfidf'] for kw in top_keywords]
        
        fig = go.Figure()
        
        # Job description scores
        fig.add_trace(go.Bar(
            name='Job Description',
            y=top_keywords,
            x=job_scores,
            orientation='h',
            marker=dict(color=self.colors['primary']),
            offsetgroup=1
        ))
        
        # Resume scores
        fig.add_trace(go.Bar(
            name='Resume',
            y=top_keywords,
            x=resume_scores,
            orientation='h',
            marker=dict(color=self.colors['secondary']),
            offsetgroup=2
        ))
        
        fig.update_layout(
            title=f"Top Keyword Matches - {keyword_matches.get('resume_name', 'Resume')}",
            xaxis_title="TF-IDF Score",
            yaxis_title="Keywords",
            barmode='group',
            template="plotly_white",
            height=max(400, len(top_keywords) * 25),
            margin=dict(l=120, r=50, t=80, b=50)
        )
        
        return fig
    
    def create_skill_overlap_radar(self, 
                                 skill_analysis: Dict[str, Any]) -> go.Figure:
        """
        Create a radar chart showing skill overlap by category
        
        Args:
            skill_analysis: Skill overlap analysis from scorer
            
        Returns:
            Plotly figure object
        """
        category_analysis = skill_analysis.get('category_analysis', {})
        
        categories = list(category_analysis.keys())
        overlap_percentages = [
            category_analysis[cat]['overlap_percentage'] 
            for cat in categories
        ]
        
        # Add first category at the end to close the radar chart
        categories.append(categories[0])
        overlap_percentages.append(overlap_percentages[0])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=overlap_percentages,
            theta=categories,
            fill='toself',
            name='Skill Overlap %',
            line=dict(color=self.colors['primary']),
            fillcolor='rgba(46, 134, 171, 0.3)'  # Convert to proper RGBA format
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title=f"Skill Overlap Analysis - {skill_analysis.get('resume_name', 'Resume')}",
            template="plotly_white"
        )
        
        return fig
    
    def create_wordcloud(self, 
                        text: str, 
                        title: str = "Word Cloud",
                        max_words: int = 100,
                        width: int = 800,
                        height: int = 400) -> plt.Figure:
        """
        Create a word cloud from text
        
        Args:
            text: Text to generate word cloud from
            title: Title for the word cloud
            max_words: Maximum number of words
            width: Width of the word cloud
            height: Height of the word cloud
            
        Returns:
            Matplotlib figure object
        """
        if not text.strip():
            # Create empty figure if no text
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No text available', 
                   horizontalalignment='center', 
                   verticalalignment='center',
                   transform=ax.transAxes,
                   fontsize=16)
            ax.set_title(title)
            ax.axis('off')
            return fig
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=width,
            height=height,
            max_words=max_words,
            background_color='white',
            colormap=self.color_scheme,
            relative_scaling=0.5,
            random_state=42
        ).generate(text)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, pad=20)
        
        return fig
    
    def create_comparison_matrix(self, 
                               top_resumes: List[Tuple[str, float]],
                               keyword_data: Dict[str, Dict],
                               top_n: int = 10) -> go.Figure:
        """
        Create a heatmap matrix comparing top resumes across key metrics
        
        Args:
            top_resumes: List of (resume_name, score) tuples
            keyword_data: Dictionary of keyword analysis for each resume
            top_n: Number of top resumes to include
            
        Returns:
            Plotly figure object
        """
        if not top_resumes or not keyword_data:
            # Return empty figure
            fig = go.Figure()
            fig.update_layout(title="No data available for comparison matrix")
            return fig
        
        # Prepare data
        resume_names = [name for name, _ in top_resumes[:top_n]]
        metrics = ['Similarity Score', 'Keyword Matches', 'Match Percentage']
        
        matrix_data = []
        for resume_name, score in top_resumes[:top_n]:
            kw_data = keyword_data.get(resume_name, {})
            row = [
                score * 100,  # Convert to percentage
                kw_data.get('total_matches', 0),
                kw_data.get('match_percentage', 0)
            ]
            matrix_data.append(row)
        
        # Normalize data for better visualization
        matrix_data = np.array(matrix_data)
        normalized_data = np.zeros_like(matrix_data)
        
        for i in range(matrix_data.shape[1]):
            col = matrix_data[:, i]
            if col.max() > 0:
                normalized_data[:, i] = (col / col.max()) * 100
        
        fig = go.Figure(data=go.Heatmap(
            z=normalized_data,
            x=metrics,
            y=resume_names,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Normalized Score")
        ))
        
        # Add text annotations
        for i, resume in enumerate(resume_names):
            for j, metric in enumerate(metrics):
                value = matrix_data[i, j]
                fig.add_annotation(
                    x=j, y=i,
                    text=f'{value:.1f}',
                    showarrow=False,
                    font=dict(color='white', size=10)
                )
        
        fig.update_layout(
            title="Resume Comparison Matrix",
            template="plotly_white",
            height=max(400, len(resume_names) * 40)
        )
        
        return fig
    
    def create_summary_dashboard(self, 
                               scorer_results: Dict[str, Any]) -> go.Figure:
        """
        Create a comprehensive dashboard with multiple metrics
        
        Args:
            scorer_results: Complete results from scorer
            
        Returns:
            Plotly figure with subplots
        """
        # Extract data
        similarity_scores = scorer_results.get('similarity_scores', {})
        top_resumes = scorer_results.get('top_performers', [])
        
        if not similarity_scores:
            fig = go.Figure()
            fig.update_layout(title="No data available for dashboard")
            return fig
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Top 10 Similarity Scores',
                'Score Distribution',
                'Performance Tiers',
                'Summary Statistics'
            ),
            specs=[[{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "pie"}, {"type": "table"}]]
        )
        
        # Top 10 bar chart
        top_10 = dict(list(similarity_scores.items())[:10])
        names, scores = zip(*sorted(top_10.items(), key=lambda x: x[1], reverse=True))
        
        fig.add_trace(
            go.Bar(x=list(scores), y=list(names), orientation='h', name="Scores"),
            row=1, col=1
        )
        
        # Score distribution
        all_scores = list(similarity_scores.values())
        fig.add_trace(
            go.Histogram(x=all_scores, name="Distribution"),
            row=1, col=2
        )
        
        # Performance tiers pie chart
        tier_counts = self._categorize_scores(all_scores)
        fig.add_trace(
            go.Pie(labels=list(tier_counts.keys()), values=list(tier_counts.values())),
            row=2, col=1
        )
        
        # Summary table
        stats = self._calculate_summary_stats(all_scores)
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value']),
                cells=dict(values=[list(stats.keys()), list(stats.values())])
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Resume Scoring Dashboard",
            template="plotly_white",
            height=800,
            showlegend=False
        )
        
        return fig
    
    def _get_score_color(self, score: float) -> str:
        """Get color based on score value (works with both decimal and percentage scores)"""
        # Handle both percentage (0-100) and decimal (0-1) scores
        if score > 1:  # Percentage score
            if score >= 60:
                return self.colors['success']    # Emerald for excellent
            elif score >= 40:
                return self.colors['primary']    # Dark blue for good
            elif score >= 20:
                return self.colors['secondary']  # Blue for average
            else:
                return self.colors['warning']    # Amber for below average
        else:  # Decimal score
            if score >= 0.6:
                return self.colors['success']
            elif score >= 0.4:
                return self.colors['primary']
            elif score >= 0.2:
                return self.colors['secondary']
            else:
                return self.colors['warning']
    
    def _categorize_scores(self, scores: List[float]) -> Dict[str, int]:
        """Categorize scores into performance tiers"""
        tiers = {
            'Excellent (≥0.8)': 0,
            'Good (0.6-0.8)': 0,
            'Average (0.4-0.6)': 0,
            'Below Average (<0.4)': 0
        }
        
        for score in scores:
            if score >= 0.8:
                tiers['Excellent (≥0.8)'] += 1
            elif score >= 0.6:
                tiers['Good (0.6-0.8)'] += 1
            elif score >= 0.4:
                tiers['Average (0.4-0.6)'] += 1
            else:
                tiers['Below Average (<0.4)'] += 1
        
        return tiers
    
    def _calculate_summary_stats(self, scores: List[float]) -> Dict[str, str]:
        """Calculate summary statistics"""
        if not scores:
            return {}
        
        return {
            'Total Resumes': str(len(scores)),
            'Average Score': f'{np.mean(scores):.3f}',
            'Highest Score': f'{np.max(scores):.3f}',
            'Lowest Score': f'{np.min(scores):.3f}',
            'Standard Deviation': f'{np.std(scores):.3f}',
            'Scores > 0.5': str(sum(1 for s in scores if s > 0.5))
        }
    
    def export_figure_as_html(self, fig: go.Figure, filename: str) -> str:
        """
        Export Plotly figure as HTML
        
        Args:
            fig: Plotly figure
            filename: Output filename
            
        Returns:
            HTML string
        """
        return fig.to_html(include_plotlyjs=True, filename=filename)
    
    def save_matplotlib_figure(self, fig: plt.Figure, filename: str, dpi: int = 300):
        """
        Save matplotlib figure
        
        Args:
            fig: Matplotlib figure
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        fig.savefig(filename, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)


def create_visualization_report(scorer_results: Dict[str, Any], 
                              output_dir: str = "visualizations") -> Dict[str, str]:
    """
    Create a complete set of visualizations
    
    Args:
        scorer_results: Results from ResumeScorer
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary mapping visualization names to file paths
    """
    import os
    
    visualizer = ResumeVisualizer()
    file_paths = {}
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Similarity scores bar chart
        if 'similarity_scores' in scorer_results:
            fig = visualizer.create_similarity_bar_chart(scorer_results['similarity_scores'])
            filepath = os.path.join(output_dir, "similarity_scores.html")
            fig.write_html(filepath)
            file_paths['similarity_scores'] = filepath
        
        # Score distribution
        if 'similarity_scores' in scorer_results:
            fig = visualizer.create_score_distribution_plot(scorer_results['similarity_scores'])
            filepath = os.path.join(output_dir, "score_distribution.html")
            fig.write_html(filepath)
            file_paths['score_distribution'] = filepath
        
        # Dashboard
        fig = visualizer.create_summary_dashboard(scorer_results)
        filepath = os.path.join(output_dir, "dashboard.html")
        fig.write_html(filepath)
        file_paths['dashboard'] = filepath
        
        logger.info(f"Created {len(file_paths)} visualizations in {output_dir}")
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")
    
    return file_paths


# Example usage
if __name__ == "__main__":
    # Test the visualizer
    sample_scores = {
        'resume1.pdf': 0.85,
        'resume2.pdf': 0.72,
        'resume3.pdf': 0.68,
        'resume4.pdf': 0.45,
        'resume5.pdf': 0.32
    }
    
    visualizer = ResumeVisualizer()
    
    # Test bar chart
    fig = visualizer.create_similarity_bar_chart(sample_scores)
    print("Bar chart created successfully")
    
    # Test distribution plot
    fig = visualizer.create_score_distribution_plot(sample_scores)
    print("Distribution plot created successfully")
