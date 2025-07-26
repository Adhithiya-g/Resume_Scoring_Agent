import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
import zipfile
import tempfile
import os
from typing import Dict, List, Tuple, Any, Optional
import logging

from utils.extractor import TextExtractor, extract_multiple_resumes
from utils.preprocessor import TextPreprocessor
from utils.scorer import ResumeScorer, score_resumes_pipeline
from utils.visualizer import ResumeVisualizer

def ensure_nltk_data():
    """Ensure NLTK data is available"""
    import nltk
    import os
    
    # Set NLTK data path for deployment environments
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir, exist_ok=True)
    
    nltk.data.path.append(nltk_data_dir)
    
    required_data = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'), 
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
        ('corpora/omw-1.4', 'omw-1.4')
    ]
    
    for path, name in required_data:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                print(f"Downloading NLTK data: {name}")
                nltk.download(name, quiet=True)
            except Exception as e:
                print(f"Warning: Could not download {name}: {e}")
                continue  # Continue if download fails

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


st.set_page_config(
    page_title="Resume Score Analyser Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #2E86AB;
    }
    
    .score-excellent {
        color: #28a745;
        font-weight: bold;
    }
    
    .score-good {
        color: #17a2b8;
        font-weight: bold;
    }
    
    .score-average {
        color: #ffc107;
        font-weight: bold;
    }
    
    .score-poor {
        color: #dc3545;
        font-weight: bold;
    }
    
    .stTab {
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)


class ResumeAgent:
    """Main Resume Scoring Agent class"""
    
    def __init__(self):
        """Initialize the agent with default configurations"""
        self.text_extractor = TextExtractor()
        self.preprocessor = None
        self.scorer = None
        self.visualizer = ResumeVisualizer()
        
        # Initialize session state
        if 'job_description' not in st.session_state:
            st.session_state.job_description = ""
        if 'resume_texts' not in st.session_state:
            st.session_state.resume_texts = {}
        if 'scoring_results' not in st.session_state:
            st.session_state.scoring_results = None
        if 'preprocessor_config' not in st.session_state:
            st.session_state.preprocessor_config = {
                'use_spacy': False, 
                'remove_stopwords': True,
                'use_lemmatization': True,
                'min_word_length': 2
            }
    
    def render_header(self):
        """Render the application header"""
        st.markdown('<h1 class="main-header">📄 Resume Scoring Analyser Agent</h1>', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <p style="font-size: 1.2rem; color: #666;">
                Evaluate and rank resumes against job descriptions using TF-IDF vectorization and Cosine Similarity
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with configuration options"""
        st.sidebar.title("⚙️ Configuration")
        
        # Preprocessing options
        st.sidebar.subheader("Text Preprocessing")
        st.session_state.preprocessor_config['remove_stopwords'] = st.sidebar.checkbox(
            "Remove Stopwords", 
            value=st.session_state.preprocessor_config['remove_stopwords']
        )
        st.session_state.preprocessor_config['use_lemmatization'] = st.sidebar.checkbox(
            "Use Lemmatization", 
            value=st.session_state.preprocessor_config['use_lemmatization']
        )
        st.session_state.preprocessor_config['min_word_length'] = st.sidebar.slider(
            "Minimum Word Length", 
            1, 5, 
            st.session_state.preprocessor_config['min_word_length']
        )
        
        # TF-IDF options
        st.sidebar.subheader("TF-IDF Settings")
        max_features = st.sidebar.slider("Max Features", 1000, 10000, 5000)
        min_df = st.sidebar.slider("Min Document Frequency", 1, 5, 1)
        max_df = st.sidebar.slider("Max Document Frequency", 0.8, 1.0, 0.95, 0.05)
        
        # Scoring options
        st.sidebar.subheader("Scoring Options")
        similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.0, 0.05)
        top_n_display = st.sidebar.slider("Top N Resumes to Display", 5, 50, 15)
        
        return {
            'max_features': max_features,
            'min_df': min_df,
            'max_df': max_df,
            'similarity_threshold': similarity_threshold,
            'top_n_display': top_n_display
        }
    
    def handle_job_description_input(self):
        """Handle job description input"""
        st.markdown('<h2 class="sub-header">📋 Job Description</h2>', unsafe_allow_html=True)
        
        input_method = st.radio(
            "Choose input method:",
            ["Paste Text", "Upload File"],
            horizontal=True
        )
        
        if input_method == "Paste Text":
            job_description = st.text_area(
                "Paste the job description here:",
                height=200,
                placeholder="Enter the job description that you want to match resumes against..."
            )
            if job_description.strip():
                st.session_state.job_description = job_description
        
        else:  # Upload File
            uploaded_file = st.file_uploader(
                "Upload job description file",
                type=['txt', 'pdf', 'docx'],
                help="Upload a text, PDF, or Word document containing the job description"
            )
            
            if uploaded_file is not None:
                try:
                    extracted_text = self.text_extractor.extract_text_from_uploaded_file(uploaded_file)
                    if extracted_text:
                        st.session_state.job_description = extracted_text
                        st.success(f"✅ Job description extracted from {uploaded_file.name}")
                        with st.expander("Preview extracted text"):
                            st.text(extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text)
                    else:
                        st.error("❌ Could not extract text from the uploaded file")
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
        
        # Display current job description
        if st.session_state.job_description:
            with st.expander("Current Job Description Preview"):
                st.text(st.session_state.job_description[:500] + "..." if len(st.session_state.job_description) > 500 else st.session_state.job_description)
    
    def handle_resume_uploads(self):
        """Handle resume file uploads"""
        st.markdown('<h2 class="sub-header">📁 Upload Resumes</h2>', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Upload resume files",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Upload multiple resume files in PDF, DOCX, or TXT format"
        )
        
        if uploaded_files:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            resume_texts = {}
            successful_extractions = 0
            
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    status_text.text(f"Processing {uploaded_file.name}...")
                    extracted_text = self.text_extractor.extract_text_from_uploaded_file(uploaded_file)
                    
                    if extracted_text:
                        resume_texts[uploaded_file.name] = extracted_text
                        successful_extractions += 1
                    else:
                        st.warning(f"⚠️ Could not extract text from {uploaded_file.name}")
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
            
            if resume_texts:
                st.session_state.resume_texts = resume_texts
                st.success(f"✅ Successfully processed {successful_extractions} out of {len(uploaded_files)} resumes")
                
                # Display summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Resumes", len(resume_texts))
                with col2:
                    total_words = sum(len(text.split()) for text in resume_texts.values())
                    st.metric("Total Words", f"{total_words:,}")
                with col3:
                    avg_words = total_words / len(resume_texts) if resume_texts else 0
                    st.metric("Avg Words/Resume", f"{avg_words:.0f}")
            
            status_text.empty()
            progress_bar.empty()
        
        # Display current resumes
        if st.session_state.resume_texts:
            with st.expander(f"Uploaded Resumes ({len(st.session_state.resume_texts)})"):
                for name, text in st.session_state.resume_texts.items():
                    st.write(f"**{name}**: {len(text.split())} words")
    
    def run_scoring_analysis(self, config: Dict[str, Any]):
        """Run the scoring analysis"""
        if not st.session_state.job_description:
            st.error("❌ Please provide a job description first")
            return False
        
        if not st.session_state.resume_texts:
            st.error("❌ Please upload resume files first")
            return False
        
        try:
            # Initialize preprocessor
            self.preprocessor = TextPreprocessor(**st.session_state.preprocessor_config)
            
            # Preprocess texts
            with st.spinner("Preprocessing texts..."):
                # Preprocess job description
                processed_job_desc = self.preprocessor.preprocess_text(st.session_state.job_description)
                
                # Preprocess resumes
                processed_resumes = {}
                for name, text in st.session_state.resume_texts.items():
                    processed_resumes[name] = self.preprocessor.preprocess_text(text)
            
            # Configure scorer
            scorer_config = {
                'max_features': config['max_features'],
                'min_df': config['min_df'],
                'max_df': config['max_df'],
                'ngram_range': (1, 2)
            }
            
            # Run scoring pipeline
            with st.spinner("Computing similarity scores..."):
                self.scorer = score_resumes_pipeline(
                    processed_job_desc, 
                    processed_resumes, 
                    scorer_config
                )
            
            # Generate comprehensive report
            with st.spinner("Generating analysis report..."):
                report = self.scorer.generate_comprehensive_report(
                    top_n_resumes=config['top_n_display'],
                    include_keyword_analysis=True
                )
            
            st.session_state.scoring_results = {
                'similarity_scores': self.scorer.similarity_scores,
                'ranked_resumes': self.scorer.get_ranked_resumes(config['similarity_threshold']),
                'comprehensive_report': report,
                'scorer': self.scorer
            }
            
            st.success("✅ Analysis completed successfully!")
            return True
        
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            logger.error(f"Scoring analysis error: {str(e)}")
            return False
    
    def display_results(self):
        """Display the scoring results"""
        if not st.session_state.scoring_results:
            st.info("ℹ️ Run the analysis to see results")
            return
        
        results = st.session_state.scoring_results
        similarity_scores = results['similarity_scores']
        ranked_resumes = results['ranked_resumes']
        report = results['comprehensive_report']
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Rankings", "📈 Visualizations", "🔍 Detailed Analysis", 
            "💡 Insights", "📋 Export"
        ])
        
        with tab1:
            self.display_rankings_tab(ranked_resumes, similarity_scores)
        
        with tab2:
            self.display_visualizations_tab(results)
        
        with tab3:
            self.display_detailed_analysis_tab(report)
        
        with tab4:
            self.display_insights_tab(report)
        
        with tab5:
            self.display_export_tab(results)
    
    def display_rankings_tab(self, ranked_resumes: List[Tuple[str, float]], similarity_scores: Dict[str, float]):
        """Display rankings tab"""
        st.markdown('<h3 class="sub-header">🏆 Resume Rankings</h3>', unsafe_allow_html=True)
        
        if not ranked_resumes:
            st.warning("No resumes meet the similarity threshold")
            return
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        scores = list(similarity_scores.values())
        
        with col1:
            st.metric("Total Resumes", len(similarity_scores))
        with col2:
            st.metric("Average Score", f"{np.mean(scores):.3f}")
        with col3:
            st.metric("Highest Score", f"{np.max(scores):.3f}")
        with col4:
            st.metric("Above Threshold", len(ranked_resumes))
        
        # Rankings table
        st.subheader("Ranked Results")
        
        ranking_data = []
        for rank, (resume_name, score) in enumerate(ranked_resumes, 1):
            # Determine score category (score is already a percentage)
            if score >= 60:
                score_class = "score-excellent"
                category = "Excellent"
            elif score >= 40:
                score_class = "score-good"
                category = "Good"
            elif score >= 20:
                score_class = "score-average"
                category = "Average"
            else:
                score_class = "score-poor"
                category = "Below Average"
            
            ranking_data.append({
                'Rank': rank,
                'Resume': resume_name,
                'Score': f"{score:.3f}",
                'Percentage': f"{score:.1f}%",  # Score is already a percentage
                'Category': category
            })
        
        # Display as dataframe
        df = pd.DataFrame(ranking_data)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )
        
        # Top performers highlight
        if len(ranked_resumes) >= 3:
            st.subheader("🥇 Top 3 Performers")
            cols = st.columns(3)
            
            for i, (col, (resume_name, score)) in enumerate(zip(cols, ranked_resumes[:3])):
                with col:
                    medal = ["🥇", "🥈", "🥉"][i]
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>{medal} #{i+1}</h4>
                        <p><strong>{resume_name}</strong></p>
                        <p>Score: <span class="score-excellent">{score:.3f}</span></p>
                    </div>
                    """, unsafe_allow_html=True)
    
    def display_visualizations_tab(self, results: Dict[str, Any]):
        """Display visualizations tab"""
        st.markdown('<h3 class="sub-header">📈 Visual Analytics</h3>', unsafe_allow_html=True)
        
        similarity_scores = results['similarity_scores']
        
        # Similarity scores bar chart
        st.subheader("Similarity Scores")
        fig = self.visualizer.create_similarity_bar_chart(similarity_scores)
        st.plotly_chart(fig, use_container_width=True)
        
        # Score distribution
        st.subheader("Score Distribution")
        fig = self.visualizer.create_score_distribution_plot(similarity_scores)
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary dashboard
        st.subheader("Dashboard Overview")
        fig = self.visualizer.create_summary_dashboard(results)
        st.plotly_chart(fig, use_container_width=True)
    
    def display_detailed_analysis_tab(self, report: Dict[str, Any]):
        """Display detailed analysis tab"""
        st.markdown('<h3 class="sub-header">🔍 Detailed Analysis</h3>', unsafe_allow_html=True)
        
        detailed_analysis = report.get('detailed_analysis', {})
        
        if not detailed_analysis:
            st.warning("No detailed analysis available")
            return
        
        # Resume selector
        resume_names = list(detailed_analysis.keys())
        selected_resume = st.selectbox("Select resume for detailed analysis:", resume_names)
        
        if selected_resume and selected_resume in detailed_analysis:
            analysis = detailed_analysis[selected_resume]
            
            # Basic metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rank", f"#{analysis['rank']}")
            with col2:
                st.metric("Similarity Score", f"{analysis['similarity_score']:.3f}")
            with col3:
                st.metric("Percentile", f"{analysis['score_percentile']:.1f}%")
            
            # Keyword analysis
            if 'keyword_matches' in analysis:
                keyword_matches = analysis['keyword_matches']
                
                st.subheader("🔤 Keyword Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Matches", keyword_matches['total_matches'])
                    st.metric("Match Percentage", f"{keyword_matches['match_percentage']:.1f}%")
                
                with col2:
                    # Top matching keywords
                    if keyword_matches['top_matches']:
                        st.write("**Top Matching Keywords:**")
                        for i, keyword in enumerate(keyword_matches['top_matches'][:5], 1):
                            st.write(f"{i}. {keyword}")
                
                # Keyword match visualization
                if keyword_matches['detailed_matches']:
                    fig = self.visualizer.create_keyword_match_chart(keyword_matches)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Skill analysis
            if 'skill_overlap' in analysis:
                skill_overlap = analysis['skill_overlap']
                
                st.subheader("🛠️ Skill Analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Overall Skill Overlap", f"{skill_overlap['overall_overlap_percentage']:.1f}%")
                    st.metric("Common Skills", skill_overlap['total_common_skills'])
                
                with col2:
                    st.metric("Job Skills Required", skill_overlap['total_job_skills'])
                    st.metric("Resume Skills Found", skill_overlap['total_resume_skills'])
                
                # Skill overlap radar chart
                fig = self.visualizer.create_skill_overlap_radar(skill_overlap)
                st.plotly_chart(fig, use_container_width=True)
                
                # Missing skills
                if skill_overlap['missing_skills']:
                    st.subheader("❌ Missing Skills")
                    missing_skills_text = ", ".join(skill_overlap['missing_skills'][:10])
                    st.write(missing_skills_text)
                
                # Extra skills
                if skill_overlap['extra_skills']:
                    st.subheader("➕ Additional Skills")
                    extra_skills_text = ", ".join(skill_overlap['extra_skills'][:10])
                    st.write(extra_skills_text)
    
    def display_insights_tab(self, report: Dict[str, Any]):
        """Display insights and recommendations tab"""
        st.markdown('<h3 class="sub-header">💡 Insights & Recommendations</h3>', unsafe_allow_html=True)
        
        summary = report.get('summary', {})
        rankings = report.get('rankings', [])
        
        # Overall insights
        st.subheader("📊 Overall Analysis")
        
        if summary:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **Resume Pool Quality:**
                - Total resumes analyzed: {summary.get('total_resumes', 0)}
                - Average similarity: {summary.get('avg_similarity', 0):.3f}
                - Score range: {summary.get('min_similarity', 0):.3f} - {summary.get('max_similarity', 0):.3f}
                """)
            
            with col2:
                # Quality distribution
                scores = [score for _, score in rankings]
                excellent = sum(1 for s in scores if s >= 0.8)
                good = sum(1 for s in scores if 0.6 <= s < 0.8)
                average = sum(1 for s in scores if 0.4 <= s < 0.6)
                poor = sum(1 for s in scores if s < 0.4)
                
                st.markdown(f"""
                **Quality Distribution:**
                - Excellent (≥0.8): {excellent} resumes
                - Good (0.6-0.8): {good} resumes
                - Average (0.4-0.6): {average} resumes
                - Below Average (<0.4): {poor} resumes
                """)
        
        # Recommendations
        st.subheader("🎯 Recommendations")
        
        if rankings:
            top_score = rankings[0][1] if rankings else 0
            
            if top_score >= 0.8:
                st.success("✅ You have excellent candidates! Focus on the top performers for interviews.")
            elif top_score >= 0.6:
                st.info("ℹ️ Good candidate pool. Consider expanding search or adjusting requirements.")
            elif top_score >= 0.4:
                st.warning("⚠️ Average matches. Consider revising job requirements or expanding search.")
            else:
                st.error("❌ Low match quality. Review job description and sourcing strategy.")
            
            # Specific recommendations
            recommendations = self.generate_recommendations(report)
            for rec in recommendations:
                st.write(f"• {rec}")
    
    def generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations based on analysis"""
        recommendations = []
        
        summary = report.get('summary', {})
        avg_score = summary.get('avg_similarity', 0)
        total_resumes = summary.get('total_resumes', 0)
        
        if avg_score < 0.3:
            recommendations.append("Consider broadening the job description or requirements")
            recommendations.append("Review if the required skills are too specific or rare")
        
        if avg_score > 0.7:
            recommendations.append("Great candidate pool! Focus on top performers for next steps")
            recommendations.append("Consider additional screening criteria to further narrow the field")
        
        if total_resumes < 5:
            recommendations.append("Small candidate pool - consider expanding your sourcing strategy")
        
        if total_resumes > 50:
            recommendations.append("Large candidate pool - consider initial filtering before detailed review")
        
        return recommendations
    
    def display_export_tab(self, results: Dict[str, Any]):
        """Display export options tab"""
        st.markdown('<h3 class="sub-header">📋 Export Results</h3>', unsafe_allow_html=True)
        
        # Export options
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Export Data")
            
            # CSV export
            if st.button("📄 Download CSV Report"):
                csv_data = self.generate_csv_report(results)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name="resume_scoring_report.csv",
                    mime="text/csv"
                )
            
            # JSON export
            if st.button("📋 Download JSON Report"):
                json_data = self.generate_json_report(results)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name="resume_scoring_report.json",
                    mime="application/json"
                )
        
        with col2:
            st.subheader("📈 Export Visualizations")
            
            # HTML visualizations
            if st.button("🌐 Generate HTML Report"):
                html_report = self.generate_html_report(results)
                st.download_button(
                    label="Download HTML Report",
                    data=html_report,
                    file_name="resume_scoring_visualizations.html",
                    mime="text/html"
                )
    
    def generate_csv_report(self, results: Dict[str, Any]) -> str:
        """Generate CSV report"""
        ranked_resumes = results['ranked_resumes']
        
        data = []
        for rank, (resume_name, score) in enumerate(ranked_resumes, 1):
            data.append({
                'Rank': rank,
                'Resume': resume_name,
                'Similarity_Score': score,
                'Score_Percentage': score  # Score is already a percentage
            })
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False)
    
    def generate_json_report(self, results: Dict[str, Any]) -> str:
        """Generate JSON report"""
        import json
        
        # Prepare JSON-serializable data
        report_data = {
            'similarity_scores': results['similarity_scores'],
            'ranked_resumes': results['ranked_resumes'],
            'summary': results['comprehensive_report']['summary'],
            'top_performers': results['comprehensive_report']['top_performers']
        }
        
        return json.dumps(report_data, indent=2)
    
    def generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML report with visualizations"""
        similarity_scores = results['similarity_scores']
        
        # Create visualizations
        bar_chart = self.visualizer.create_similarity_bar_chart(similarity_scores)
        distribution_chart = self.visualizer.create_score_distribution_plot(similarity_scores)
        
        # Generate HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Resume Scoring Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>Resume Scoring Analysis Report</h1>
            
            <h2>Similarity Scores</h2>
            <div id="bar-chart">{bar_chart.to_html(include_plotlyjs=False, div_id="bar-chart")}</div>
            
            <h2>Score Distribution</h2>
            <div id="dist-chart">{distribution_chart.to_html(include_plotlyjs=False, div_id="dist-chart")}</div>
        </body>
        </html>
        """
        
        return html_content
    
    def run(self):
        """Main application runner"""
        # Render header
        self.render_header()
        
        # Render sidebar and get configuration
        config = self.render_sidebar()
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Job description input
            self.handle_job_description_input()
            
            # Resume uploads
            self.handle_resume_uploads()
        
        with col2:
            # Control panel
            st.markdown('<h2 class="sub-header">🎮 Control Panel</h2>', unsafe_allow_html=True)
            
            # Analysis button
            if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
                success = self.run_scoring_analysis(config)
                if success:
                    st.rerun()
            
            # Clear data button
            if st.button("🗑️ Clear All Data", use_container_width=True):
                st.session_state.job_description = ""
                st.session_state.resume_texts = {}
                st.session_state.scoring_results = None
                st.rerun()
            
            # Status indicators
            st.markdown("### 📊 Status")
            job_status = "✅" if st.session_state.job_description else "❌"
            resume_status = "✅" if st.session_state.resume_texts else "❌"
            analysis_status = "✅" if st.session_state.scoring_results else "❌"
            
            st.markdown(f"""
            - Job Description: {job_status}
            - Resumes Uploaded: {resume_status} ({len(st.session_state.resume_texts)} files)
            - Analysis Complete: {analysis_status}
            """)
        
        # Results section
        st.markdown("---")
        self.display_results()


def main():
    """Main function to run the Streamlit app"""
    try:
        # Ensure NLTK data is available
        try:
            ensure_nltk_data()
        except Exception as nltk_error:
            st.warning(f"NLTK data setup warning: {nltk_error}")
            logger.warning(f"NLTK data setup warning: {nltk_error}")
        
        agent = ResumeAgent()
        agent.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}")


if __name__ == "__main__":
    main()
