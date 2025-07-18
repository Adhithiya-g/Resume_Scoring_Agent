# Resume Scoring AI Agent

A sophisticated AI-powered resume scoring system that evaluates and ranks resumes against job descriptions using TF-IDF vectorization and Cosine Similarity.

## 🎯 Features

- **Multi-format Support**: Extract text from PDF, DOCX, and TXT files
- **Intelligent Preprocessing**: Advanced NLP preprocessing with stopword removal, tokenization, and lemmatization
- **TF-IDF Vectorization**: Convert text data into meaningful feature vectors
- **Cosine Similarity Scoring**: Compute similarity scores between resumes and job descriptions
- **Smart Ranking**: Sort and rank resumes based on relevance scores
- **Interactive UI**: User-friendly Streamlit interface
- **Explainability**: Highlight matching keywords and skills
- **Export Functionality**: Download results as CSV or PDF reports
- **Visualizations**: Interactive charts and word clouds

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python, scikit-learn
- **NLP**: NLTK, spaCy
- **File Processing**: PyPDF2, pdfplumber, python-docx
- **Visualization**: matplotlib, plotly, seaborn
- **Data Processing**: pandas, numpy

## 📁 Project Structure

```
resume_scoring_agent/
├── app.py                  # Main Streamlit application
├── utils/
│   ├── extractor.py        # Text extraction from various file formats
│   ├── preprocessor.py     # NLP preprocessing and cleaning
│   ├── scorer.py           # TF-IDF vectorization and scoring
│   └── visualizer.py       # Charts and visualizations
├── data/
│   └── sample_resumes/     # Sample resume files
├── templates/              # Report templates
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download required NLTK data:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## 💻 Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to the provided local URL

3. Upload your job description and resume files

4. View the ranked results with similarity scores

5. Explore matching keywords and download reports

## 📊 Example Output

```
Job Title: Data Scientist
Uploaded Resumes: 10

Top Matches:
1. resume_raj.pdf - Score: 0.82 ✅
2. resume_anita.pdf - Score: 0.76
3. resume_sam.docx - Score: 0.69

Keywords Matched: ['machine learning', 'python', 'data analysis']
```

## 🔧 Configuration

The system can be customized through various parameters:
- Minimum similarity threshold
- Number of top keywords to display
- TF-IDF parameters (min_df, max_df, ngram_range)
- Preprocessing options

## 📈 Advanced Features

- **Keyword Highlighting**: Identifies and highlights matching terms
- **Skill Overlap Analysis**: Shows common skills between JD and resumes
- **Score Distribution**: Visualizes similarity score distribution
- **Word Clouds**: Generate word clouds for job descriptions and resumes
- **Batch Processing**: Handle multiple resumes efficiently

## 🤝 Contributing

Feel free to contribute by:
- Adding new file format support
- Improving preprocessing algorithms
- Enhancing visualization features
- Adding new scoring methods

## 📄 License

This project is open source and available under the MIT License.
