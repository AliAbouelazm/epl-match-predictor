# English Premier League Match Outcome Predictor

An end-to-end machine learning project that predicts EPL match outcomes (Home Win, Draw, Away Win) using XGBoost. The project demonstrates data collection through web scraping, data preprocessing, feature engineering, model training, and an interactive Streamlit visualization.

## Live Demo

Try the interactive Streamlit app: **[https://pl-predict.streamlit.app/](https://pl-predict.streamlit.app/)**

## Technologies

- **Python** - Primary programming language
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning utilities
- **xgboost** - Gradient boosting classifier
- **BeautifulSoup** - HTML parsing for web scraping
- **Selenium** - Browser automation for JavaScript-rendered content
- **Streamlit** - Interactive web application
- **matplotlib/seaborn/plotly** - Data visualization

## Project Structure

```
epl-match-predictor/
├── data/
│   ├── raw/              # Raw CSV files and scraped data
│   ├── interim/          # Cleaned data
│   └── processed/        # Feature matrices ready for modeling
├── notebooks/
│   ├── 01_eda.ipynb      # Exploratory data analysis
│   └── 02_model_dev.ipynb # Model prototyping
├── src/
│   ├── config.py         # Configuration constants
│   ├── data_acquisition/
│   │   ├── download_public_datasets.py
│   │   ├── scrape_matches_bs4.py
│   │   └── scrape_matches_selenium.py
│   ├── data_preprocessing/
│   │   ├── clean_raw_data.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── train_xgboost.py
│   │   ├── evaluate_model.py
│   │   ├── feature_importance.py
│   │   └── prediction_utils.py
│   ├── visualization/
│   │   ├── plot_feature_importance.py
│   │   └── plot_performance_metrics.py
│   └── app/
│       └── streamlit_app.py
├── tests/                # Unit tests
├── models/               # Saved trained models
├── reports/              # Evaluation reports and visualizations
├── requirements.txt
└── README.md
```

## Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AliAbouelazm/epl-match-predictor.git
cd epl-match-predictor
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install ChromeDriver (for Selenium)

**macOS:**
```bash
brew install chromedriver
```

**Linux:**
```bash
sudo apt-get install chromium-chromedriver
```

**Windows:**
Download from [ChromeDriver downloads](https://chromedriver.chromium.org/downloads) and add to PATH.

## Data Overview

### Public Dataset

The project uses historical EPL match data. To obtain the dataset:

1. Download EPL historical data from [Kaggle](https://www.kaggle.com/datasets) or similar sources
2. Place the CSV file in `data/raw/epl_matches_raw.csv`
3. Ensure the CSV contains columns: `date`, `home_team`, `away_team`, `home_goals`, `away_goals`, `result` (or similar)

### Web Scraping

The project includes two scraping examples:

- **BeautifulSoup** (`scrape_matches_bs4.py`) - Scrapes static HTML content
- **Selenium** (`scrape_matches_selenium.py`) - Handles JavaScript-rendered pages

**Note:** These scripts are for educational purposes. Always respect `robots.txt` and website Terms of Service.

## How to Run the Pipeline

### Step 1: Acquire Data

```bash
# Download public dataset (or place manually in data/raw/)
python src/data_acquisition/download_public_datasets.py

# Scrape with BeautifulSoup
python src/data_acquisition/scrape_matches_bs4.py

# Scrape with Selenium
python src/data_acquisition/scrape_matches_selenium.py
```

### Step 2: Clean and Preprocess Data

```bash
# Clean raw data
python src/data_preprocessing/clean_raw_data.py

# Build feature matrix
python src/data_preprocessing/feature_engineering.py
```

### Step 3: Train Model

```bash
python src/models/train_xgboost.py
```

### Step 4: Evaluate Model

```bash
python src/models/evaluate_model.py
python src/models/feature_importance.py
```

### Step 5: Run Streamlit App

```bash
streamlit run src/app/streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`.

## Model Details

### Target Variable

- **0** = Away Win
- **1** = Draw
- **2** = Home Win

### Features

The model uses rolling statistics calculated over the last 5 matches:

- **Team Performance:**
  - Average goals scored
  - Average goals conceded
  - Average points per match
  - Home/away specific statistics

- **Difference Features:**
  - Difference in goals scored (home - away)
  - Difference in goals conceded
  - Difference in points

### Evaluation

The model is evaluated using:
- Accuracy
- Balanced Accuracy
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)

## Running Tests

```bash
pytest tests/
```

## Limitations and Future Work

### Current Limitations

- Does not include player injuries, transfers, or tactical changes
- Limited to historical match statistics
- No real-time data integration
- Does not consider betting odds or market information

### Future Improvements

- Integrate player-level statistics
- Add injury and transfer data
- Include betting odds as features
- Real-time data pipeline for live predictions
- Deploy to cloud platform (Streamlit Community Cloud, AWS, etc.)
- Add more sophisticated feature engineering (head-to-head records, etc.)

## Ethical & Legal Note

The web scraping scripts in this project are provided for educational and demonstration purposes. Users must:

- Respect `robots.txt` files
- Comply with website Terms of Service
- Not overload servers with excessive requests
- Use scraped data responsibly and ethically

## License

This project is for educational purposes.

## Author

**AliAbouelazm**

---

For questions or issues, please open an issue on GitHub.

