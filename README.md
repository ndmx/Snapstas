# Snapstats - Nigeria States Map Visualization

This project provides an interactive visualization of Nigeria's states using GeoJSON data. It includes an ETL pipeline for data processing and a Streamlit web application for visualization.

## Features

- Interactive map with state boundaries and tooltips
- Static map with colored regions and state labels
- Data table with state information
- Automated ETL pipeline for data processing

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Snapstats
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the ETL pipeline (optional - will run automatically if needed):
```bash
python etl.py
```

2. Start the Streamlit app:
```bash
streamlit run app.py
```

3. Open your web browser and navigate to `http://localhost:8501`

## Project Structure

- `etl.py`: ETL pipeline for fetching and processing GeoJSON data
- `app.py`: Streamlit web application
- `requirements.txt`: Project dependencies
- `data/`: Directory for storing processed data (created automatically)

## Data Sources

The GeoJSON data is sourced from the [geoBoundaries](https://github.com/wmgeolab/geoBoundaries) project.
