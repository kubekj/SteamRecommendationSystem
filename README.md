# Steam Games Recommendation System
A collaborative filtering recommendation system for Steam games that uses K-means clustering to group users with similar gaming preferences and generate personalized recommendations.

## Project Structure (Only Relevant Files)

```
├── data/                            # Raw data files from Kaggle
│   ├── games.csv                    # Game information
│   ├── games_metadata.json          # Game metadata
│   ├── games_metadata_fixed.json    # Fixed metadata
│   ├── games_metadata_scraped.json  # Fixed and webscraped metadata
│   ├── recommendations.csv          # User recommendations
│   └── users.csv                    # User information
│
├── processed_data/                     # Processed datasets
│   ├── processed_games.csv             # Filtered games data
│   ├── processed_metadata.json         # Filtered metadata
│   └── processed_recommendations.csv   # Filtered recommendations
│
├── scripts/                       # Core implementation files
│   ├── fix_metadata_json.py       # Metadata cleaning script
│   ├── kmeans_recommender.py      # K-means clustering implementation
│   ├── preprocess_data.py         # Data preprocessing pipeline
│
|                                   # Jupyter notebooks for analysis
├── clustering.ipynb                # Clustering analysis
├── data_exploration.ipynb          # Initial data exploration
└── data_visualisation.ipynb        # Initial data visualisation
```

## Setup and Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Data Processing

**IMPORTANT**: Before running any files (except `data_exploration.ipynb`), you must first process the raw data:

1. Ensure you have the raw data files in the `data/` directory - download from [Kaggle](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam) if absent.
2. Run the metadata cleaning script:
```bash
python scripts/fix_metadata_json.py
```
3. Run the preprocessing script:
```bash
python scripts/preprocess_data.py
```
This will:
- Clean and validate the raw data
- Filter out invalid entries
- Generate processed datasets in `processed_data/`
- Create necessary data structures for the recommendation systems

4. Train/run the recommender:
```bash
python scripts/kmeans_recommender.py
```

## Implementation Overview
### Data Preprocessing
The system begins with data preparation through `preprocess_data.py`, which:
- Cleans and validates raw Steam data
- Removes invalid or incomplete entries
- Generates processed datasets
- Prepares feature vectors for clustering

### Clustering Engine
The core recommendation logic in kmeans_recommender.py implements:

- User feature vector creation based on:
  - Game preferences (liked/disliked)
  - Gameplay hours
  - Tag preferences
- K-means clustering for user grouping
- Cluster profile computation
- Recommendation generation

### Recommendation Generation
The system generates recommendations by:

- Assigning users to appropriate clusters
- Analyzing cluster-wide preferences
- Ranking potential recommendations using:
  - Cluster game ratings
  - User-game tag similarity
  - Game popularity within clusters

## Documentation & Analysis

The project includes three key notebooks that provide comprehensive analysis and documentation:

### Data Understanding
`data_exploration.ipynb` and `data_visualization.ipynb`:
- Initial dataset analysis and statistics
- Data quality assessment
- Feature distributions and relationships
- Visualization of gaming patterns
- Identification of potential data issues

### Implementation Analysis  
`clustering.ipynb`:
- Detailed clustering methodology
- Feature engineering process
- Parameter tuning experiments
- Performance evaluation
- System design decisions

For complete implementation details and hands-on examples, start with the data exploration notebooks before diving into the clustering analysis.
