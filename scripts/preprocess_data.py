"""
Summary of Data Preprocessing Steps:
1. Metadata Filtering
   ├── Remove empty tags
   ├── Remove empty descriptions
   └── Keep only English descriptions
       ↓
2. Games Filtering
   ├── Keep only games with valid metadata
   └── Create set of valid game IDs
       ↓
3. Users & Recommendations Filtering
   ├── First Pass:
   │   ├── Count valid recommendations per user
   │   └── Identify users with minimum recommendations
   │
   └── Second Pass:
       ├── Keep only recommendations for valid games
       └── Keep only recommendations from valid users

Final Output Files:
processed_data/
├── processed_games.csv
│   └── Original games filtered by metadata validity
│
├── processed_metadata.json
│   └── Clean metadata for valid games only
│
├── processed_recommendations.csv
│   └── Filtered recommendations matching valid games/users
│
└── processing_stats.json
    └── Statistics about the preprocessing:
        ├── Original counts
        ├── Processed counts
        └── Processing date
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Dict, List

import langdetect
import pandas as pd
from tqdm import tqdm


class SteamDataPreprocessor:
    def __init__(self, root_folder: str = '../data/'):
        """
        Initialize preprocessor with paths.

        Args:
            root_folder: Root directory containing the raw data files
        """
        self.root_folder = Path(root_folder)
        self.valid_games = set()
        self.valid_users = set()

        self.filenames = {
            'games': 'games.csv',
            'users': 'users.csv',
            'recommendations': 'recommendations.csv',
            'metadata': 'games_metadata_fixed.json'
        }

        if not self.root_folder.exists():
            raise FileNotFoundError(f"Data directory not found: {self.root_folder}")

        self._verify_files()

    def _verify_files(self):
        """Verify all required files exist."""
        missing_files = []
        for name, filename in self.filenames.items():
            file_path = self.root_folder / filename
            if not file_path.exists():
                missing_files.append(filename)

        if missing_files:
            raise FileNotFoundError(
                f"Missing required files in {self.root_folder}:\n" +
                "\n".join(missing_files)
            )

    def _get_file_path(self, file_key: str) -> Path:
        """Get full path for a file."""
        return self.root_folder / self.filenames[file_key]

    @staticmethod
    def _is_english_text(text: str) -> bool:
        """Check if text is in English."""
        try:
            return langdetect.detect(text) == 'en'
        except:
            return False

    def _process_metadata(self, metadata_path: str) -> List[Dict]:
        """Process and filter games metadata."""
        print("Loading game metadata...")
        with open(metadata_path, 'r') as f:
            games_metadata = json.load(f)

        print(f"Processing {len(games_metadata)} games metadata entries...")
        valid_metadata = []

        # Convert input to list if it's not already
        if isinstance(games_metadata, dict):
            games_metadata = [
                {"app_id": k, **v}
                for k, v in games_metadata.items()
            ]

        for game_data in tqdm(games_metadata, desc="Processing game metadata"):
            # Get app_id
            app_id = game_data.get('app_id')
            if not app_id:
                continue

            # Check tags and description
            description = game_data.get('description', '').strip()
            tags = game_data.get('tags', [])

            if (tags and
                    description and
                    len(tags) > 0 and
                    len(description) > 0 and
                    self._is_english_text(description)):
                valid_metadata.append({
                    'app_id': app_id,
                    'description': description,
                    'tags': tags
                })

        print(f"\nValid metadata entries: {len(valid_metadata)} out of {len(games_metadata)}")

        # Create game tags dictionary for later use
        self.game_tags = {
            str(game['app_id']): set(game['tags'])
            for game in valid_metadata
        }

        return valid_metadata

    def _process_games(self, games_path: str, valid_metadata: List[Dict]) -> pd.DataFrame:
        """Load and filter games DataFrame."""
        games_df = pd.read_csv(games_path)

        # Extract valid game IDs from metadata list
        valid_game_ids = [game['app_id'] for game in valid_metadata]
        filtered_games = games_df[games_df['app_id'].isin(valid_game_ids)].copy()

        print(f"\nGames after filtering:")
        print(f"Original: {len(games_df)}")
        print(f"Filtered: {len(filtered_games)}")

        return filtered_games

    def _process_users_and_recommendations(self,
                                           users_path: str,
                                           recommendations_path: str,
                                           valid_game_ids: List,
                                           batch_size: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process users and recommendations data in chunks."""
        # First pass: identify valid users
        user_recommendation_counts = defaultdict(int)

        print("\nFirst pass: identifying valid users...")
        for chunk in tqdm(pd.read_csv(recommendations_path, chunksize=batch_size),
                          desc="Processing recommendations"):

            # Filter recommendations for valid games
            valid_recommendations = chunk[chunk['app_id'].isin(valid_game_ids)]

            # Update user recommendation counts
            for user_id in valid_recommendations['user_id'].unique():
                user_recommendation_counts[user_id] += len(
                    valid_recommendations[valid_recommendations['user_id'] == user_id]
                )

        # Filter users with minimum recommendations
        MIN_RECOMMENDATIONS = 5
        valid_users = {
            user_id for user_id, count in user_recommendation_counts.items()
            if count >= MIN_RECOMMENDATIONS
        }

        # Load and filter users DataFrame
        users_df = pd.read_csv(users_path)
        filtered_users = users_df[users_df['user_id'].isin(valid_users)].copy()

        # Second pass: collect filtered recommendations
        filtered_recommendations_list = []

        print("\nSecond pass: collecting filtered recommendations...")
        for chunk in tqdm(pd.read_csv(recommendations_path, chunksize=batch_size),
                          desc="Filtering recommendations"):
            valid_chunk = chunk[(chunk['app_id'].isin(valid_game_ids)) & (chunk['user_id'].isin(valid_users))]
            filtered_recommendations_list.append(valid_chunk)

        filtered_recommendations = pd.concat(filtered_recommendations_list, ignore_index=True)

        print(f"\nUsers after filtering:")
        print(f"Original: {len(users_df)}")
        print(f"Filtered: {len(filtered_users)}")

        print(f"\nRecommendations after filtering:")
        print(f"Original games: {len(valid_game_ids)}")
        print(f"Original users: {len(valid_users)}")
        print(f"Total valid recommendations: {len(filtered_recommendations)}")

        return filtered_users, filtered_recommendations

    def load_and_clean_data(self,
                            output_dir: str = None,
                            batch_size: int = 10000) -> Tuple[pd.DataFrame, List[Dict], pd.DataFrame]:
        """
        Load and clean all datasets, ensuring consistency across them.

        Args:
            output_dir: Directory to save processed data (default: processed_data under root_folder)
            batch_size: Size of chunks for processing large files

        Returns:
            Tuple of (games_df, valid_metadata, filtered_recommendations)
        """
        print("=== Loading and Cleaning Steam Data ===")
        print(f"Reading data from: {self.root_folder}")

        if output_dir is None:
            output_dir = self.root_folder / 'processed_data'
        else:
            output_dir = Path(output_dir)

        print("\nProcessing games metadata...")
        valid_metadata = self._process_metadata(self._get_file_path('metadata'))

        print("\nProcessing games data...")
        games_df = self._process_games(self._get_file_path('games'), valid_metadata)

        # Load users and recommendations in chunks
        print("\nProcessing recommendations and users...")
        users_df, filtered_recommendations = self._process_users_and_recommendations(
            self._get_file_path('users'),
            self._get_file_path('recommendations'),
            games_df['app_id'].unique(),
            batch_size
        )

        self.save_processed_data(
            games_df=games_df,
            metadata=valid_metadata,
            recommendations_df=filtered_recommendations,
            output_dir=output_dir
        )

        return games_df, valid_metadata, filtered_recommendations

    def process_metadata_only(self, output_dir: str = None) -> List[Dict]:
        """
        Load and clean only the metadata file.

        Args:
            output_dir: Directory to save processed metadata (default: processed_data under root_folder)

        Returns:
            List of processed metadata entries
        """
        print("=== Processing Steam Games Metadata ===")
        print(f"Reading data from: {self.root_folder}")

        if output_dir is None:
            output_dir = self.root_folder / 'processed_data'
        else:
            output_dir = Path(output_dir)

        # Load and process metadata
        print("\nProcessing games metadata...")
        valid_metadata = self._process_metadata(self._get_file_path('metadata'))

        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'processed_metadata.json'

        with open(output_path, 'w') as f:
            json.dump(valid_metadata, f, indent=4)

        print(f"\nProcessed metadata saved to: {output_path}")
        print(f"Total valid entries: {len(valid_metadata)}")

        return valid_metadata

    def save_processed_data(self,
                            games_df: pd.DataFrame,
                            metadata: List[Dict],
                            recommendations_df: pd.DataFrame,
                            output_dir: Path):
        """Save processed datasets."""
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving processed data to: {output_dir}")

        games_df.to_csv(output_dir / 'processed_games.csv', index=False)
        recommendations_df.to_csv(output_dir / 'processed_recommendations.csv', index=False)

        with open(output_dir / 'processed_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)

        stats = {
            'original_games': len(self.valid_games),
            'processed_games': len(games_df),
            'original_users': len(self.valid_users),
            'processed_recommendations': len(recommendations_df),
            'processing_date': pd.Timestamp.now().isoformat()
        }

        with open(output_dir / 'processing_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        print("Saved files:")
        print(f"- {output_dir / 'processed_games.csv'}")
        print(f"- {output_dir / 'processed_recommendations.csv'}")
        print(f"- {output_dir / 'processed_metadata.json'}")
        print(f"- {output_dir / 'processing_stats.json'}")


def preprocess():
    preprocessor = SteamDataPreprocessor(root_folder='../data/')
    preprocessor.load_and_clean_data(
        output_dir='../processed_data/',
        batch_size=10000
    )


if __name__ == "__main__":
    preprocess()
