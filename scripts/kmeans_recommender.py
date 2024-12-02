import json
import pickle
import tempfile
import traceback
import zipfile
import random

from collections import defaultdict
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from kneed import KneeLocator
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class SteamDataLoader:
    """Handles loading and preprocessing of Steam data."""

    def __init__(self, data_dir: str = "../processed_data"):
        self.data_dir = Path(data_dir)
        self._validate_data_directory()

    def _validate_data_directory(self):
        required_files = ['processed_games.csv', 'processed_metadata.json', 'processed_recommendations.csv']
        missing_files = [f for f in required_files if not (self.data_dir / f).exists()]
        if missing_files:
            raise FileNotFoundError(f"Missing required files: {missing_files}")

    def load_games_data(self) -> pd.DataFrame:
        print("Loading games data...")
        return pd.read_csv(self.data_dir / "processed_games.csv")

    def load_metadata(self) -> list:
        print("Loading metadata...")
        with open(self.data_dir / "processed_metadata.json", 'r') as f:
            return json.load(f)

    def load_recommendations(self, batch_size: int = 10000):
        """Load recommendations in chunks."""
        print("Loading recommendations data in batches...")
        return pd.read_csv(
            self.data_dir / "processed_recommendations.csv",
            usecols=['user_id', 'app_id', 'is_recommended', 'hours'],
            chunksize=batch_size
        )


class UserFeatureExtractor:
    """Handles user feature extraction and processing."""

    def __init__(self, game_tags: dict):
        self.game_tags = game_tags
        self.user_features = defaultdict(self._create_user_feature_dict)

    @staticmethod
    def _create_user_feature_dict():
        return {
            'games': defaultdict(float),
            'total_hours': 0,
            'preferred_tags': defaultdict(int),
            'positive_ratio': 0,
            'n_games': 0
        }

    def process_user_interactions(self, recommendations_chunks):
        for chunk in tqdm(recommendations_chunks, desc="Processing user data"):
            self._process_chunk(chunk)
        self._calculate_positive_ratios()
        return self.user_features

    def _process_chunk(self, chunk):
        for _, row in chunk.iterrows():
            user_id = row['user_id']
            game_id = str(row['app_id'])
            self._update_user_features(user_id, game_id, row['is_recommended'], row['hours'])

    def _update_user_features(self, user_id, game_id, is_recommended, hours):
        user = self.user_features[user_id]
        user['games'][game_id] = is_recommended
        user['total_hours'] += hours
        user['n_games'] += 1

        if game_id in self.game_tags:
            self._update_tag_preferences(user, game_id, is_recommended)

    def _update_tag_preferences(self, user, game_id, is_recommended):
        for tag in self.game_tags[game_id]:
            if is_recommended:
                user['preferred_tags'][tag] += 1
            else:
                user['preferred_tags'][tag] -= 0.5

    def _calculate_positive_ratios(self):
        for user_id, user in self.user_features.items():
            positives = sum(1 for r in user['games'].values() if r > 0)
            total = len(user['games'])
            user['positive_ratio'] = positives / total if total > 0 else 0


class ModelPersistence:
    """Handles model saving and loading operations."""

    def __init__(self, model_dir: Path):
        self.large_components = None
        self.model_dir = model_dir
        self.model_file = self.model_dir / "model.zip"
        self.required_components = {
            'kmeans', 'user_clusters', 'cluster_recommendations',
            'user_to_idx', 'users', 'scaler', 'user_features',
            'game_tags'
        }
        self.component_filenames = {
            'user_features': 'user_features.pkl',
            'cluster_recommendations': 'cluster_recommendations.pkl',
            'game_tags': 'game_tags.pkl',
            'users': 'users.pkl',
            'core_model': 'core_model.pkl'
        }

    def save_model(self, model_instance) -> bool:
        """Save model components to a zip file."""
        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)

            # Create a temporary directory to store components before zipping
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                for component, filename in self.component_filenames.items():
                    if component != 'core_model':
                        with open(temp_path / filename, 'wb') as f:
                            pickle.dump(getattr(model_instance, component), f)

                core_model = {
                    'kmeans': model_instance.kmeans,
                    'user_clusters': model_instance.user_clusters,
                    'user_to_idx': model_instance.user_to_idx,
                    'scaler': model_instance.scaler,
                    'tags_list': model_instance.tags_list,
                    'model_metadata': {
                        'n_clusters': model_instance.n_clusters,
                        'n_users': len(model_instance.users),
                        'n_games': len(model_instance.game_tags),
                        'save_date': pd.Timestamp.now().isoformat()
                    }
                }

                with open(temp_path / self.component_filenames['core_model'], 'wb') as f:
                    pickle.dump(core_model, f)

                with zipfile.ZipFile(self.model_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for filename in self.component_filenames.values():
                        zipf.write(temp_path / filename, filename)

            print(f"Model saved to: {self.model_file}")
            return True

        except Exception as e:
            print(f"Error saving model: {str(e)}")
            traceback.print_exc()
            return False

    def load_model(self, model_instance) -> bool:
        """Load model components from a zip file."""
        try:
            if not self.model_file.exists():
                print(f"No saved model found at: {self.model_file}")
                return False

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                with zipfile.ZipFile(self.model_file, 'r') as zipf:
                    zipf.extractall(temp_path)

                print("Loading model components...")
                for component, filename in tqdm(self.component_filenames.items()):
                    if component != 'core_model':
                        with open(temp_path / filename, 'rb') as f:
                            setattr(model_instance, component, pickle.load(f))

                with open(temp_path / self.component_filenames['core_model'], 'rb') as f:
                    core_model = pickle.load(f)

                # Set core model attributes
                for key, value in core_model.items():
                    if key != 'model_metadata':
                        setattr(model_instance, key, value)

                # Set number of clusters
                if 'model_metadata' in core_model:
                    model_instance.n_clusters = core_model['model_metadata']['n_clusters']

                # Recreate tags_list if needed
                if not hasattr(model_instance, 'tags_list'):
                    all_tags = set()
                    for tags in model_instance.game_tags.values():
                        all_tags.update(tags)
                    model_instance.tags_list = sorted(all_tags)

                model_instance.visualizer = ClusterVisualizer(model_instance)

                print(f"\nModel loaded successfully!")
                print(f"Number of clusters: {model_instance.n_clusters}")
                print(f"Number of users: {len(model_instance.users)}")
                print(f"Number of games: {len(model_instance.game_tags)}")
                print(f"Number of unique tags: {len(model_instance.tags_list)}")
                return True

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            traceback.print_exc()
            return False


class ClusterVisualizer:
    def __init__(self, model):
        self.model = model

    def plot_all_analyses(self):
        """Generate all available visualizations."""
        print("\nGenerating visualization analysis...")
        self.plot_cluster_sizes()
        self.plot_cluster_engagement()
        self.plot_tag_importance()

    def plot_cluster_sizes(self):
        """Plot distribution of users across clusters."""
        plt.figure(figsize=(12, 6))

        cluster_sizes = pd.Series(self.model.user_clusters).value_counts().sort_index()
        bars = plt.bar(range(self.model.n_clusters), cluster_sizes)

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{int(height):,}',
                     ha='center', va='bottom')

        plt.title('Distribution of Users Across Clusters')
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Users')
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_cluster_engagement(self):
        """Plot cluster engagement metrics."""
        cluster_stats = self.model.analyze_clusters()

        clusters = list(cluster_stats.keys())
        hours = [stats['avg_hours'] for stats in cluster_stats.values()]
        ratios = [stats['avg_positive_ratio'] for stats in cluster_stats.values()]
        sizes = [stats['size'] for stats in cluster_stats.values()]

        fig, ax1 = plt.subplots(figsize=(12, 6))

        color1, color2 = 'tab:blue', 'tab:orange'
        ax1.set_xlabel('Cluster ID')
        ax1.set_ylabel('Average Hours per Game', color=color1)
        ax1.scatter(clusters, hours, c=color1, s=np.array(sizes) / 100,
                    alpha=0.6, label='Avg Hours')
        ax1.tick_params(axis='y', labelcolor=color1)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Positive Recommendation Ratio', color=color2)
        ax2.scatter(clusters, ratios, c=color2, s=np.array(sizes) / 100,
                    alpha=0.6, label='Positive Ratio')
        ax2.tick_params(axis='y', labelcolor=color2)

        plt.title('Cluster Engagement Metrics')
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_tag_importance(self, n_top_tags: int = 10):
        """Plot tag importance heatmap."""
        cluster_stats = self.model.analyze_clusters()

        all_tags = defaultdict(float)
        for stats in cluster_stats.values():
            for tag, importance in stats['top_tags']:
                all_tags[tag] += abs(importance)

        top_tags = sorted(all_tags.items(), key=lambda x: x[1], reverse=True)[:n_top_tags]
        top_tag_names = [tag for tag, _ in top_tags]

        importance_matrix = np.zeros((self.model.n_clusters, n_top_tags))
        for cluster in range(self.model.n_clusters):
            cluster_tag_dict = dict(cluster_stats[cluster]['top_tags'])
            for j, tag in enumerate(top_tag_names):
                importance_matrix[cluster, j] = cluster_tag_dict.get(tag, 0)

        plt.figure(figsize=(15, 8))
        sns.heatmap(importance_matrix, annot=True, fmt='.2f',
                    xticklabels=top_tag_names, yticklabels=range(self.model.n_clusters),
                    cmap='YlOrRd')
        plt.title(f'Top {n_top_tags} Tag Importance Across Clusters')
        plt.xlabel('Tags')
        plt.ylabel('Cluster ID')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


class HybridClusterRecommender:
    def __init__(self, data_dir: str = "../processed_data", n_clusters: int = 10):
        self.data_loader = SteamDataLoader(data_dir)
        self.model_persistence = ModelPersistence(Path(data_dir))
        self.n_clusters = n_clusters
        self.visualizer = None  # Will be initialized after model is ready

        # Initialize attributes needed for model persistence
        self.user_features = None
        self.game_tags = None
        self.kmeans = None
        self.user_clusters = None
        self.user_to_idx = None
        self.users = None
        self.tags_list = None
        self.scaler = None

    def train(self, auto_k: bool = True, k_range: Tuple[int, int] = (5, 30), save_model: bool = True) -> None:
        """
        Train the recommender system.

        Args:
            auto_k: Whether to automatically find optimal k
            k_range: Range of k values to test if auto_k is True
            save_model: Whether to save the trained model
        """
        try:
            print("Starting training process...")
            self._load_and_process_data()

            if auto_k:
                self.n_clusters = self.find_optimal_k(k_range)

            self._train_model()

            self.visualizer = ClusterVisualizer(self)

            if save_model:
                if self.model_persistence.save_model(self):
                    print("Model saved successfully!")
                else:
                    print("Warning: Failed to save model")

        except Exception as e:
            print(f"Error during training: {str(e)}")
            traceback.print_exc()
            raise

    def find_optimal_k(self, k_range: Tuple[int, int] = (5, 30)) -> int:
        """
        Find optimal number of clusters using elbow method.

        Args:
            k_range: Tuple of (min_k, max_k) to test

        Returns:
            Optimal number of clusters
        """
        print(f"Finding optimal k in range {k_range}...")

        # Calculate distortion for different k values
        distortions = []
        k_values = range(k_range[0], k_range[1] + 1)

        for k in tqdm(k_values, desc="Testing different k values"):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.feature_matrix)
            distortions.append(kmeans.inertia_)

        # Find elbow point
        elbow_locator = KneeLocator(
            list(k_values),
            distortions,
            curve='convex',
            direction='decreasing'
        )
        optimal_k = elbow_locator.elbow

        # Visualize elbow curve
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, distortions, 'bo-')
        plt.axvline(x=optimal_k, color='r', linestyle='--',
                    label=f'Elbow at k={optimal_k}')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Distortion (Inertia)')
        plt.title('Elbow Method for Optimal k')
        plt.legend()
        plt.grid(True)
        plt.show()

        print(f"\nOptimal number of clusters found: {optimal_k}")
        return optimal_k

    def analyze_clusters(self) -> dict:
        """Analyze characteristics of each cluster."""
        cluster_stats = {}

        for cluster in range(self.n_clusters):
            cluster_users = np.where(self.user_clusters == cluster)[0]
            centroid = self.kmeans.cluster_centers_[cluster]
            scaled_centroid = self.scaler.inverse_transform([centroid])[0]

            # Get top tags
            tag_importance = [
                (tag, scaled_centroid[i])
                for i, tag in enumerate(self.tags_list)
                if i < len(self.tags_list)
            ]
            top_tags = sorted(tag_importance, key=lambda x: x[1], reverse=True)[:5]

            cluster_stats[cluster] = {
                'size': len(cluster_users),
                'avg_hours': scaled_centroid[-2],
                'avg_positive_ratio': scaled_centroid[-1],
                'top_tags': top_tags,
                'top_games': self.cluster_recommendations[cluster][:5]
            }

        return cluster_stats

    def get_recommendations(self, user_id, n_recommendations=10, min_score=0.0):
        """
        Get personalized recommendations for a user.

        Args:
            user_id: User ID to get recommendations for
            n_recommendations: Number of recommendations to return
            min_score: Minimum score threshold for recommendations

        Returns:
            List of tuples (game_id, game_info) with recommendations
        """
        try:
            if user_id not in self.user_to_idx:
                raise ValueError(f"User {user_id} not found in training data")

            # Get user's cluster and games
            user_idx = self.user_to_idx[user_id]
            user_cluster = self.user_clusters[user_idx]
            user_games = set(self.user_features[user_id]['games'].keys())

            # Get recommendations from user's cluster
            cluster_recs = self.cluster_recommendations[user_cluster]

            # Load all games if not already loaded (needed after loading the model)
            if not hasattr(self, 'games_df'):
                self.games_df = self.data_loader.load_games_data()

            # Filter recommendations
            filtered_recommendations = []
            for game_id, score in cluster_recs:
                if score < min_score:
                    continue

                if game_id in user_games:
                    continue

                try:
                    game_row = self.games_df[self.games_df['app_id'] == int(game_id)].iloc[0]

                    game_info = {
                        'title': game_row['title'],
                        'tags': self.game_tags.get(game_id, []),
                        'score': score,
                        'cluster': user_cluster,
                        # Add additional game info if available
                        'price_final': game_row.get('price_final', None),
                        'positive_ratio': game_row.get('positive_ratio', None),
                        'user_reviews': game_row.get('user_reviews', None)
                    }

                    # Calculate tag similarity score
                    user_tags = set(tag for tag, score in
                                    self.user_features[user_id]['preferred_tags'].items()
                                    if score > 0)
                    game_tags = set(game_info['tags'])
                    tag_similarity = len(user_tags & game_tags) / len(user_tags | game_tags) if game_tags else 0

                    # Adjust score based on tag similarity
                    game_info['score'] = 0.5 * score + 0.5 * tag_similarity

                    filtered_recommendations.append((game_id, game_info))

                except (IndexError, KeyError) as e:
                    print(f"Warning: Error processing game {game_id}: {str(e)}")
                    continue

                if len(filtered_recommendations) >= n_recommendations:
                    break

            # Sort by adjusted score
            filtered_recommendations.sort(key=lambda x: x[1]['score'], reverse=True)

            return filtered_recommendations[:n_recommendations]

        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            traceback.print_exc()
            return []

    def display_user_analysis(self, user_id):
        """
        Display detailed analysis for a user and their recommendations.

        Args:
            user_id: User ID to analyze
        """
        try:
            if user_id not in self.user_to_idx:
                print(f"Error: User {user_id} not found in training data")
                return

            # Get user data
            user_idx = self.user_to_idx[user_id]
            user_cluster = self.user_clusters[user_idx]
            user_data = self.user_features[user_id]
            cluster_stats = self.analyze_clusters()[user_cluster]

            # Get recommendations
            recommendations = self.get_recommendations(user_id, n_recommendations=10)

            print(f"\n=== Random User Analysis (ID: {user_id}) ===")
            print(f"Cluster: {user_cluster}")

            print("\nUser Statistics:")
            total_games = len(user_data['games'])
            avg_hours = user_data['total_hours'] / total_games
            pos_ratio = user_data['positive_ratio'] * 100

            print(f"Total games played: {total_games}")
            print(f"Average hours per game: {avg_hours:.1f}")
            print(f"Positive recommendation ratio: {pos_ratio:.1f}%")

            print("\nUser's Top Tags:")
            top_tags = sorted(user_data['preferred_tags'].items(),
                              key=lambda x: x[1],
                              reverse=True)[:5]
            for tag, score in top_tags:
                print(f"- {tag}: {score:.1f}")

            print("\nTop 10 Recommended Games:")
            print("Game Title".ljust(50), "Score".ljust(10), "Top Tags")
            print("-" * 90)

            for game_id, info in recommendations:
                title = info['title'][:47] + "..." if len(info['title']) > 47 else info['title']
                tags = ", ".join(list(info['tags'])[:3])
                print(f"{title.ljust(50)} {f'{info['score']:.3f}'.ljust(10)} {tags}")

            print("\nCluster Context:")
            print(f"User's cluster size: {cluster_stats['size']:,} users")
            print(f"Cluster's top tags: {', '.join(tag for tag, _ in cluster_stats['top_tags'][:3])}")
            print(f"Cluster's average hours per game: {cluster_stats['avg_hours']:.1f}")
            print(f"Cluster's positive ratio: {cluster_stats['avg_positive_ratio'] * 100:.1f}%")

            print("\nUser vs Cluster Comparison:")
            print(f"Hours per game: {avg_hours:.1f} (Cluster: {cluster_stats['avg_hours']:.1f})")
            print(f"Positive ratio: {pos_ratio:.1f}% (Cluster: {cluster_stats['avg_positive_ratio'] * 100:.1f}%)")

        except Exception as e:
            print(f"Error displaying user analysis: {str(e)}")
            traceback.print_exc()

    def load_feature_matrix(self):
        self._create_feature_matrix()

    def _load_and_process_data(self):
        """Load and process all required data."""
        try:
            self.games_df = self.data_loader.load_games_data()
            metadata_list = self.data_loader.load_metadata()

            # Process game tags - convert list to dictionary format
            self.game_tags = {
                str(game['app_id']): set(game['tags'])
                for game in metadata_list
            }

            # Load and process user features
            feature_extractor = UserFeatureExtractor(self.game_tags)
            recommendations_chunks = self.data_loader.load_recommendations()
            self.user_features = feature_extractor.process_user_interactions(recommendations_chunks)

            self._create_feature_matrix()

        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def _train_model(self):
        """Train the KMeans clustering model and compute cluster profiles."""
        print(f"\nTraining model with {self.n_clusters} clusters...")

        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.user_clusters = self.kmeans.fit_predict(self.feature_matrix)

        self._compute_cluster_profiles()

        cluster_stats = self.analyze_clusters()
        print("\nCluster Analysis:")
        for cluster, stats in cluster_stats.items():
            print(f"\nCluster {cluster}:")
            print(f"Size: {stats['size']} users")
            print(f"Average hours per game: {stats['avg_hours']:.1f}")
            print(f"Average positive ratio: {stats['avg_positive_ratio']:.2f}")
            print("Top tags:", ', '.join(tag for tag, _ in stats['top_tags']))

    def _create_feature_matrix(self):
        """Create feature matrix for clustering from user features."""
        print("Creating feature matrix for clustering...")

        # Get all unique tags
        if self.tags_list is None:
            all_tags = set()
            for tags in self.game_tags.values():
                all_tags.update(tags)
            self.tags_list = sorted(all_tags)

        tag_to_idx = {tag: idx for idx, tag in enumerate(self.tags_list)}

        # Create feature matrix
        self.users = sorted(self.user_features.keys())
        self.user_to_idx = {user: idx for idx, user in enumerate(self.users)}

        n_features = len(self.tags_list) + 2  # tags + hours + positive ratio
        feature_matrix = np.zeros((len(self.users), n_features))

        for user_idx, user_id in enumerate(self.users):
            user = self.user_features[user_id]

            # Add tag preferences
            for tag, count in user['preferred_tags'].items():
                if tag in tag_to_idx:
                    feature_matrix[user_idx, tag_to_idx[tag]] = count / user['n_games']

            # Add behavioral features
            feature_matrix[user_idx, -2] = user['total_hours'] / user['n_games']
            feature_matrix[user_idx, -1] = user['positive_ratio']

        # Scale features
        self.scaler = StandardScaler()
        self.feature_matrix = self.scaler.fit_transform(feature_matrix)

    def _compute_cluster_profiles(self):
        """Compute detailed profiles for each cluster."""
        print("Computing cluster profiles...")
        self.cluster_profiles = defaultdict(lambda: defaultdict(list))

        # Track both ratings and hours for better scoring
        for user_idx, cluster in enumerate(self.user_clusters):
            user_id = self.users[user_idx]
            user = self.user_features[user_id]

            for game_id, rating in user['games'].items():
                hours = user['games'].get(game_id, 0)
                self.cluster_profiles[cluster]['games'].append((game_id, rating, hours))

        # Compute recommendations for each cluster
        self.cluster_recommendations = {}
        for cluster in range(self.n_clusters):
            games = self.cluster_profiles[cluster]['games']

            # Track ratings and hours
            game_stats = defaultdict(lambda: {'ratings': [], 'hours': []})
            for game_id, rating, hours in games:
                game_stats[game_id]['ratings'].append(rating)
                game_stats[game_id]['hours'].append(hours)

            # Compute weighted scores
            MIN_RATINGS = 5
            avg_ratings = {}

            for game_id, stats in game_stats.items():
                if len(stats['ratings']) >= MIN_RATINGS:
                    # Calculate weighted score
                    avg_rating = sum(stats['ratings']) / len(stats['ratings'])
                    n_ratings = len(stats['ratings'])
                    popularity = np.log2(n_ratings + 1) / 10
                    avg_hours = sum(stats['hours']) / len(stats['hours'])
                    engagement = min(np.log2(avg_hours + 1) / 10, 0.5)

                    final_score = (avg_rating + popularity + engagement) / 3
                    avg_ratings[game_id] = final_score

            self.cluster_recommendations[cluster] = sorted(
                avg_ratings.items(),
                key=lambda x: x[1],
                reverse=True
            )


def main():
    recommender = HybridClusterRecommender()

    try:
        if recommender.model_persistence.model_dir.exists():
            if recommender.model_persistence.load_model(recommender):
                print("Successfully loaded existing model")
            else:
                print("Training new model...")
                recommender.train(auto_k=True, save_model=True)
        else:
            print("Training new model...")
            recommender.train(auto_k=True, save_model=True)

        recommender.visualizer.plot_all_analyses()

        random_user = recommender.users[random.randint(0, len(recommender.users) - 1)]
        recommender.display_user_analysis(random_user)

    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
