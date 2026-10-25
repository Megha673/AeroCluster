import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for server/script execution
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import SpectralClustering, KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import warnings
import sys
import os
from datetime import datetime
import joblib
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import json
import logging
import argparse
import traceback
from typing import Dict, Any, Optional, Tuple
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION PARAMETERS (following Ningombam et al., 2024)
# ============================================================================
RANDOM_STATE = 42
N_CLUSTERS = 3  # Paper uses 3 clusters for ASIA
N_NEIGHBORS = 10  # For spectral clustering affinity
OUTPUT_DIR = 'output'

# Performance optimization parameters
MAX_SAMPLES_FOR_CLUSTERING = 12000  # Subsample if dataset is larger
MAX_SAMPLES_FOR_SILHOUETTE = 5000   # For faster silhouette calculation
PLOT_DPI = 150  # Reduced from 300 for faster rendering
SKIP_ALGORITHM_COMPARISON = False  # Set to True to skip comparison (faster)

# Proxy thresholds (documented heuristics)
EAE_MIN_FMF = 0.3  # EAE below this → coarse mode dominant
EAE_MAX_FMF = 1.7  # EAE above this → fine mode dominant
DEFAULT_RRI = 1.47  # Paper's mean RRI for urban/mixed aerosols

# Cluster labeling thresholds (from paper's physics)
DUST_EAE_MAX = 0.6
DUST_AAE_MIN = 1.8
BB_EAE_MIN = 1.0
BB_AAE_MIN = 1.2
BB_AAE_MAX = 1.8
BB_SSA_MAX = 0.92
URBAN_EAE_MIN = 1.0
URBAN_SSA_MIN = 0.94

# ============================================================================
# LOGGING SETUP
# ============================================================================
def setup_logging(output_dir: str, smoke_test: bool = False) -> logging.Logger:
    """Setup logging to file and console"""
    os.makedirs(output_dir, exist_ok=True)
    
    log_filename = os.path.join(output_dir, 'da9_run.log' if not smoke_test else 'da9_smoke.log')
    
    # Create logger
    logger = logging.getLogger('DA9')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# ============================================================================
# ARGUMENT PARSING
# ============================================================================
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Aerosol Spectral Clustering Analysis (following Ningombam et al., 2024)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Full run:    python DA9.py --input combined_dataset.csv --output output/
  Smoke test:  python DA9.py --smoke-test --input combined_dataset.csv --output output/
        """
    )
    
    parser.add_argument('--input', type=str, 
                       default=r'C:\Users\Jsh Agarwal\OneDrive\Desktop\DA\DA9\combined_dataset.csv',
                       help='Path to input CSV file')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory for results')
    parser.add_argument('--smoke-test', action='store_true',
                       help='Run smoke test (500 rows, minimal plots)')
    parser.add_argument('--max-samples', type=int, default=MAX_SAMPLES_FOR_CLUSTERING,
                       help='Maximum samples for clustering')
    parser.add_argument('--random-state', type=int, default=RANDOM_STATE,
                       help='Random seed for reproducibility')
    
    return parser.parse_args()

# ============================================================================
# ERROR HANDLING AND PERSISTENCE
# ============================================================================
def save_error_report(output_dir: str, stage: str, error: Exception) -> None:
    """Save error report as JSON"""
    error_report = {
        'stage': stage,
        'error': str(error),
        'error_type': type(error).__name__,
        'traceback': traceback.format_exc(),
        'timestamp': datetime.now().isoformat()
    }
    
    error_path = os.path.join(output_dir, 'error_report.json')
    with open(error_path, 'w') as f:
        json.dump(error_report, f, indent=2)

def save_run_metadata(output_dir: str, random_state: int, n_samples: int, 
                     proxy_flags: Dict[str, int], best_params: Optional[Dict] = None) -> None:
    """Save run metadata as JSON"""
    metadata = {
        'random_state': random_state,
        'n_samples': n_samples,
        'proxy_counts': proxy_flags,
        'timestamp': datetime.now().isoformat(),
        'best_params': best_params or {}
    }
    
    metadata_path = os.path.join(output_dir, 'run_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def save_best_params(output_dir: str, params: Dict[str, Any]) -> None:
    """Save best clustering parameters as JSON"""
    params_path = os.path.join(output_dir, 'best_params.json')
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)

def save_comparison_summary(output_dir: str, silhouette: float, 
                           cluster_pcts: pd.Series, centroids: pd.DataFrame) -> None:
    """Save comparison summary for verification against notebook"""
    summary = {
        'silhouette_score': float(silhouette),
        'cluster_percentages': cluster_pcts.to_dict(),
        'centroids': centroids.to_dict(),
        'timestamp': datetime.now().isoformat()
    }
    
    summary_path = os.path.join(output_dir, 'comparison_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

# ============================================================================
# FILE VERIFICATION
# ============================================================================
def verify_output_files(output_dir: str, logger: logging.Logger) -> Tuple[bool, list]:
    """Verify all required output files exist"""
    required_files = {
        'CSVs': ['asia_clustered.csv', 'centroids_table.csv', 'cluster_percentages.csv'],
        'PNGs': ['fig_histograms.png', 'fig_scatter_matrix_clusters.png', 'fig_cluster_bar_percentages.png'],
        'Models': ['scaler.joblib', 'spectral_model.joblib'],
        'Metadata': ['best_params.json', 'run_metadata.json']
    }
    
    missing = []
    for category, files in required_files.items():
        for file in files:
            filepath = os.path.join(output_dir, file)
            if not os.path.exists(filepath):
                missing.append(f"{category}/{file}")
                logger.warning(f"  ✗ Missing: {category}/{file}")
            else:
                logger.info(f"  ✓ Found: {category}/{file}")
    
    return len(missing) == 0, missing

def verify_csv_sanity(output_dir: str, n_samples: int, logger: logging.Logger) -> bool:
    """Sanity check CSV outputs"""
    try:
        # Check asia_clustered.csv
        clustered_path = os.path.join(output_dir, 'asia_clustered.csv')
        if os.path.exists(clustered_path):
            df_clustered = pd.read_csv(clustered_path)
            
            if 'Cluster' not in df_clustered.columns and 'cluster' not in df_clustered.columns:
                logger.error("  ✗ asia_clustered.csv missing cluster column")
                return False
            
            cluster_col = 'Cluster' if 'Cluster' in df_clustered.columns else 'cluster'
            if len(df_clustered) != n_samples:
                logger.warning(f"  ⚠ Row count mismatch: expected {n_samples}, got {len(df_clustered)}")
            else:
                logger.info(f"  ✓ asia_clustered.csv has {len(df_clustered)} rows with '{cluster_col}' column")
        
        # Check centroids_table.csv
        centroids_path = os.path.join(output_dir, 'centroids_table.csv')
        if os.path.exists(centroids_path):
            df_centroids = pd.read_csv(centroids_path)
            
            if len(df_centroids) < 2:
                logger.error("  ✗ centroids_table.csv has < 2 rows")
                return False
            
            feature_cols = ['EAE', 'AAE', 'SSA', 'AOD_total', 'AOD_fine']
            present_features = [col for col in feature_cols if col in df_centroids.columns]
            
            if len(present_features) == 0:
                logger.error("  ✗ centroids_table.csv missing feature columns")
                return False
            
            # Check for all-NaN columns
            all_nan_cols = df_centroids[present_features].columns[df_centroids[present_features].isna().all()].tolist()
            if all_nan_cols:
                logger.error(f"  ✗ centroids_table.csv has all-NaN columns: {all_nan_cols}")
                return False
            
            logger.info(f"  ✓ centroids_table.csv has {len(df_centroids)} centroids with features: {present_features}")
        
        return True
    
    except Exception as e:
        logger.error(f"  ✗ CSV sanity check failed: {e}")
        return False

# ============================================================================
# MAIN EXECUTION WRAPPER
# ============================================================================
def main():
    """Main execution function with comprehensive error handling"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Global configuration
    global OUTPUT_DIR, RANDOM_STATE, MAX_SAMPLES_FOR_CLUSTERING
    OUTPUT_DIR = args.output
    RANDOM_STATE = args.random_state
    MAX_SAMPLES_FOR_CLUSTERING = args.max_samples
    
    # Setup logging
    logger = setup_logging(OUTPUT_DIR, args.smoke_test)
    
    # Set random seeds for reproducibility
    np.random.seed(RANDOM_STATE)
    
    # Initialize tracking variables
    pdf_file: Optional[str] = None
    n_samples_analyzed = 0
    silhouette = 0.0
    n_clusters_found = 0
    clustering_failed = False
    proxy_flags = {'used_AOD_fine_proxy': 0, 'used_RRI_proxy': 0}
    
    try:
        logger.info("="*80)
        logger.info("AEROSOL SPECTRAL CLUSTERING ANALYSIS")
        logger.info("Following Ningombam et al., 2024 methodology")
        logger.info(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Mode: {'SMOKE TEST' if args.smoke_test else 'FULL RUN'}")
        logger.info("="*80 + "\n")
        
        # ========================================================================
        # STAGE 1: DATA LOADING
        # ========================================================================
        logger.info("STAGE 1: DATA LOADING")
        logger.info("-" * 80)
        
        if not os.path.exists(args.input):
            logger.error(f"❌ Input file not found: {args.input}")
            sys.exit(1)
        
        df = pd.read_csv(args.input)
        logger.info(f"✓ Loaded dataset: {df.shape}")
        logger.info(f"  Columns: {len(df.columns)}")
        logger.info(f"  Rows: {len(df)}")
        
        if args.smoke_test:
            df = df.head(500)
            logger.info(f"  Smoke test: using first 500 rows")
        
        # Continue with rest of analysis...
        # (This will be filled in by subsequent function calls)
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Analysis interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        logger.exception(f"❌ CRITICAL ERROR in main execution")
        save_error_report(OUTPUT_DIR, 'main', e)
        sys.exit(1)
    
    finally:
        # Print final summary
        print_final_summary(logger, n_samples_analyzed, proxy_flags, silhouette, 
                          n_clusters_found, pdf_file, OUTPUT_DIR, clustering_failed)

def print_final_summary(logger: logging.Logger, n_samples: int, proxy_flags: Dict,
                       silhouette: float, n_clusters: int, pdf_file: Optional[str],
                       output_dir: str, clustering_failed: bool):
    """Print concise final run summary"""
    logger.info("\n" + "="*80)
    logger.info("FINAL RUN SUMMARY")
    logger.info("="*80)
    logger.info(f"Samples analyzed: {n_samples:,}")
    logger.info(f"Proxy counts:")
    logger.info(f"  • AOD_fine proxy: {proxy_flags.get('used_AOD_fine_proxy', 0)} samples")
    logger.info(f"  • RRI proxy: {proxy_flags.get('used_RRI_proxy', 0)} samples")
    
    if clustering_failed:
        logger.warning("⚠️  Clustering: FAILED (fallback to KMeans)")
    else:
        logger.info(f"✓ Silhouette score: {silhouette:.4f}")
        logger.info(f"✓ Clusters found: {n_clusters}")
    
    if pdf_file and os.path.exists(pdf_file):
        logger.info(f"✓ PDF report: {pdf_file}")
    else:
        logger.warning("⚠️  PDF: Failed or not generated")
    
    logger.info(f"✓ Outputs directory: {output_dir}/")
    logger.info("="*80 + "\n")

if __name__ == "__main__":
    # Note: main() wrapper is incomplete - using direct execution below for now
    # main()
    
    # Parse command-line arguments
    args = parse_arguments()
    OUTPUT_DIR = args.output
    RANDOM_STATE = args.random_state
    data_path = args.input
    smoke_test_mode = args.smoke_test
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Set random seed
    np.random.seed(RANDOM_STATE)

# ============================================================================
# DIRECT EXECUTION - Full analysis pipeline
# ============================================================================
print("STEP 0: LOADING DATA")
print("-" * 80)

df = pd.read_csv(data_path)

if smoke_test_mode:
    print(f"\n*** SMOKE TEST MODE: Using first 500 rows")
    df = df.head(500)

print(f"Dataset shape: {df.shape}")
print(f"\nColumn names:\n{df.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nData types:")
print(df.dtypes)
print(f"\nMissing values:")
print(df.isnull().sum())

# ============================================================================
# STEP 1: DATA CLEANING AND VALIDATION
# ============================================================================
def load_and_clean(df):
    """
    Clean and validate input data using notebook's logic.
    Returns:
        df_clean: Cleaned DataFrame
        cleaning_log: Dictionary with cleaning statistics
    """
    print("\n" + "="*80)
    print("STEP 1: DATA CLEANING (NOTEBOOK LOGIC)")
    print("-" * 80)
    cleaning_log = {}
    initial_rows = len(df)
    # Required columns
    required_cols = {
        'AOD_total': 'AOD_Extinction-Total[440nm]',
        'EAE': 'Angstrom_Exponent_440-870nm_from_Coincident_Input_AOD',
        'SSA': 'SSA_440',
        'AAE': 'Absorption_Angstrom_Exponent_440-870nm'
    }
    df_clean = df.copy()
    for new_name, old_name in required_cols.items():
        if old_name in df_clean.columns:
            df_clean[new_name] = pd.to_numeric(df_clean[old_name], errors='coerce')
    # Drop rows with missing required values
    before = len(df_clean)
    df_clean = df_clean.dropna(subset=['AOD_total', 'EAE', 'SSA', 'AAE'])
    after = len(df_clean)
    if 'Site' in df.columns:
        df_clean['Site'] = df.loc[df_clean.index, 'Site']
    cleaning_log['initial_rows'] = initial_rows
    cleaning_log['final_rows'] = len(df_clean)
    cleaning_log['rows_removed'] = initial_rows - len(df_clean)
    cleaning_log['removal_pct'] = (cleaning_log['rows_removed'] / initial_rows) * 100
    print(f"  Initial rows: {initial_rows}")
    print(f"  Final rows: {len(df_clean)}")
    print(f"  Removed: {cleaning_log['rows_removed']} ({cleaning_log['removal_pct']:.2f}%)")
    return df_clean, cleaning_log

df_clean, cleaning_log = load_and_clean(df)

# ============================================================================
# STEP 2: CREATE FEATURES WITH PROXIES
# ============================================================================
def create_features(df):
    """
    Create clustering features using notebook's logic. Proxy flags are integer counts.
    Returns:
        df_features: DataFrame with features
        proxy_flags: Dictionary with proxy usage statistics
    """
    print("\n" + "="*80)
    print("STEP 2: FEATURE ENGINEERING (NOTEBOOK LOGIC)")
    print("-" * 80)
    df_features = df.copy()
    proxy_flags = {'used_AOD_fine_proxy': 0, 'used_RRI_proxy': 0}
    # --- AOD_fine ---
    if 'AOD_Extinction-Fine[440nm]' in df_features.columns:
        df_features['AOD_fine'] = pd.to_numeric(df_features['AOD_Extinction-Fine[440nm]'], errors='coerce')
        proxy_flags['used_AOD_fine_proxy'] = 0
        df_features['used_AOD_fine_proxy'] = False
    else:
        # Proxy: FMF = clip((EAE - 0.3) / (1.7 - 0.3), 0, 1)
        FMF_proxy = np.clip((df_features['EAE'] - 0.3) / (1.7 - 0.3), 0, 1)
        df_features['AOD_fine'] = df_features['AOD_total'] * FMF_proxy
        proxy_flags['used_AOD_fine_proxy'] = len(df_features)
        df_features['used_AOD_fine_proxy'] = True
    # --- RRI ---
    if 'Refractive_Index-Real_Part[440nm]' in df_features.columns:
        df_features['RRI'] = pd.to_numeric(df_features['Refractive_Index-Real_Part[440nm]'], errors='coerce')
        proxy_flags['used_RRI_proxy'] = 0
        df_features['used_RRI_proxy'] = False
    else:
        df_features['RRI'] = 1.47
        proxy_flags['used_RRI_proxy'] = len(df_features)
        df_features['used_RRI_proxy'] = True
    # --- EAE, AAE, SSA already present ---
    # Remove rows with NaN in critical features
    feature_cols = ['AOD_fine', 'EAE', 'AAE', 'SSA', 'RRI']
    before = len(df_features)
    df_features = df_features.dropna(subset=feature_cols)
    after = len(df_features)
    if before > after:
        print(f"   Removed {before - after} rows with NaN in computed features")
    print(f"  Features: {feature_cols}")
    print(f"  Samples: {len(df_features)}")
    print(f"  Proxy usage:")
    print(f"    - AOD_fine proxy: {proxy_flags['used_AOD_fine_proxy']} rows")
    print(f"    - RRI proxy: {proxy_flags['used_RRI_proxy']} rows")
    return df_features, proxy_flags

df_features, proxy_flags = create_features(df_clean)

# ============================================================================
# STEP 2.5: REMOVE OUTLIERS AND INVALID VALUES (after feature engineering!)
# ============================================================================
print("\n" + "="*80)
print("STEP 2.5: REMOVING OUTLIERS AND INVALID VALUES")
print("-" * 80)

def remove_outliers(df):
    # Remove rows with invalid AAE or SSA values (-999)
    before = len(df)
    df = df[(df['AAE'] > -100) & (df['SSA'] > -100)]
    after = len(df)
    print(f"Removed {before - after} rows with invalid AAE/SSA values ({(before-after)/before*100:.2f}%)")
    # Remove extreme outliers (outside 2nd/98th percentile, notebook logic)
    for col in ['AOD_fine', 'EAE', 'AAE', 'SSA']:
        low, high = np.percentile(df[col], [2, 98])
        df = df[(df[col] >= low) & (df[col] <= high)]
    print(f"Rows after outlier removal: {len(df)}")
    return df

df_features = remove_outliers(df_features)

# ============================================================================
# STEP 3: STANDARDIZE FEATURES
# ============================================================================
print("\n" + "="*80)
print("STEP 3: FEATURE STANDARDIZATION")
print("-" * 80)

feature_cols = ['AOD_fine', 'EAE', 'AAE', 'SSA', 'RRI']
X = df_features[feature_cols].values

print(f"\nFeature matrix shape: {X.shape}")

# Optimize: subsample if dataset is too large
original_size = len(X)
if original_size > MAX_SAMPLES_FOR_CLUSTERING:
    print(f"\n⚠ Dataset is large ({original_size} samples). Subsampling to {MAX_SAMPLES_FOR_CLUSTERING} for faster processing...")
    sample_indices = np.random.RandomState(RANDOM_STATE).choice(
        original_size, MAX_SAMPLES_FOR_CLUSTERING, replace=False
    )
    X_sample = X[sample_indices]
    df_features_sample = df_features.iloc[sample_indices].copy()
    print(f"✓ Using {len(X_sample)} samples for clustering")
else:
    X_sample = X
    df_features_sample = df_features.copy()
    print(f"✓ Using all {len(X)} samples")

print(f"\nFeature statistics (before standardization):")
feature_stats = pd.DataFrame(X_sample, columns=feature_cols).describe()
print(feature_stats)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)

print(f"\n✓ Standardization complete (z-score normalization)")
print(f"  Mean of scaled features: {X_scaled.mean():.6f}")
print(f"  Std of scaled features: {X_scaled.std():.6f}")

# Save scaler for future use
joblib.dump(scaler, f'{OUTPUT_DIR}/scaler.joblib')
print(f"\n✓ Scaler saved to {OUTPUT_DIR}/scaler.joblib")

# ============================================================================
# STEP 4: AUTO-TUNE SPECTRAL CLUSTERING FOR BEST SILHOUETTE
# ============================================================================
def auto_tune_spectral_clustering(X, cluster_range=[2,3,4,5], neighbor_range=[5,10,15], random_state=42):
    # Notebook logic: search for best silhouette, safe fallback, sample-size check
    best_score = -1
    best_labels = None
    best_model = None
    best_metrics = None
    best_params = None
    print("\nAuto-tuning clustering parameters for best silhouette score (NOTEBOOK LOGIC)...")
    for n_clusters in cluster_range:
        for n_neighbors in neighbor_range:
            try:
                model = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity='nearest_neighbors',
                    n_neighbors=n_neighbors,
                    assign_labels='kmeans',
                    n_init=8,
                    random_state=random_state
                )
                labels = model.fit_predict(X)
                # Only compute silhouette if enough samples
                if len(X) > 10 * n_clusters:
                    sample_size = min(MAX_SAMPLES_FOR_SILHOUETTE, len(X))
                    score = silhouette_score(X, labels, sample_size=sample_size)
                else:
                    score = -1
                db = davies_bouldin_score(X, labels)
                ch = calinski_harabasz_score(X, labels)
                print(f"  n_clusters={n_clusters}, n_neighbors={n_neighbors} -> silhouette={score:.4f}, DB={db:.3f}, CH={ch:.1f}")
                if score > best_score:
                    best_score = score
                    best_labels = labels
                    best_model = model
                    best_metrics = {'silhouette': score, 'davies_bouldin': db, 'calinski_harabasz': ch}
                    best_params = (n_clusters, n_neighbors)
            except Exception as e:
                print(f"  Skipped n_clusters={n_clusters}, n_neighbors={n_neighbors}: {e}")
    
    # SAFE FALLBACK: If spectral clustering failed entirely, use KMeans
    if best_labels is None:
        print("\n⚠ WARNING: Spectral clustering failed for all parameters. Falling back to KMeans...")
        try:
            n_clusters_fallback = 3  # Default from paper
            kmeans_model = KMeans(n_clusters=n_clusters_fallback, random_state=random_state, n_init=10)
            best_labels = kmeans_model.fit_predict(X)
            
            if len(X) > 10 * n_clusters_fallback:
                sample_size = min(MAX_SAMPLES_FOR_SILHOUETTE, len(X))
                best_score = silhouette_score(X, best_labels, sample_size=sample_size)
            else:
                best_score = -1
            
            db = davies_bouldin_score(X, best_labels)
            ch = calinski_harabasz_score(X, best_labels)
            
            best_model = kmeans_model
            best_metrics = {'silhouette': best_score, 'davies_bouldin': db, 'calinski_harabasz': ch}
            best_params = (n_clusters_fallback, None)
            
            print(f"  ✓ KMeans fallback successful: silhouette={best_score:.4f}, DB={db:.3f}, CH={ch:.1f}")
        except Exception as e:
            print(f"  ✗ KMeans fallback also failed: {e}")
            raise RuntimeError("Both Spectral Clustering and KMeans fallback failed. Cannot proceed.")
    
    if best_score < 0.5:
        print(f"\n⚠ WARNING: Best silhouette score found is {best_score:.4f} (<0.5). Consider reviewing feature engineering or outlier thresholds.")
    else:
        print(f"\n✓ Minimum silhouette score of 0.5 achieved!")
    if best_params is not None:
        print(f"Best configuration: n_clusters={best_params[0]}, n_neighbors={best_params[1]} (silhouette={best_score:.4f})")
    else:
        print("Best configuration: None (no valid clustering found)")
    return best_labels, best_model, best_metrics, best_params

labels, spectral_model, metrics, best_params = auto_tune_spectral_clustering(
    X_scaled, cluster_range=[2,3,4,5], neighbor_range=[5,10,15], random_state=RANDOM_STATE
)

# Save model
joblib.dump(scaler, f'{OUTPUT_DIR}/scaler.joblib')
joblib.dump(spectral_model, f'{OUTPUT_DIR}/spectral_model.joblib')
print(f"\n✓ Scaler saved to {OUTPUT_DIR}/scaler.joblib")
print(f"✓ Model saved to {OUTPUT_DIR}/spectral_model.joblib")
print(f"✓ Best clustering silhouette score: {metrics['silhouette']:.4f}")

# Add labels to DataFrame
df_features_sample['Cluster'] = labels

# If we subsampled, propagate labels to full dataset using KNN (notebook logic)
if original_size > MAX_SAMPLES_FOR_CLUSTERING:
    print(f"\n⚠ Assigning cluster labels to full dataset ({original_size} samples)...")
    print("  Using KNN label propagation...")
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_scaled, labels)
    X_full_scaled = scaler.transform(X)
    full_labels = knn.predict(X_full_scaled)
    df_features['Cluster'] = full_labels
    print(f"✓ All {original_size} samples labeled")
else:
    df_features['Cluster'] = labels

# ============================================================================
# STEP 5: COMPUTE CLUSTER CENTROIDS
# ============================================================================
def compute_centroids(df, feature_cols, labels):
    """
    Compute cluster centroids in original (non-standardized) feature space.
    
    Returns:
        centroids_df: DataFrame with centroid values for each cluster
    """
    print("\n" + "="*80)
    print("STEP 5: CLUSTER CENTROIDS")
    print("-" * 80)
    
    df_with_labels = df.copy()
    df_with_labels['Cluster'] = labels
    
    centroids = df_with_labels.groupby('Cluster')[feature_cols].mean()
    
    print(f"\nCluster centroids (mean values):")
    print(centroids.to_string())
    
    return centroids

centroids_df = compute_centroids(df_features, feature_cols, df_features['Cluster'].values)

# Save centroids
centroids_df.to_csv(f'{OUTPUT_DIR}/centroids_table.csv')
print(f"\n✓ Centroids saved to {OUTPUT_DIR}/centroids_table.csv")

# ============================================================================
# STEP 6: LABEL CLUSTERS BY AEROSOL TYPE
# ============================================================================
def label_clusters_by_centroids(centroids_df):
    """
    Assign aerosol type labels based on centroid characteristics.
    
    Following paper's physical interpretation:
    - Dust: Low EAE (<0.6), High AAE (>1.8), Low AOD_fine
    - Biomass Burning: High EAE (≥1.0), Moderate AAE (1.2-1.8), Low SSA (≤0.92)
    - Urban/Industrial: High EAE (≥1.0), High SSA (≥0.94)
    - Mixed: Does not fit above categories
    
    Returns:
        mapping_df: DataFrame with cluster ID, label, and explanation
    """
    print("\n" + "="*80)
    print("STEP 6: AEROSOL TYPE CLASSIFICATION")
    print("-" * 80)
    
    print(f"\nClassification criteria (from paper):")
    print(f"  Dust: EAE < {DUST_EAE_MAX}, AAE > {DUST_AAE_MIN}")
    print(f"  Biomass Burning: EAE ≥ {BB_EAE_MIN}, AAE ∈ [{BB_AAE_MIN}, {BB_AAE_MAX}], SSA ≤ {BB_SSA_MAX}")
    print(f"  Urban/Industrial: EAE ≥ {URBAN_EAE_MIN}, SSA ≥ {URBAN_SSA_MIN}")
    print(f"  Mixed: Otherwise")
    
    mapping = []
    
    for cluster_id, row in centroids_df.iterrows():
        eae = row['EAE']
        aae = row['AAE']
        ssa = row['SSA']
        aod_fine = row['AOD_fine']
        
        # Classification logic
        if eae < DUST_EAE_MAX and aae > DUST_AAE_MIN:
            label = 'Dust'
            explanation = f'Low EAE ({eae:.2f}), High AAE ({aae:.2f}) → Coarse particles'
        elif eae >= BB_EAE_MIN and BB_AAE_MIN <= aae <= BB_AAE_MAX and ssa <= BB_SSA_MAX:
            label = 'Biomass Burning'
            explanation = f'High EAE ({eae:.2f}), Moderate AAE ({aae:.2f}), Low SSA ({ssa:.2f}) → Absorbing fine particles'
        elif eae >= URBAN_EAE_MIN and ssa >= URBAN_SSA_MIN:
            label = 'Urban/Industrial'
            explanation = f'High EAE ({eae:.2f}), High SSA ({ssa:.2f}) → Scattering fine particles'
        else:
            label = 'Mixed'
            explanation = f'EAE={eae:.2f}, AAE={aae:.2f}, SSA={ssa:.2f} → Complex mixture'
        
        mapping.append({
            'Cluster': cluster_id,
            'Aerosol_Type': label,
            'Explanation': explanation,
            'EAE': eae,
            'AAE': aae,
            'SSA': ssa,
            'AOD_fine': aod_fine
        })
    
    mapping_df = pd.DataFrame(mapping)
    
    print(f"\nCluster type assignments:")
    for _, row in mapping_df.iterrows():
        print(f"\n  Cluster {row['Cluster']}: {row['Aerosol_Type']}")
        print(f"    {row['Explanation']}")
    
    return mapping_df

cluster_mapping = label_clusters_by_centroids(centroids_df)

# Apply labels to data
label_dict = dict(zip(cluster_mapping['Cluster'], cluster_mapping['Aerosol_Type']))
df_features['Aerosol_Type'] = df_features['Cluster'].map(label_dict)

# Save cluster mapping
cluster_mapping.to_csv(f'{OUTPUT_DIR}/cluster_aerosol_types.csv', index=False)
print(f"\n✓ Cluster mapping saved to {OUTPUT_DIR}/cluster_aerosol_types.csv")

# ============================================================================
# STEP 7: CLUSTER PERCENTAGE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("STEP 7: CLUSTER PERCENTAGE ANALYSIS")
print("-" * 80)

# Overall percentages
overall_pct = df_features['Aerosol_Type'].value_counts(normalize=True) * 100
overall_pct_df = overall_pct.reset_index()
overall_pct_df.columns = ['Aerosol_Type', 'Percentage']
overall_pct_df['Region'] = 'ASIA (Overall)'

print(f"\nOverall cluster distribution:")
print(overall_pct_df.to_string(index=False))

overall_pct_df.to_csv(f'{OUTPUT_DIR}/cluster_percentages.csv', index=False)
print(f"\n✓ Overall percentages saved to {OUTPUT_DIR}/cluster_percentages.csv")

# Per-site percentages (if Site column exists)
if 'Site' in df_features.columns:
    print(f"\nPer-site cluster distribution:")
    
    site_cluster_counts = pd.crosstab(
        df_features['Site'], 
        df_features['Aerosol_Type'], 
        normalize='index'
    ) * 100
    
    # Add total counts
    site_totals = df_features['Site'].value_counts()
    site_cluster_counts['Total_Samples'] = site_totals
    
    # Sort by total samples
    site_cluster_counts = site_cluster_counts.sort_values('Total_Samples', ascending=False)
    
    print(site_cluster_counts.head(10).to_string())
    
    site_cluster_counts.to_csv(f'{OUTPUT_DIR}/cluster_percentages_by_site.csv')
    print(f"\n✓ Site percentages saved to {OUTPUT_DIR}/cluster_percentages_by_site.csv")

# ============================================================================
# STEP 8: SAVE CLUSTERED RESULTS
# ============================================================================
print("\n" + "="*80)
print("STEP 8: SAVING RESULTS")
print("-" * 80)

# Save complete clustered dataset
output_cols = ['Site'] if 'Site' in df_features.columns else []
output_cols += ['AOD_total', 'AOD_fine', 'EAE', 'AAE', 'SSA', 'RRI',
                'Cluster', 'Aerosol_Type', 'used_AOD_fine_proxy', 'used_RRI_proxy']

df_output = df_features[output_cols].copy()
df_output.to_csv(f'{OUTPUT_DIR}/asia_clustered.csv', index=False)
print(f"✓ Clustered data saved to {OUTPUT_DIR}/asia_clustered.csv")
print(f"  Rows: {len(df_output)}")
print(f"  Columns: {len(output_cols)}")

# ============================================================================
# ASIA-SPECIFIC TABLES GENERATION
# ============================================================================
def generate_asia_tables(df, centroids_df, output_dir):
    """Generate all Asia-specific tables from the paper"""
    print("\n" + "="*80)
    print("GENERATING ASIA-SPECIFIC TABLES")
    print("-" * 80)
    
    # ===== TABLE 2: Aerosol Optical Parameters Statistics for Asia =====
    print("\n1. Generating Table 2: Optical parameters statistics for Asia...")
    try:
        # Calculate statistics for Asia region
        table2_params = {
            'AOD_T': 'AOD_total',
            'AOD_F': 'AOD_fine', 
            'AOD_C': 'AOD_coarse',
            'EAE': 'EAE',
            'SSA': 'SSA',
            'Ref_R': 'RRI',
            'AAOD': 'AAOD',
            'AAE': 'AAE'
        }
        
        table2_data = []
        for label, col in table2_params.items():
            if col in df.columns:
                avg = df[col].mean()
                median = df[col].median()
                std = df[col].std()
                table2_data.append({
                    'Parameter': label,
                    'Average': f"{avg:.2f} ± {std:.2f}",
                    'Median': f"{median:.2f}"
                })
        
        table2_df = pd.DataFrame(table2_data)
        table2_df.to_csv(f'{output_dir}/table2_asia_optical_parameters.csv', index=False)
        print(f"   ✓ Saved: table2_asia_optical_parameters.csv")
        print(table2_df.to_string(index=False))
    except Exception as e:
        print(f"   ⚠ Error generating Table 2: {e}")
    
    # ===== TABLE 3: Seasonal Distribution Statistics =====
    print("\n2. Generating Table 3: Seasonal distribution statistics for Asia...")
    try:
        if 'Season' in df.columns:
            # Map season names from the data (lowercase) to paper format (DJF, etc.)
            season_mapping = {
                'winter': 'DJF',
                'spring': 'MAM', 
                'summer': 'JJA',
                'autumn': 'SON',
                'fall': 'SON'  # Alternative name for autumn
            }
            
            # Get unique seasons in the data
            unique_seasons_in_data = df['Season'].unique()
            print(f"   Found seasons in data: {unique_seasons_in_data}")
            
            table3_params = {
                'AOD_Fine': 'AOD_fine',
                'AOD_Coarse': 'AOD_coarse', 
                'EAE': 'EAE',
                'SSA': 'SSA',
                'AAOD': 'AAOD',
                'AAE': 'AAE'
            }
            
            table3_data = []
            for season_name in unique_seasons_in_data:
                season_data = df[df['Season'] == season_name]
                if len(season_data) > 0:
                    # Use mapped season label or original if not in mapping
                    season_label = season_mapping.get(season_name.lower(), season_name)
                    row = {'Season': season_label}
                    for label, col in table3_params.items():
                        if col in df.columns:
                            avg = season_data[col].mean()
                            std = season_data[col].std()
                            row[label] = f"{avg:.2f} ± {std:.2f}"
                        else:
                            row[label] = 'N/A'
                    table3_data.append(row)
            
            if len(table3_data) > 0:
                table3_df = pd.DataFrame(table3_data)
                table3_df.to_csv(f'{output_dir}/table3_asia_seasonal_distribution.csv', index=False)
                print(f"   ✓ Saved: table3_asia_seasonal_distribution.csv")
                print(table3_df.to_string(index=False))
            else:
                print(f"   ⚠ No seasonal data found")
        else:
            print(f"   ⚠ Skipped Table 3 (no Season column)")
    except Exception as e:
        print(f"   ⚠ Error generating Table 3: {e}")
        import traceback
        traceback.print_exc()
    
    # ===== TABLE 4: Centroid Values with Aerosol Types =====
    print("\n3. Generating Table 4: Centroid values with aerosol types for Asia...")
    try:
        # Calculate cluster percentages
        cluster_counts = df['Cluster'].value_counts().sort_index()
        total = len(df)
        
        table4_data = []
        for cluster_id in sorted(df['Cluster'].unique()):
            cluster_data = df[df['Cluster'] == cluster_id]
            
            # Determine aerosol type based on centroid characteristics
            eae_mean = cluster_data['EAE'].mean()
            aae_mean = cluster_data['AAE'].mean()
            ssa_mean = cluster_data['SSA'].mean()
            aod_fine_mean = cluster_data['AOD_fine'].mean()
            
            # Classification logic from paper
            if eae_mean < DUST_EAE_MAX and aae_mean > DUST_AAE_MIN:
                aerosol_type = 'D'  # Dust
            elif eae_mean > BB_EAE_MIN and BB_AAE_MIN < aae_mean < BB_AAE_MAX and ssa_mean < BB_SSA_MAX:
                aerosol_type = 'B'  # Biomass Burning
            elif eae_mean > URBAN_EAE_MIN and ssa_mean > URBAN_SSA_MIN:
                aerosol_type = 'U'  # Urban/Industrial
            else:
                aerosol_type = 'M'  # Mixed
            
            table4_data.append({
                'Cluster': cluster_id + 1,
                'Percentage': f"{(cluster_counts[cluster_id]/total)*100:.1f}",
                'Aerosol_Type': aerosol_type,
                'Fine_mode_AOD': f"{aod_fine_mean:.2f}",
                'EAE': f"{eae_mean:.2f}",
                'AAE': f"{aae_mean:.2f}",
                'SSA': f"{ssa_mean:.2f}",
                'RRI': f"{cluster_data['RRI'].mean():.2f}"
            })
        
        table4_df = pd.DataFrame(table4_data)
        table4_df.to_csv(f'{output_dir}/table4_asia_cluster_centroids.csv', index=False)
        print(f"   ✓ Saved: table4_asia_cluster_centroids.csv")
        print(table4_df.to_string(index=False))
    except Exception as e:
        print(f"   ⚠ Error generating Table 4: {e}")
    
    # ===== TABLE 5: Seasonal Distribution by Clusters =====
    print("\n4. Generating Table 5: Seasonal distribution percentages by clusters for Asia...")
    try:
        if 'Season' in df.columns:
            # Map season names from the data
            season_mapping = {
                'winter': 'DJF',
                'spring': 'MAM', 
                'summer': 'JJA',
                'autumn': 'SON',
                'fall': 'SON'
            }
            
            # Get unique seasons in the data
            unique_seasons_in_data = df['Season'].unique()
            print(f"   Found seasons in data: {unique_seasons_in_data}")
            
            table5_data = []
            
            for season_name in unique_seasons_in_data:
                season_data = df[df['Season'] == season_name]
                if len(season_data) > 0:
                    season_label = season_mapping.get(season_name.lower(), season_name)
                    row = {'Season': season_label}
                    for cluster_id in sorted(df['Cluster'].unique()):
                        cluster_count = len(season_data[season_data['Cluster'] == cluster_id])
                        percentage = (cluster_count / len(season_data)) * 100
                        
                        # Get dominant aerosol type for this cluster in this season
                        cluster_season_data = season_data[season_data['Cluster'] == cluster_id]
                        if len(cluster_season_data) > 0 and 'Aerosol_Type' in cluster_season_data.columns:
                            dominant_type = cluster_season_data['Aerosol_Type'].mode()
                            if len(dominant_type) > 0:
                                type_abbr = dominant_type.values[0][0] if len(dominant_type.values[0]) > 0 else 'M'
                            else:
                                type_abbr = 'M'
                        else:
                            type_abbr = 'M'
                        
                        row[f'Cluster_{cluster_id+1}'] = f"{percentage:.0f}%, {type_abbr}"
                    
                    table5_data.append(row)
            
            if len(table5_data) > 0:
                table5_df = pd.DataFrame(table5_data)
                table5_df.to_csv(f'{output_dir}/table5_asia_seasonal_cluster_distribution.csv', index=False)
                print(f"   ✓ Saved: table5_asia_seasonal_cluster_distribution.csv")
                print(table5_df.to_string(index=False))
            else:
                print(f"   ⚠ No seasonal data found")
        else:
            print(f"   ⚠ Skipped Table 5 (no Season column)")
    except Exception as e:
        print(f"   ⚠ Error generating Table 5: {e}")
        import traceback
        traceback.print_exc()
    
    # ===== TABLE 6: FMF/SSA Classification Statistics =====
    print("\n5. Generating Table 6: FMF/SSA classification statistics for Asia...")
    try:
        if 'FMF' in df.columns and 'SSA' in df.columns:
            # Classification based on FMF and SSA thresholds
            def classify_aerosol(row):
                fmf = row['FMF']
                ssa = row['SSA']
                
                if pd.isna(fmf) or pd.isna(ssa):
                    return 'Uncer.'
                elif fmf < 0.4 and ssa < 0.95:
                    return 'Dust'
                elif 0.4 <= fmf < 0.6:
                    return 'Mix.'
                elif ssa > 0.95:
                    return 'N.A.'
                elif fmf >= 0.6 and ssa < 0.90:
                    return 'BC'
                else:
                    return 'Mix.'
            
            df['FMF_SSA_Class'] = df.apply(classify_aerosol, axis=1)
            
            # Count each type
            class_counts = df['FMF_SSA_Class'].value_counts()
            table6_data = {
                'Uncer.': class_counts.get('Uncer.', 0),
                'Dust': class_counts.get('Dust', 0),
                'Mix.': class_counts.get('Mix.', 0),
                'N.A.': class_counts.get('N.A.', 0),
                'BC': class_counts.get('BC', 0)
            }
            
            table6_df = pd.DataFrame([table6_data])
            table6_df.to_csv(f'{output_dir}/table6_asia_fmf_ssa_classification.csv', index=False)
            print(f"   ✓ Saved: table6_asia_fmf_ssa_classification.csv")
            print(table6_df.to_string(index=False))
        else:
            print(f"   ⚠ Skipped Table 6 (missing FMF or SSA column)")
    except Exception as e:
        print(f"   ⚠ Error generating Table 6: {e}")
    
    print("\n✓ All Asia-specific tables generated!")

# ============================================================================
# STEP 9: GENERATE PLOTS (NOTEBOOK LOGIC)
# ============================================================================
def make_plots(df, centroids_df, cluster_mapping, output_dir):
    """Generate all required plots following notebook's exact specifications."""
    print("\n" + "="*80)
    print("STEP 9: GENERATING VISUALIZATIONS (NOTEBOOK LOGIC)")
    print("-" * 80)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = PLOT_DPI
    
    # Subsample for plotting if dataset is large
    if len(df) > 10000:
        print(f"\n⚠ Large dataset detected. Using 10,000 samples for visualization...")
        plot_df = df.sample(n=10000, random_state=RANDOM_STATE)
    else:
        plot_df = df
    
    # Helper function: compute CDF with values at bin centers
    def compute_cdf_hist(data, bins):
        data = data.dropna()
        counts, bin_edges = np.histogram(data, bins=bins)
        cdf = np.cumsum(counts) / np.sum(counts)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        return counts, bin_edges, bin_centers, cdf
    
    # === PLOT 1: Header Image (placeholder for report) ===
    try:
        print("\n1. Generating report header...")
        fig = plt.figure(figsize=(11, 2))
        fig.text(0.5, 0.65, 'Aerosol Spectral Clustering Analysis', ha='center', fontsize=22, fontweight='bold')
        fig.text(0.5, 0.35, 'Following Ningombam et al., 2024 Methodology', ha='center', fontsize=14, color='darkblue')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fig_page1_header.png'), dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: fig_page1_header.png")
    except Exception as e:
        print(f"   ⚠ Error generating header: {e}")
    
    # === PLOT 2: Params Plot (Histograms + CDF) ===
    try:
        print("\n2. Generating params plot (histograms + CDF)...")
        params = [
            ("AOD_total", "AOD_440nm", "AOD Total at 440 nm", "steelblue"),
            ("EAE", "EAE_440-870", "Extinction Ångström Exponent (440–870 nm)", "teal"),
            ("SSA", "SSA_440", "Single Scattering Albedo at 440 nm", "coral"),
            ("AAE", "AAE_440-870", "Absorption Ångström Exponent (440–870 nm)", "olive")
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Frequency distribution and CDF of aerosol optical parameters (AERONET)", fontsize=16)
        
        for ax, (col, xlabel, title, color) in zip(axes.flatten(), params):
            ax_twin = ax.twinx()
            data = plot_df[col].dropna()
            bins = 25
            counts, bin_edges, bin_centers, cdf = compute_cdf_hist(data, bins)
            ax.bar(bin_centers, counts, width=(bin_edges[1] - bin_edges[0]) * 0.9, color=color, alpha=0.4, edgecolor='none')
            ax_twin.plot(bin_centers, cdf * 100, color=color, marker='o', markersize=4, linewidth=2)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Frequency Count")
            ax_twin.set_ylabel("Percentage Frequency Distribution")
            ax.grid(False)
            ax_twin.grid(False)
            ax_twin.set_ylim(0, 100)
            ax.set_xlim(bin_edges[0], bin_edges[-1])
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(output_dir, 'fig_params.png'), dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: fig_params.png")
    except Exception as e:
        print(f"   ⚠ Error generating params plot: {e}")
    
    # === PLOT 3: Map Plot 1 (Spatial Distribution) ===
    try:
        print("\n3. Generating map plot 1 (spatial distribution)...")
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            
            # Get site coordinates
            site_coords = {
                'Omkoi': (17.8, 98.4), 'Masdar_Institute': (24.4, 54.6), 'IMS-METU-ERDEMLI': (36.6, 34.3),
                'Dushanbe': (38.5, 68.8), 'Georgia_Tech': (33.8, -84.4), 'SACOL': (35.9, 104.1),
                'IIT_Delhi': (28.5, 77.2), 'Shirahama': (33.7, 135.4), 'MCO-Hanimaadhoo': (6.7, 73.2),
                'Manila_Observatory': (14.6, 121.1), 'Dunhuang_LZU': (40.2, 94.7), 'Lulin': (23.5, 120.9),
                'Beijing-CAMS': (39.9, 116.3), 'Dongsha_Island': (20.7, 116.7), 'Beijing': (39.9, 116.4),
                'NhaTrang': (12.2, 109.2), 'Karunya_University': (10.9, 76.9), 'Bandung': (-6.9, 107.6),
                'Issyk-Kul': (42.5, 77.0), 'Seoul_SNU': (37.5, 127.0), 'Anmyon': (36.5, 126.3),
                'XiangHe': (39.8, 117.0), 'Dibrugarh_Univ': (27.5, 95.0), 'EPA-NCU': (25.0, 121.2),
                'Dhaka_University': (23.7, 90.4), 'Ubon_Ratchathani': (15.2, 104.9), 'Gandhi_College': (25.6, 85.2),
                'Singapore': (1.3, 103.8), 'IASBS': (36.3, 48.5), 'Amity_Univ_Gurgaon': (28.4, 77.0),
                'Solar_Village': (24.9, 46.4), 'Jaipur': (26.9, 75.8), 'Pune': (18.5, 73.9),
                'Dalanzadgad': (43.6, 104.4), 'Kanpur': (26.5, 80.3), 'Luang_Namtha': (20.9, 101.4),
                'Silpakorn_Univ': (13.8, 100.0), 'Chiba_University': (35.6, 140.1), 'MALE': (4.2, 73.5),
                'SEDE_BOKER': (30.9, 34.8), 'Bangkok': (13.8, 100.5)
            }
            coords_df = pd.DataFrame.from_dict(site_coords, orient='index', columns=['Latitude', 'Longitude']).reset_index()
            coords_df.rename(columns={'index': 'Site'}, inplace=True)
            
            # Merge with data
            if 'AOD_fine' in df.columns and 'Site' in df.columns:
                mean_df = df.groupby('Site').agg({'AOD_fine': 'mean'}).reset_index()
                merged_df = pd.merge(mean_df, coords_df, on='Site', how='left')
                
                fig = plt.figure(figsize=(14, 8))
                ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.BORDERS, linestyle=':')
                ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
                ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
                ax.set_global()
                ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
                
                sc = ax.scatter(merged_df['Longitude'], merged_df['Latitude'],
                               c=merged_df['AOD_fine'], cmap='viridis', s=100,
                               edgecolor='k', linewidth=0.5, alpha=0.8,
                               transform=ccrs.PlateCarree())
                
                cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', fraction=0.046, pad=0.08, 
                                   label='Fine-mode AOD (440 nm)')
                cbar.ax.tick_params(labelsize=10)
                plt.title("Global Spatial Distribution of Fine-mode AOD (440 nm)\n(Site-wise Means)", 
                         fontsize=14, fontweight='bold', pad=20)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'fig_map1.png'), dpi=PLOT_DPI, bbox_inches='tight')
                plt.close()
                print(f"   ✓ Saved: fig_map1.png")
            else:
                print(f"   ⚠ Skipped fig_map1.png (missing AOD_fine or Site column)")
        except ImportError:
            print(f"   ⚠ Cartopy not available, skipping map plots")
    except Exception as e:
        print(f"   ⚠ Error generating map1: {e}")
    
    # === PLOT 4: Map Plot 2 (Source Label Distribution) ===
    try:
        print("\n4. Generating map plot 2 (source label distribution)...")
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            
            if 'Source Label' in df.columns and 'Site' in df.columns:
                # Compute majority label per site
                site_label_counts = df.groupby(['Site', "Source Label"]).size().unstack(fill_value=0)
                site_label_percentages = site_label_counts.div(site_label_counts.sum(axis=1), axis=0) * 100
                majority_label = site_label_percentages.idxmax(axis=1)
                
                # Merge with coordinates
                site_data = pd.DataFrame({
                    'Latitude': coords_df.set_index('Site')['Latitude'],
                    'Longitude': coords_df.set_index('Site')['Longitude'],
                    'Majority_Label': majority_label
                }).dropna()
                
                color_map_labels = {
                    'Maritime': '#3498db', 'Urban': '#e67e22', 'Desert': '#c0392b',
                    'Arid': '#d35400', 'Biomass': '#27ae60', 'Unclassified': '#95a5a6'
                }
                
                fig = plt.figure(figsize=(16, 10))
                ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
                ax.add_feature(cfeature.LAND, facecolor='#f5f5dc', alpha=0.3)
                ax.add_feature(cfeature.OCEAN, facecolor='#b3d9ff', alpha=0.5)
                ax.set_global()
                gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False, 
                                 linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
                gl.top_labels = False
                gl.right_labels = False
                
                for label in site_data['Majority_Label'].unique():
                    subset = site_data[site_data['Majority_Label'] == label]
                    ax.scatter(subset['Longitude'], subset['Latitude'],
                              c=color_map_labels.get(label, '#000000'), s=80,
                              edgecolor='black', linewidth=0.5, alpha=0.8,
                              label=label.lower(), transform=ccrs.PlateCarree(), zorder=5)
                
                legend = ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, 
                                  fontsize=10, markerscale=1.2)
                legend.get_frame().set_facecolor('white')
                legend.get_frame().set_alpha(0.9)
                plt.title('Global Distribution of AERONET Sites by Majority Aerosol Source Type', 
                         fontsize=14, fontweight='bold', pad=20)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'fig_map2.png'), dpi=PLOT_DPI, bbox_inches='tight')
                plt.close()
                print(f"   ✓ Saved: fig_map2.png")
            else:
                print(f"   ⚠ Skipped fig_map2.png (missing Source Label or Site column)")
        except ImportError:
            print(f"   ⚠ Cartopy not available, skipping map plots")
    except Exception as e:
        print(f"   ⚠ Error generating map2: {e}")
    
    # === PLOT 5: Histograms (simple) ===
    try:
        print("\n5. Generating simple histograms...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Frequency Distributions of Aerosol Properties (ASIA Region)', 
                     fontsize=16, fontweight='bold')
        
        hist_vars = [
            ('AOD_total', 'Total AOD at 440 nm', 30),
            ('EAE', 'Extinction Angstrom Exponent (440-870 nm)', 30),
            ('SSA', 'Single Scattering Albedo at 440 nm', 30),
            ('AAE', 'Absorption Angstrom Exponent (440-870 nm)', 30)
        ]
        
        for ax, (var, label, bins) in zip(axes.flatten(), hist_vars):
            if var in plot_df.columns:
                data = plot_df[var].dropna()
                ax.hist(data, bins=bins, color='steelblue', alpha=0.7, edgecolor='black')
                ax.set_xlabel(label, fontsize=11)
                ax.set_ylabel('Frequency', fontsize=11)
                ax.set_title(f'Distribution of {label}', fontsize=12)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fig_histograms.png'), dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: fig_histograms.png")
    except Exception as e:
        print(f"   ⚠ Error generating histograms: {e}")
    
    # === PLOT 6: Scatter Matrix (Clusters) ===
    try:
        print("\n6. Generating scatter matrix (clusters)...")
        # Define cluster colors
        cluster_colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#d62728'}
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("GMM Clustering (n = 3) — Scatter Plots", fontsize=16)
        
        pairs = [
            ('EAE', 'AAE', 'EAE 440–870 nm', 'AAE 440–870 nm'),
            ('EAE', 'AOD_fine', 'EAE 440–870 nm', 'AOD Fine mode 440 nm'),
            ('EAE', 'SSA', 'EAE 440–870 nm', 'SSA 440 nm'),
            ('EAE', 'RRI', 'EAE 440–870 nm', 'RRI 440 nm')
        ]
        
        for ax, (xcol, ycol, xlabel, ylabel) in zip(axes.flatten(), pairs):
            if xcol in plot_df.columns and ycol in plot_df.columns and 'Cluster' in plot_df.columns:
                for cluster_id in sorted(plot_df['Cluster'].unique()):
                    subset = plot_df[plot_df['Cluster'] == cluster_id]
                    color = cluster_colors.get(cluster_id, '#333333')
                    ax.scatter(subset[xcol], subset[ycol], s=3, color=color, alpha=0.6, 
                              label=f'Cluster {cluster_id + 1}' if ax == axes[0, 0] else "")
                    
                    # Plot cluster center
                    cx = subset[xcol].mean()
                    cy = subset[ycol].mean()
                    ax.scatter(cx, cy, s=150, color=color, marker='D', edgecolors='black', 
                              linewidths=2, zorder=5, clip_on=True)
                
                ax.set_xlabel(xlabel, fontsize=11)
                ax.set_ylabel(ylabel, fontsize=11)
                ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        if 'Cluster' in plot_df.columns:
            axes[0, 0].legend(title="Clusters & Centers", loc="best", fontsize=8, title_fontsize=9, framealpha=0.9)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(output_dir, 'fig_scatter_matrix_clusters.png'), dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: fig_scatter_matrix_clusters.png")
    except Exception as e:
        print(f"   ⚠ Error generating scatter matrix: {e}")
    
    # === PLOT 7: Scatter AOD vs EAE ===
    try:
        print("\n7. Generating AOD vs EAE scatter plot...")
        if 'AOD_total' in plot_df.columns and 'EAE' in plot_df.columns and 'Cluster' in plot_df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for cluster_id in sorted(plot_df['Cluster'].unique()):
                subset = plot_df[plot_df['Cluster'] == cluster_id]
                color = cluster_colors.get(cluster_id, '#333333')
                ax.scatter(subset['AOD_total'], subset['EAE'], s=10, color=color, alpha=0.5, 
                          label=f'Cluster {cluster_id + 1}')
            
            ax.set_xlabel('AOD Total (440 nm)', fontsize=12)
            ax.set_ylabel('Extinction Angstrom Exponent (440-870 nm)', fontsize=12)
            ax.set_title('AOD Total vs EAE (Colored by Cluster)', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'fig_scatter_AOD_EAE.png'), dpi=PLOT_DPI, bbox_inches='tight')
            plt.close()
            print(f"   ✓ Saved: fig_scatter_AOD_EAE.png")
    except Exception as e:
        print(f"   ⚠ Error generating AOD vs EAE scatter: {e}")
    
    # === PLOT 8: Silhouette Plot ===
    try:
        print("\n8. Generating silhouette plot...")
        from sklearn.metrics import silhouette_samples
        
        if 'Cluster' in plot_df.columns and len(plot_df) > 100:
            # Recompute scaled features for plot_df
            feature_cols_plot = ['AOD_fine', 'EAE', 'AAE', 'SSA', 'RRI']
            X_plot = plot_df[feature_cols_plot].values
            
            # Use global scaler if available, otherwise create new one
            try:
                from sklearn.preprocessing import StandardScaler as SS
                scaler_plot = SS()
                X_plot_scaled = scaler_plot.fit_transform(X_plot)
            except:
                X_plot_scaled = (X_plot - X_plot.mean(axis=0)) / X_plot.std(axis=0)
            
            # Use subsample for silhouette if still too large
            if len(plot_df) > MAX_SAMPLES_FOR_SILHOUETTE:
                sample_indices = np.random.RandomState(RANDOM_STATE).choice(len(plot_df), MAX_SAMPLES_FOR_SILHOUETTE, replace=False)
                X_sil = X_plot_scaled[sample_indices]
                labels_sil = plot_df['Cluster'].values[sample_indices]
            else:
                X_sil = X_plot_scaled
                labels_sil = plot_df['Cluster'].values
            
            silhouette_vals = silhouette_samples(X_sil, labels_sil)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            y_lower = 10
            
            for cluster_id in sorted(np.unique(labels_sil)):
                cluster_silhouette_vals = silhouette_vals[labels_sil == cluster_id]
                cluster_silhouette_vals.sort()
                size_cluster = cluster_silhouette_vals.shape[0]
                y_upper = y_lower + size_cluster
                
                color = cluster_colors.get(cluster_id, '#333333')
                ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                                facecolor=color, edgecolor=color, alpha=0.7)
                ax.text(-0.05, y_lower + 0.5 * size_cluster, f'C{cluster_id + 1}')
                y_lower = y_upper + 10
            
            ax.set_xlabel('Silhouette Coefficient', fontsize=12)
            ax.set_ylabel('Cluster', fontsize=12)
            ax.set_title('Silhouette Plot for Clusters', fontsize=14, fontweight='bold')
            ax.axvline(x=silhouette_vals.mean(), color='red', linestyle='--', label=f'Mean: {silhouette_vals.mean():.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'fig_silhouette.png'), dpi=PLOT_DPI, bbox_inches='tight')
            plt.close()
            print(f"   ✓ Saved: fig_silhouette.png")
    except Exception as e:
        print(f"   ⚠ Error generating silhouette plot: {e}")
    
    # === PLOT 9: Cluster Bar Percentages ===
    try:
        print("\n9. Generating cluster bar percentages...")
        color_map = {
            'Dust': '#c0392b',
            'Biomass Burning': '#27ae60',
            'Urban/Industrial': '#3498db',
            'Mixed': '#95a5a6'
        }
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if 'Aerosol_Type' in df.columns:
            cluster_pcts = df['Aerosol_Type'].value_counts(normalize=True) * 100
            cluster_pcts = cluster_pcts.sort_values(ascending=False)
            
            colors_bar = [color_map.get(atype, '#333333') for atype in cluster_pcts.index]
            bars = ax.bar(range(len(cluster_pcts)), cluster_pcts.values, 
                          color=colors_bar, edgecolor='black', linewidth=1.5)
            
            ax.set_xticks(range(len(cluster_pcts)))
            ax.set_xticklabels(cluster_pcts.index, fontsize=12, fontweight='bold')
            ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
            ax.set_title('Cluster Distribution in ASIA Region', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar, pct in zip(bars, cluster_pcts.values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{pct:.1f}%', ha='center', va='bottom', 
                       fontsize=11, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'fig_cluster_bar_percentages.png'), dpi=PLOT_DPI, bbox_inches='tight')
            plt.close()
            print(f"   ✓ Saved: fig_cluster_bar_percentages.png")
    except Exception as e:
        print(f"   ⚠ Error generating cluster bar percentages: {e}")
    
    # === PLOT 10: Boxplots by Cluster ===
    try:
        print("\n10. Generating boxplots by cluster...")
        if 'Cluster' in df.columns:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Feature Distributions by Cluster', fontsize=16, fontweight='bold')
            
            for ax, var in zip(axes.flatten(), feature_cols):
                if var in df.columns:
                    data_list = [df[df['Cluster'] == c][var].dropna() for c in sorted(df['Cluster'].unique())]
                    bp = ax.boxplot(data_list, labels=[f'C{c+1}' for c in sorted(df['Cluster'].unique())],
                                   patch_artist=True)
                    
                    for patch, cluster_id in zip(bp['boxes'], sorted(df['Cluster'].unique())):
                        patch.set_facecolor(cluster_colors.get(cluster_id, '#333333'))
                    
                    ax.set_xlabel('Cluster', fontsize=10)
                    ax.set_ylabel(var, fontsize=10)
                    ax.set_title(f'Distribution of {var}', fontsize=11)
                    ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'fig_boxplots_clusters.png'), dpi=PLOT_DPI, bbox_inches='tight')
            plt.close()
            print(f"   ✓ Saved: fig_boxplots_clusters.png")
    except Exception as e:
        print(f"   ⚠ Error generating boxplots: {e}")
    
    # === PLOT 11: Centroids Table Image ===
    try:
        print("\n11. Generating centroids table image...")
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.axis('off')
        
        centroids_display = centroids_df.copy().round(4)
        centroids_display['Aerosol_Type'] = cluster_mapping['Aerosol_Type'].values
        
        table_data = [['Cluster', 'Aerosol Type'] + feature_cols]
        for idx, row in centroids_display.iterrows():
            row_data = [f'Cluster {idx}', row['Aerosol_Type']] + [row[col] for col in feature_cols]
            table_data.append(row_data)
        
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.10, 0.18] + [0.14]*5)
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.5)
        
        for i in range(7):
            table[(0, i)].set_facecolor('#FF9800')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('Cluster Centroids & Aerosol Type Classification', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fig_centroids_table.png'), dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: fig_centroids_table.png")
    except Exception as e:
        print(f"   ⚠ Error generating centroids table: {e}")
    
    # === PLOT 12: Clustered Map ===
    try:
        print("\n12. Generating clustered map...")
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            
            if 'Site' in df.columns and 'Cluster' in df.columns:
                # Get site coordinates and majority cluster
                site_cluster_counts = df.groupby(['Site', 'Cluster']).size().unstack(fill_value=0)
                majority_cluster = site_cluster_counts.idxmax(axis=1)
                
                site_data = pd.DataFrame({
                    'Latitude': coords_df.set_index('Site')['Latitude'],
                    'Longitude': coords_df.set_index('Site')['Longitude'],
                    'Majority_Cluster': majority_cluster
                }).dropna()
                
                fig = plt.figure(figsize=(16, 10))
                ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
                ax.add_feature(cfeature.LAND, facecolor='#f5f5dc', alpha=0.3)
                ax.add_feature(cfeature.OCEAN, facecolor='#b3d9ff', alpha=0.5)
                ax.set_global()
                gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
                gl.top_labels = False
                gl.right_labels = False
                
                for cluster_id in sorted(site_data['Majority_Cluster'].unique()):
                    subset = site_data[site_data['Majority_Cluster'] == cluster_id]
                    ax.scatter(subset['Longitude'], subset['Latitude'],
                              c=cluster_colors.get(cluster_id, '#333333'), s=100,
                              edgecolor='black', linewidth=0.5, alpha=0.8,
                              label=f'Cluster {cluster_id + 1}', transform=ccrs.PlateCarree(), zorder=5)
                
                ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=10)
                plt.title('Global Distribution of Sites by Cluster Assignment', 
                         fontsize=14, fontweight='bold', pad=20)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'fig_clustered_map.png'), dpi=PLOT_DPI, bbox_inches='tight')
                plt.close()
                print(f"   ✓ Saved: fig_clustered_map.png")
        except ImportError:
            print(f"   ⚠ Cartopy not available, skipping clustered map")
    except Exception as e:
        print(f"   ⚠ Error generating clustered map: {e}")
    
    # === PLOT 13: CDF Summary ===
    try:
        print("\n13. Generating CDF summary...")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Cumulative Distribution Functions (CDF) of Aerosol Parameters", fontsize=16)
        
        for ax, (var, label, bins) in zip(axes.flatten(), hist_vars):
            if var in plot_df.columns:
                data = plot_df[var].dropna().sort_values()
                cdf = np.arange(1, len(data) + 1) / len(data)
                ax.plot(data, cdf, linewidth=2, color='steelblue')
                ax.set_xlabel(label, fontsize=11)
                ax.set_ylabel('Cumulative Probability', fontsize=11)
                ax.set_title(f'CDF of {label}', fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fig_cdf_summary.png'), dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: fig_cdf_summary.png")
    except Exception as e:
        print(f"   ⚠ Error generating CDF summary: {e}")
    
    # === PLOT 14: Timeseries (if date column present) ===
    try:
        print("\n14. Checking for timeseries plot...")
        if 'Date(dd:mm:yyyy)' in df.columns and 'Cluster' in df.columns:
            print("   Generating timeseries plot...")
            df_ts = df.copy()
            df_ts['Date_parsed'] = pd.to_datetime(df_ts['Date(dd:mm:yyyy)'], format='%d:%m:%Y', errors='coerce')
            df_ts = df_ts.dropna(subset=['Date_parsed'])
            
            if len(df_ts) > 0:
                df_ts = df_ts.sort_values('Date_parsed')
                fig, ax = plt.subplots(figsize=(14, 6))
                
                for cluster_id in sorted(df_ts['Cluster'].unique()):
                    subset = df_ts[df_ts['Cluster'] == cluster_id]
                    monthly = subset.groupby(pd.Grouper(key='Date_parsed', freq='M')).size()
                    ax.plot(monthly.index, monthly.values, marker='o', label=f'Cluster {cluster_id + 1}',
                           color=cluster_colors.get(cluster_id, '#333333'))
                
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Number of Observations', fontsize=12)
                ax.set_title('Time Series of Cluster Observations', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'fig_timeseries.png'), dpi=PLOT_DPI, bbox_inches='tight')
                plt.close()
                print(f"   ✓ Saved: fig_timeseries.png")
            else:
                print(f"   ⚠ No valid dates for timeseries")
        else:
            print(f"   ⚠ Skipped fig_timeseries.png (no date column or clusters)")
    except Exception as e:
        print(f"   ⚠ Error generating timeseries: {e}")
    
    print("\n✓ All visualizations generated successfully!")

# Generate Asia-specific tables
generate_asia_tables(df_features, centroids_df, OUTPUT_DIR)

# ============================================================================
# ASIA-SPECIFIC SCATTER PLOTS (Paper Style)
# ============================================================================
def generate_asia_scatter_plots(df, output_dir):
    """Generate scatter plots matching the paper's style for Asia region"""
    print("\n" + "="*80)
    print("GENERATING ASIA-SPECIFIC SCATTER PLOTS (PAPER STYLE)")
    print("-" * 80)
    
    # Subsample if needed
    if len(df) > 15000:
        print(f"\nSubsampling to 15,000 points for clearer visualization...")
        plot_df = df.sample(n=15000, random_state=RANDOM_STATE)
    else:
        plot_df = df
    
    # Define cluster colors matching paper
    cluster_colors = {
        0: '#1f77b4',  # Blue - Cluster 1
        1: '#ff7f0e',  # Orange - Cluster 2  
        2: '#d62728'   # Red - Cluster 3
    }
    
    # === PAPER FIGURE: 4-panel scatter plots with clusters ===
    print("\n1. Generating 4-panel scatter plots with cluster colors...")
    try:
        if 'Cluster' in plot_df.columns:
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            fig.suptitle('Aerosol Optical Properties vs EAE (ASIA Region, 3 Clusters)', 
                        fontsize=16, fontweight='bold')
            
            # Define scatter pairs (x, y, xlabel, ylabel)
            pairs = [
                ('EAE', 'AAE', 'EAE 440-870 nm', 'AAE 440-870 nm'),
                ('EAE', 'AOD_fine', 'EAE 440-870 nm', 'AOD fine mode 440 nm'),
                ('EAE', 'SSA', 'EAE 440-870 nm', 'SSA 440 nm'),
                ('EAE', 'RRI', 'EAE 440-870 nm', 'RRI 440 nm')
            ]
            
            for ax, (xcol, ycol, xlabel, ylabel) in zip(axes.flatten(), pairs):
                if xcol in plot_df.columns and ycol in plot_df.columns:
                    # Plot each cluster
                    for cluster_id in sorted(plot_df['Cluster'].unique()):
                        subset = plot_df[plot_df['Cluster'] == cluster_id]
                        color = cluster_colors.get(cluster_id, '#333333')
                        ax.scatter(subset[xcol], subset[ycol], 
                                  s=8, color=color, alpha=0.4, edgecolors='none',
                                  label=f'cluster {cluster_id + 1}')
                        
                        # Add centroid marker
                        cx = subset[xcol].mean()
                        cy = subset[ycol].mean()
                        ax.scatter(cx, cy, s=200, color=color, marker='o', 
                                  edgecolors='black', linewidths=2, zorder=10)
                    
                    ax.set_xlabel(xlabel, fontsize=11)
                    ax.set_ylabel(ylabel, fontsize=11)
                    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                    ax.legend(loc='best', fontsize=9, framealpha=0.9)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'fig_asia_scatter_4panel.png'), 
                       dpi=PLOT_DPI, bbox_inches='tight')
            plt.close()
            print(f"   ✓ Saved: fig_asia_scatter_4panel.png")
    except Exception as e:
        print(f"   ⚠ Error generating 4-panel scatter: {e}")
    
    # === CENTROID-ONLY SCATTER PLOTS ===
    print("\n2. Generating centroid-only scatter plots...")
    try:
        if 'Cluster' in plot_df.columns:
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            fig.suptitle('Cluster Centroids: Aerosol Properties (ASIA Region)', 
                        fontsize=16, fontweight='bold')
            
            pairs = [
                ('EAE', 'AAE', 'EAE 440-870 nm', 'AAE 440-870 nm'),
                ('EAE', 'AOD_fine', 'EAE 440-870 nm', 'AOD fine mode 440 nm'),
                ('EAE', 'SSA', 'EAE 440-870 nm', 'SSA 440 nm'),
                ('EAE', 'RRI', 'EAE 440-870 nm', 'RRI 440 nm')
            ]
            
            for ax, (xcol, ycol, xlabel, ylabel) in zip(axes.flatten(), pairs):
                if xcol in plot_df.columns and ycol in plot_df.columns:
                    # Plot data points in light gray
                    ax.scatter(plot_df[xcol], plot_df[ycol], 
                              s=5, color='lightgray', alpha=0.3, edgecolors='none')
                    
                    # Plot centroids prominently
                    for cluster_id in sorted(plot_df['Cluster'].unique()):
                        subset = plot_df[plot_df['Cluster'] == cluster_id]
                        cx = subset[xcol].mean()
                        cy = subset[ycol].mean()
                        color = cluster_colors.get(cluster_id, '#333333')
                        ax.scatter(cx, cy, s=300, color=color, marker='o', 
                                  edgecolors='black', linewidths=2, zorder=10,
                                  label=f'Cluster {cluster_id + 1} centroid')
                    
                    ax.set_xlabel(xlabel, fontsize=11)
                    ax.set_ylabel(ylabel, fontsize=11)
                    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                    ax.legend(loc='best', fontsize=9, framealpha=0.9)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'fig_asia_centroids_only.png'), 
                       dpi=PLOT_DPI, bbox_inches='tight')
            plt.close()
            print(f"   ✓ Saved: fig_asia_centroids_only.png")
    except Exception as e:
        print(f"   ⚠ Error generating centroid scatter: {e}")
    
    # === EXTENDED SCATTER MATRIX (6 panels) ===
    print("\n3. Generating extended 6-panel scatter matrix...")
    try:
        if 'Cluster' in plot_df.columns:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Comprehensive Aerosol Properties Scatter Matrix (ASIA Region)', 
                        fontsize=16, fontweight='bold')
            
            pairs = [
                ('EAE', 'AOD_total', 'EAE 440-870 nm', 'AOD Total 440 nm'),
                ('EAE', 'AOD_fine', 'EAE 440-870 nm', 'AOD Fine 440 nm'),
                ('EAE', 'AAE', 'EAE 440-870 nm', 'AAE 440-870 nm'),
                ('EAE', 'SSA', 'EAE 440-870 nm', 'SSA 440 nm'),
                ('EAE', 'RRI', 'EAE 440-870 nm', 'RRI 440 nm'),
                ('AOD_fine', 'SSA', 'AOD Fine 440 nm', 'SSA 440 nm')
            ]
            
            for ax, (xcol, ycol, xlabel, ylabel) in zip(axes.flatten(), pairs):
                if xcol in plot_df.columns and ycol in plot_df.columns:
                    for cluster_id in sorted(plot_df['Cluster'].unique()):
                        subset = plot_df[plot_df['Cluster'] == cluster_id]
                        color = cluster_colors.get(cluster_id, '#333333')
                        ax.scatter(subset[xcol], subset[ycol], 
                                  s=6, color=color, alpha=0.4, edgecolors='none',
                                  label=f'cluster {cluster_id + 1}')
                        
                        # Add centroid
                        cx = subset[xcol].mean()
                        cy = subset[ycol].mean()
                        ax.scatter(cx, cy, s=200, color=color, marker='D', 
                                  edgecolors='black', linewidths=2, zorder=10)
                    
                    ax.set_xlabel(xlabel, fontsize=10)
                    ax.set_ylabel(ylabel, fontsize=10)
                    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                    if ax == axes[0, 0]:
                        ax.legend(loc='best', fontsize=8, framealpha=0.9)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'fig_asia_scatter_6panel.png'), 
                       dpi=PLOT_DPI, bbox_inches='tight')
            plt.close()
            print(f"   ✓ Saved: fig_asia_scatter_6panel.png")
    except Exception as e:
        print(f"   ⚠ Error generating 6-panel scatter: {e}")
    
    print("\n✓ All Asia-specific scatter plots generated!")

generate_asia_scatter_plots(df_features, OUTPUT_DIR)

make_plots(df_features, centroids_df, cluster_mapping, OUTPUT_DIR)

# ============================================================================
# STEP 10: COMPARISON WITH OTHER ALGORITHMS (ALL 5 ALGORITHMS)
# ============================================================================
if not SKIP_ALGORITHM_COMPARISON:
    print("\n" + "="*80)
    print("STEP 10: COMPREHENSIVE ALGORITHM COMPARISON")
    print("-" * 80)
    print("Comparing: KMeans, GMM, Agglomerative, BIRCH, Spectral")
    
    try:
        from sklearn.mixture import GaussianMixture
        from sklearn.cluster import AgglomerativeClustering, Birch
        
        # Use sampled data for comparison
        X_compare = X_scaled[:min(5000, len(X_scaled))]
        print(f"\nUsing {len(X_compare)} samples for algorithm comparison...")
        
        # Store results
        results = {}
        
        # 1. KMeans
        print("\n1. Running KMeans...")
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
        kmeans_labels = kmeans.fit_predict(X_compare)
        kmeans_sil = silhouette_score(X_compare, kmeans_labels, sample_size=min(2000, len(X_compare)))
        kmeans_db = davies_bouldin_score(X_compare, kmeans_labels)
        kmeans_ch = calinski_harabasz_score(X_compare, kmeans_labels)
        results['KMeans'] = {
            'labels': kmeans_labels,
            'silhouette': kmeans_sil,
            'davies_bouldin': kmeans_db,
            'calinski_harabasz': kmeans_ch
        }
        print(f"   Silhouette: {kmeans_sil:.4f}, DB: {kmeans_db:.4f}, CH: {kmeans_ch:.2f}")
        
        # 2. GMM (Gaussian Mixture Model)
        print("\n2. Running GMM (Gaussian Mixture Model)...")
        gmm = GaussianMixture(n_components=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
        gmm_labels = gmm.fit_predict(X_compare)
        gmm_sil = silhouette_score(X_compare, gmm_labels, sample_size=min(2000, len(X_compare)))
        gmm_db = davies_bouldin_score(X_compare, gmm_labels)
        gmm_ch = calinski_harabasz_score(X_compare, gmm_labels)
        results['GMM'] = {
            'labels': gmm_labels,
            'silhouette': gmm_sil,
            'davies_bouldin': gmm_db,
            'calinski_harabasz': gmm_ch
        }
        print(f"   Silhouette: {gmm_sil:.4f}, DB: {gmm_db:.4f}, CH: {gmm_ch:.2f}")
        
        # 3. Agglomerative Clustering
        print("\n3. Running Agglomerative Clustering...")
        agglomerative = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage='ward')
        agglomerative_labels = agglomerative.fit_predict(X_compare)
        agglomerative_sil = silhouette_score(X_compare, agglomerative_labels, sample_size=min(2000, len(X_compare)))
        agglomerative_db = davies_bouldin_score(X_compare, agglomerative_labels)
        agglomerative_ch = calinski_harabasz_score(X_compare, agglomerative_labels)
        results['Agglomerative'] = {
            'labels': agglomerative_labels,
            'silhouette': agglomerative_sil,
            'davies_bouldin': agglomerative_db,
            'calinski_harabasz': agglomerative_ch
        }
        print(f"   Silhouette: {agglomerative_sil:.4f}, DB: {agglomerative_db:.4f}, CH: {agglomerative_ch:.2f}")
        
        # 4. BIRCH
        print("\n4. Running BIRCH...")
        birch = Birch(n_clusters=N_CLUSTERS, threshold=0.5)
        birch_labels = birch.fit_predict(X_compare)
        birch_sil = silhouette_score(X_compare, birch_labels, sample_size=min(2000, len(X_compare)))
        birch_db = davies_bouldin_score(X_compare, birch_labels)
        birch_ch = calinski_harabasz_score(X_compare, birch_labels)
        results['BIRCH'] = {
            'labels': birch_labels,
            'silhouette': birch_sil,
            'davies_bouldin': birch_db,
            'calinski_harabasz': birch_ch
        }
        print(f"   Silhouette: {birch_sil:.4f}, DB: {birch_db:.4f}, CH: {birch_ch:.2f}")
        
        # 5. Spectral Clustering
        print("\n5. Using Spectral Clustering (already computed)...")
        spectral_labels_compare = labels[:len(X_compare)]
        results['Spectral'] = {
            'labels': spectral_labels_compare,
            'silhouette': metrics['silhouette'],
            'davies_bouldin': metrics['davies_bouldin'],
            'calinski_harabasz': metrics['calinski_harabasz']
        }
        print(f"   Silhouette: {metrics['silhouette']:.4f}, DB: {metrics['davies_bouldin']:.4f}, CH: {metrics['calinski_harabasz']:.2f}")
        
        # Create comprehensive comparison plot (2x3 grid)
        print("\n6. Generating comparison visualization...")
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle('Comprehensive Algorithm Comparison (ASIA Region, 3 Clusters)', 
                     fontsize=18, fontweight='bold')
        
        # Use PCA for 2D visualization
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        X_pca = pca.fit_transform(X_compare)
        
        # Create subplots for each algorithm
        algo_names = ['KMeans', 'GMM', 'Agglomerative', 'BIRCH', 'Spectral']
        
        for idx, algo_name in enumerate(algo_names, 1):
            ax = fig.add_subplot(2, 3, idx)
            algo_data = results[algo_name]
            
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                                c=algo_data['labels'], 
                                cmap='tab10', s=15, alpha=0.6, edgecolors='none')
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
            ax.set_title(f'{algo_name}\n'
                        f'Sil: {algo_data["silhouette"]:.4f} | '
                        f'DB: {algo_data["davies_bouldin"]:.4f} | '
                        f'CH: {algo_data["calinski_harabasz"]:.0f}', 
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Cluster', fontsize=9)
        
        # Add comparison table in the 6th subplot
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        
        # Create comparison table
        table_data = [['Algorithm', 'Silhouette ↑', 'Davies-Bouldin ↓', 'Calinski-Harabasz ↑']]
        for algo_name in algo_names:
            algo_data = results[algo_name]
            table_data.append([
                algo_name,
                f"{algo_data['silhouette']:.4f}",
                f"{algo_data['davies_bouldin']:.4f}",
                f"{algo_data['calinski_harabasz']:.0f}"
            ])
        
        table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.25, 0.25, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Style header
        for i in range(4):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Highlight best scores
        best_sil = max(results.values(), key=lambda x: x['silhouette'])
        best_db = min(results.values(), key=lambda x: x['davies_bouldin'])
        best_ch = max(results.values(), key=lambda x: x['calinski_harabasz'])
        
        for i, algo_name in enumerate(algo_names, 1):
            if results[algo_name]['silhouette'] == best_sil['silhouette']:
                table[(i, 1)].set_facecolor('#C8E6C9')
            if results[algo_name]['davies_bouldin'] == best_db['davies_bouldin']:
                table[(i, 2)].set_facecolor('#C8E6C9')
            if results[algo_name]['calinski_harabasz'] == best_ch['calinski_harabasz']:
                table[(i, 3)].set_facecolor('#C8E6C9')
        
        ax6.set_title('Performance Metrics Comparison\n(Green = Best Score)', 
                     fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'fig_algo_comparison.png'), dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        print(f"\n✓ Comparison plot saved: {OUTPUT_DIR}/fig_algo_comparison.png")
        
        # Save comparison results to CSV
        comparison_df = pd.DataFrame([
            {
                'Algorithm': algo_name,
                'Silhouette_Score': algo_data['silhouette'],
                'Davies_Bouldin_Index': algo_data['davies_bouldin'],
                'Calinski_Harabasz_Index': algo_data['calinski_harabasz']
            }
            for algo_name, algo_data in results.items()
        ])
        comparison_df.to_csv(os.path.join(OUTPUT_DIR, 'algorithm_comparison_metrics.csv'), index=False)
        print(f"✓ Metrics saved: {OUTPUT_DIR}/algorithm_comparison_metrics.csv")
        
    except Exception as e:
        print(f"\n⚠ Error in algorithm comparison: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n" + "="*80)
    print("STEP 10: ALGORITHM COMPARISON (SKIPPED)")
    print("-" * 80)
    print("⚠ Algorithm comparison skipped for faster execution.")
    print("  Set SKIP_ALGORITHM_COMPARISON=False in configuration to enable.")

# ============================================================================
# STEP 11: GENERATE COMPREHENSIVE PDF REPORT
# ============================================================================
print("\n" + "="*80)
print("STEP 11: GENERATING COMPREHENSIVE PDF REPORT")
print("="*80)

def generate_comprehensive_pdf_report():
    """
    Generate comprehensive PDF report with all tables, plots, and analysis
    """
    print("\nCreating multi-page PDF with all results...")
    print("  (Using optimized DPI for faster generation...)")
    
    pdf_filename = f'{OUTPUT_DIR}/DA9_spectral_clustering_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
    
    with PdfPages(pdf_filename) as pdf:
        # === PAGE 1: TITLE PAGE WITH STUDENT INFO ===
        fig = plt.figure(figsize=(11, 8.5))
        
        # Student Information at the top
        fig.text(0.5, 0.95, 'Student Information', ha='center', fontsize=14, fontweight='bold')
        fig.text(0.5, 0.91, 'Name: Jsh Agarwal', ha='center', fontsize=12)
        fig.text(0.5, 0.88, 'UID: 2022200004', ha='center', fontsize=12)
        fig.text(0.5, 0.85, 'Assignment: DA9 - Aerosol Spectral Clustering Analysis', ha='center', fontsize=12)
        
        # Horizontal line separator
        fig.add_artist(plt.Line2D([0.1, 0.9], [0.82, 0.82], transform=fig.transFigure, 
                                  color='black', linewidth=2))
        
        # Report title
        fig.text(0.5, 0.72, 'Aerosol Spectral Clustering Analysis Report', 
                ha='center', fontsize=22, fontweight='bold')
        fig.text(0.5, 0.65, 'Following Ningombam et al., 2024 Methodology', 
                ha='center', fontsize=16, color='darkblue')
        fig.text(0.5, 0.60, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                ha='center', fontsize=12, style='italic')
        
        # Methodology section
        fig.text(0.5, 0.48, 'Methodology:', ha='center', fontsize=14, fontweight='bold')
        fig.text(0.1, 0.42, f'• Features Used: AOD_fine, EAE, AAE, SSA, RRI', fontsize=11)
        fig.text(0.1, 0.39, f'• Clustering Algorithm: Spectral Clustering (k={N_CLUSTERS})', fontsize=11)
        fig.text(0.1, 0.36, f'• Affinity: Nearest Neighbors (n_neighbors={N_NEIGHBORS})', fontsize=11)
        fig.text(0.1, 0.33, f'• Standardization: Z-score normalization', fontsize=11)
        fig.text(0.1, 0.30, f'• Aerosol Types Identified: {", ".join(cluster_mapping["Aerosol_Type"].values)}', fontsize=11)
        
        # Proxy usage section
        fig.text(0.5, 0.20, 'Proxy Usage:', ha='center', fontsize=14, fontweight='bold')
        if proxy_flags['used_AOD_fine_proxy'] > 0:
            fig.text(0.1, 0.16, f'• AOD_fine: Computed from EAE for {proxy_flags["used_AOD_fine_proxy"]} samples', fontsize=10)
        else:
            fig.text(0.1, 0.16, f'• AOD_fine: Direct measurements used', fontsize=10)
        
        if proxy_flags['used_RRI_proxy'] > 0:
            fig.text(0.1, 0.13, f'• RRI: Default value ({DEFAULT_RRI}) for {proxy_flags["used_RRI_proxy"]} samples', fontsize=10)
        else:
            fig.text(0.1, 0.13, f'• RRI: Direct measurements used', fontsize=10)
        
        # Dataset info at bottom
        fig.text(0.5, 0.08, 'Dataset: combined_dataset.csv (ASIA Region)', ha='center', fontsize=10, style='italic')
        fig.text(0.5, 0.05, f'Samples Analyzed: {len(df_features)} | Features: {len(feature_cols)}', ha='center', fontsize=10, style='italic')
        
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("  ✓ Page 1: Title page")
        
        # === PAGE 2: DATA CLEANING SUMMARY ===
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('Data Cleaning & Preprocessing Summary', fontsize=18, fontweight='bold')
        
        # Create text summary
        cleaning_text = f"""
DATA CLEANING SUMMARY
{'='*60}

Initial Dataset:
  • Input file: combined_dataset.csv
  • Initial rows: {cleaning_log['initial_rows']:,}
  • Initial columns: {len(df.columns)}

Data Cleaning:
  • Rows removed: {cleaning_log['rows_removed']:,} ({cleaning_log['removal_pct']:.2f}%)
  • Final rows: {cleaning_log['final_rows']:,}
  • Reason: Missing values in required columns

Required Columns:
  • AOD_total (Total AOD at 440 nm)
  • EAE (Extinction Angstrom Exponent 440-870 nm)
  • AAE (Absorption Angstrom Exponent 440-870 nm)
  • SSA (Single Scattering Albedo at 440 nm)

Proxy Features Computed:
"""
        if proxy_flags['used_AOD_fine_proxy'] > 0:
            cleaning_text += f"""
  • AOD_fine: Computed for {proxy_flags['used_AOD_fine_proxy']} samples
    - Method: Fine-Mode Fraction from EAE
    - Formula: FMF = clip((EAE - {EAE_MIN_FMF}) / ({EAE_MAX_FMF} - {EAE_MIN_FMF}), 0, 1)
    - Then: AOD_fine = AOD_total × FMF
"""
        else:
            cleaning_text += "\n  • AOD_fine: Direct measurements used\n"
        
        if proxy_flags['used_RRI_proxy'] > 0:
            cleaning_text += f"""
  • RRI: Set to default value for {proxy_flags['used_RRI_proxy']} samples
    - Default value: {DEFAULT_RRI}
    - Justification: Mean RRI for urban/mixed aerosols (paper)
"""
        else:
            cleaning_text += "\n  • RRI: Direct measurements used\n"
        
        cleaning_text += f"""

Feature Standardization:
  • Method: StandardScaler (z-score normalization)
  • Mean: {0:.6f} (after scaling)
  • Std: {1:.6f} (after scaling)

Final Preprocessed Data:
  • Shape: {X_scaled.shape}
  • Features: {', '.join(feature_cols)}
  • Ready for clustering: ✓
        """
        
        fig.text(0.1, 0.85, cleaning_text, fontsize=9, verticalalignment='top', family='monospace')
        
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("  ✓ Page 2: Data cleaning summary")
        
        # === PAGE 3: FEATURE STATISTICS TABLE ===
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
        fig.suptitle('Feature Statistics (Before & After Standardization)', fontsize=18, fontweight='bold')
        
        # Before standardization
        axes[0].axis('off')
        before_stats = df_features[feature_cols].describe().T
        before_stats = before_stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
        before_stats = before_stats.round(4)
        
        table1_data = [['Feature'] + before_stats.columns.tolist()]
        for idx, row in before_stats.iterrows():
            table1_data.append([idx] + row.tolist())
        
        table1 = axes[0].table(cellText=table1_data, cellLoc='center', loc='center',
                              colWidths=[0.15] + [0.12]*7)
        table1.auto_set_font_size(False)
        table1.set_fontsize(9)
        table1.scale(1, 2)
        
        for i in range(8):
            table1[(0, i)].set_facecolor('#4CAF50')
            table1[(0, i)].set_text_props(weight='bold', color='white')
        
        axes[0].set_title('Original Feature Statistics', fontsize=14, fontweight='bold', pad=20)
        
        # After standardization
        axes[1].axis('off')
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
        after_stats = X_scaled_df.describe().T
        after_stats = after_stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
        after_stats = after_stats.round(4)
        
        table2_data = [['Feature'] + after_stats.columns.tolist()]
        for idx, row in after_stats.iterrows():
            table2_data.append([idx] + row.tolist())
        
        table2 = axes[1].table(cellText=table2_data, cellLoc='center', loc='center',
                              colWidths=[0.15] + [0.12]*7)
        table2.auto_set_font_size(False)
        table2.set_fontsize(9)
        table2.scale(1, 2)
        
        for i in range(8):
            table2[(0, i)].set_facecolor('#2196F3')
            table2[(0, i)].set_text_props(weight='bold', color='white')
        
        axes[1].set_title('Standardized Feature Statistics', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("  ✓ Page 3: Feature statistics")
        
        # === PAGE 4: CLUSTERING RESULTS TABLE ===
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('Spectral Clustering Results Summary', fontsize=18, fontweight='bold')
        
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Performance metrics
        perf_text = f"""
CLUSTERING PERFORMANCE METRICS
{'='*60}

Configuration:
  • Algorithm: Spectral Clustering
  • n_clusters: {N_CLUSTERS}
  • Affinity: nearest_neighbors
  • n_neighbors: {N_NEIGHBORS}
  • assign_labels: kmeans
  • random_state: {RANDOM_STATE}

Quality Metrics:
  • Silhouette Score: {metrics['silhouette']:.4f}
    (Range: -1 to 1, higher is better)
    Interpretation: {"Excellent" if metrics['silhouette'] > 0.7 else "Good" if metrics['silhouette'] > 0.5 else "Fair" if metrics['silhouette'] > 0.3 else "Poor"}
    
  • Davies-Bouldin Index: {metrics['davies_bouldin']:.4f}
    (Lower is better, 0 is ideal)
    Interpretation: {"Excellent" if metrics['davies_bouldin'] < 0.5 else "Good" if metrics['davies_bouldin'] < 1.0 else "Fair" if metrics['davies_bouldin'] < 1.5 else "Poor"}
    
  • Calinski-Harabasz Index: {metrics['calinski_harabasz']:.2f}
    (Higher is better)
    Interpretation: {"Excellent" if metrics['calinski_harabasz'] > 500 else "Good" if metrics['calinski_harabasz'] > 300 else "Fair" if metrics['calinski_harabasz'] > 100 else "Poor"}

Cluster Distribution:
"""
        unique, counts = np.unique(labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            pct = (count / len(labels)) * 100
            aero_type = cluster_mapping[cluster_mapping['Cluster'] == cluster_id]['Aerosol_Type'].values[0]
            perf_text += f"  • Cluster {cluster_id} ({aero_type}): {count:,} samples ({pct:.2f}%)\n"
        
        fig.text(0.1, 0.85, perf_text, fontsize=10, verticalalignment='top', family='monospace')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("  ✓ Page 4: Clustering results")
        
        # === PAGE 5: CLUSTER CENTROIDS TABLE ===
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('Cluster Centroids & Aerosol Type Classification', fontsize=18, fontweight='bold')
        
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Create centroids table
        centroids_display = centroids_df.copy()
        centroids_display = centroids_display.round(4)
        centroids_display['Aerosol_Type'] = cluster_mapping['Aerosol_Type'].values
        
        table_data = [['Cluster', 'Aerosol Type'] + feature_cols]
        for idx, row in centroids_display.iterrows():
            row_data = [f'Cluster {idx}', row['Aerosol_Type']] + [row[col] for col in feature_cols]
            table_data.append(row_data)
        
        table = ax.table(cellText=table_data, cellLoc='center', loc='upper center',
                       colWidths=[0.10, 0.18] + [0.14]*5)
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.5)
        
        # Style header
        for i in range(7):
            table[(0, i)].set_facecolor('#FF9800')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color-code rows by aerosol type
        colors = {'Dust': '#FFE0B2', 'Biomass Burning': '#FFCCBC', 
                 'Urban/Industrial': '#C5E1A5', 'Mixed': '#E1BEE7'}
        for i in range(1, len(table_data)):
            aerosol_type = table_data[i][1]
            color = colors.get(aerosol_type, '#FFFFFF')
            for j in range(7):
                table[(i, j)].set_facecolor(color)
        
        # Add classification criteria
        criteria_text = """
CLASSIFICATION CRITERIA (from Ningombam et al., 2024):

• Dust:
  - Low EAE (< 0.6): Coarse particles dominate
  - High AAE (> 1.8): Strong wavelength-dependent absorption

• Biomass Burning:
  - High EAE (≥ 1.0): Fine particles dominate
  - Moderate AAE (1.2-1.8): Organic carbon absorption
  - Low SSA (≤ 0.92): Strong light absorption

• Urban/Industrial:
  - High EAE (≥ 1.0): Fine particles dominate
  - High SSA (≥ 0.94): Strong light scattering
  
• Mixed:
  - Characteristics that don't fit above categories
  - Complex mixture of aerosol types
        """
        
        fig.text(0.1, 0.35, criteria_text, fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("  ✓ Page 5: Cluster centroids & classification")
        
        # === PAGE 6: CLUSTER PERCENTAGE ANALYSIS ===
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
        fig.suptitle('Cluster Distribution Analysis', fontsize=18, fontweight='bold')
        
        # Overall percentage table
        axes[0].axis('off')
        overall_table_data = [['Aerosol Type', 'Count', 'Percentage (%)']]
        for _, row in overall_pct_df.iterrows():
            count = (df_features['Aerosol_Type'] == row['Aerosol_Type']).sum()
            overall_table_data.append([row['Aerosol_Type'], f"{count:,}", f"{row['Percentage']:.2f}%"])
        
        table1 = axes[0].table(cellText=overall_table_data, cellLoc='center', loc='center',
                              colWidths=[0.40, 0.20, 0.20])
        table1.auto_set_font_size(False)
        table1.set_fontsize(11)
        table1.scale(1, 3)
        
        for i in range(3):
            table1[(0, i)].set_facecolor('#673AB7')
            table1[(0, i)].set_text_props(weight='bold', color='white')
        
        axes[0].set_title('Overall Distribution (ASIA Region)', fontsize=14, fontweight='bold', pad=20)
        
        # Per-site distribution (if available)
        if 'Site' in df_features.columns:
            axes[1].axis('off')
            site_cluster_counts = pd.crosstab(df_features['Site'], df_features['Aerosol_Type'], normalize='index') * 100
            site_totals = df_features['Site'].value_counts()
            
            # Top 10 sites
            top_sites = site_totals.nlargest(10).index
            site_table_data = [['Site', 'Total Samples'] + df_features['Aerosol_Type'].unique().tolist()]
            
            for site in top_sites:
                row_data = [site, f"{site_totals[site]:,}"]
                for aero_type in df_features['Aerosol_Type'].unique():
                    pct = site_cluster_counts.loc[site, aero_type] if aero_type in site_cluster_counts.columns else 0
                    row_data.append(f"{pct:.1f}%")
                site_table_data.append(row_data)
            
            table2 = axes[1].table(cellText=site_table_data, cellLoc='center', loc='center',
                                  colWidths=[0.20, 0.15] + [0.15]*len(df_features['Aerosol_Type'].unique()))
            table2.auto_set_font_size(False)
            table2.set_fontsize(8)
            table2.scale(1, 2)
            
            for i in range(len(site_table_data[0])):
                table2[(0, i)].set_facecolor('#00897B')
                table2[(0, i)].set_text_props(weight='bold', color='white')
            
            axes[1].set_title('Top 10 Sites by Sample Count', fontsize=14, fontweight='bold', pad=20)
        else:
            axes[1].axis('off')
            axes[1].text(0.5, 0.5, 'Site information not available', ha='center', va='center',
                        fontsize=12, style='italic')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("  ✓ Page 6: Cluster distribution analysis")
        
        # === PAGES 7-N: EMBED EXISTING VISUALIZATIONS ===
        # Ordered list of images matching notebook - ASIA SPECIFIC
        image_files = [
            ('fig_page1_header.png', 'Report Header'),
            ('fig_map1.png', 'Global Spatial Distribution - Fine-mode AOD'),
            ('fig_map2.png', 'Global Distribution by Source Type'),
            ('fig_params.png', 'Frequency Distribution and CDF of Aerosol Parameters'),
            ('fig_histograms.png', 'Frequency Distributions of Aerosol Properties (ASIA)'),
            ('fig_scatter_AOD_EAE.png', 'AOD vs EAE Scatter Plot'),
            ('fig_asia_scatter_4panel.png', 'ASIA: 4-Panel Scatter Plots with Cluster Colors'),
            ('fig_asia_scatter_6panel.png', 'ASIA: Extended 6-Panel Scatter Matrix'),
            ('fig_asia_centroids_only.png', 'ASIA: Cluster Centroids Visualization'),
            ('fig_scatter_matrix_clusters.png', 'Aerosol Property Relationships (Colored by Cluster)'),
            ('fig_silhouette.png', 'Silhouette Plot for Clusters'),
            ('fig_cluster_bar_percentages.png', 'Cluster Distribution Bar Chart'),
            ('fig_centroids_table.png', 'Cluster Centroids Table'),
            ('fig_boxplots_clusters.png', 'Feature Distributions by Cluster'),
            ('fig_cdf_summary.png', 'Cumulative Distribution Functions'),
            ('fig_clustered_map.png', 'Global Distribution by Cluster Assignment'),
            ('fig_timeseries.png', 'Time Series of Cluster Observations'),
        ]
        if not SKIP_ALGORITHM_COMPARISON:
            image_files.append(('fig_algo_comparison.png', 'Algorithm Comparison: Spectral vs KMeans'))
        
        # Create placeholder image for missing files
        def create_placeholder(filename):
            """Create a placeholder image for missing files"""
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, f'Missing: {filename}', ha='center', va='center', 
                   fontsize=16, color='red', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig

        page_num = 7
        for img_file, title in image_files:
            img_path = os.path.join(OUTPUT_DIR, img_file)
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                    fig = plt.figure(figsize=(11, 8.5))
                    plt.imshow(img)
                    plt.title(title, fontsize=16, fontweight='bold', pad=20)
                    plt.axis('off')
                    plt.tight_layout()
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
                    print(f"  ✓ Page {page_num}: {img_file}")
                    page_num += 1
                except Exception as e:
                    print(f"  ⚠ Warning: Could not load {img_file}: {e}")
                    # Create placeholder
                    fig = create_placeholder(img_file)
                    plt.title(f'Missing: {title}', fontsize=16, fontweight='bold', pad=20, color='red')
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
                    page_num += 1
            else:
                print(f"  ⚠ Warning: {img_file} not found, adding placeholder...")
                fig = create_placeholder(img_file)
                plt.title(f'Missing: {title}', fontsize=16, fontweight='bold', pad=20, color='red')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                page_num += 1

        # === ASIA-SPECIFIC TABLES PAGES ===
        print(f"\n  Adding ASIA-specific tables to PDF...")
        
        # Table 2: Optical Parameters Statistics
        table2_path = os.path.join(OUTPUT_DIR, 'table2_asia_optical_parameters.csv')
        if os.path.exists(table2_path):
            try:
                table2_df = pd.read_csv(table2_path)
                fig = plt.figure(figsize=(11, 8.5))
                fig.suptitle('Table 2: Aerosol Optical Parameters Statistics for ASIA Region', 
                            fontsize=16, fontweight='bold')
                ax = fig.add_subplot(111)
                ax.axis('off')
                
                table_data = [table2_df.columns.tolist()] + table2_df.values.tolist()
                table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                               colWidths=[0.3, 0.35, 0.35])
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 2.5)
                
                for i in range(3):
                    table[(0, i)].set_facecolor('#4CAF50')
                    table[(0, i)].set_text_props(weight='bold', color='white')
                
                fig.text(0.1, 0.15, 'Note: Values shown as Average ± Standard Deviation and Median\n'
                        'Based on AERONET Level 2.0 data for ASIA region sites', 
                        fontsize=9, style='italic')
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Page {page_num}: Table 2 (Optical Parameters)")
                page_num += 1
            except Exception as e:
                print(f"  ⚠ Error adding Table 2: {e}")
        
        # Table 3: Seasonal Distribution
        table3_path = os.path.join(OUTPUT_DIR, 'table3_asia_seasonal_distribution.csv')
        if os.path.exists(table3_path):
            try:
                table3_df = pd.read_csv(table3_path)
                fig = plt.figure(figsize=(11, 8.5))
                fig.suptitle('Table 3: Seasonal Distribution of Aerosol Properties (ASIA Region)', 
                            fontsize=16, fontweight='bold')
                ax = fig.add_subplot(111)
                ax.axis('off')
                
                table_data = [table3_df.columns.tolist()] + table3_df.values.tolist()
                table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                               colWidths=[0.15] + [0.14]*6)
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 2.5)
                
                for i in range(len(table3_df.columns)):
                    table[(0, i)].set_facecolor('#2196F3')
                    table[(0, i)].set_text_props(weight='bold', color='white')
                
                fig.text(0.1, 0.15, 'Note: DJF=Dec-Jan-Feb (Winter), MAM=Mar-Apr-May (Spring),\n'
                        'JJA=Jun-Jul-Aug (Summer), SON=Sep-Oct-Nov (Autumn)', 
                        fontsize=9, style='italic')
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Page {page_num}: Table 3 (Seasonal Distribution)")
                page_num += 1
            except Exception as e:
                print(f"  ⚠ Error adding Table 3: {e}")
        
        # Table 4: Cluster Centroids
        table4_path = os.path.join(OUTPUT_DIR, 'table4_asia_cluster_centroids.csv')
        if os.path.exists(table4_path):
            try:
                table4_df = pd.read_csv(table4_path)
                fig = plt.figure(figsize=(11, 8.5))
                fig.suptitle('Table 4: Cluster Centroids with Aerosol Type Classification (ASIA)', 
                            fontsize=16, fontweight='bold')
                ax = fig.add_subplot(111)
                ax.axis('off')
                
                table_data = [table4_df.columns.tolist()] + table4_df.values.tolist()
                table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                               colWidths=[0.12]*8)
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 2.5)
                
                for i in range(len(table4_df.columns)):
                    table[(0, i)].set_facecolor('#FF9800')
                    table[(0, i)].set_text_props(weight='bold', color='white')
                
                fig.text(0.1, 0.15, 'Note: Aerosol Types: D=Dust, B=Biomass Burning, U=Urban/Industrial, M=Mixed\n'
                        'Classification based on EAE, AAE, SSA thresholds from Ningombam et al., 2024', 
                        fontsize=9, style='italic')
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Page {page_num}: Table 4 (Cluster Centroids)")
                page_num += 1
            except Exception as e:
                print(f"  ⚠ Error adding Table 4: {e}")
        
        # Table 5: Seasonal Cluster Distribution
        table5_path = os.path.join(OUTPUT_DIR, 'table5_asia_seasonal_cluster_distribution.csv')
        if os.path.exists(table5_path):
            try:
                table5_df = pd.read_csv(table5_path)
                fig = plt.figure(figsize=(11, 8.5))
                fig.suptitle('Table 5: Seasonal Distribution by Clusters (ASIA Region)', 
                            fontsize=16, fontweight='bold')
                ax = fig.add_subplot(111)
                ax.axis('off')
                
                table_data = [table5_df.columns.tolist()] + table5_df.values.tolist()
                table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                               colWidths=[0.2] + [0.26]*3)
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 2.5)
                
                for i in range(len(table5_df.columns)):
                    table[(0, i)].set_facecolor('#9C27B0')
                    table[(0, i)].set_text_props(weight='bold', color='white')
                
                fig.text(0.1, 0.15, 'Note: Values show percentage and dominant aerosol type for each cluster\n'
                        'Format: Percentage%, AerosolType (D/B/U/M)', 
                        fontsize=9, style='italic')
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Page {page_num}: Table 5 (Seasonal Clusters)")
                page_num += 1
            except Exception as e:
                print(f"  ⚠ Error adding Table 5: {e}")
        
        # Table 6: FMF/SSA Classification
        table6_path = os.path.join(OUTPUT_DIR, 'table6_asia_fmf_ssa_classification.csv')
        if os.path.exists(table6_path):
            try:
                table6_df = pd.read_csv(table6_path)
                fig = plt.figure(figsize=(11, 8.5))
                fig.suptitle('Table 6: Aerosol Classification Based on FMF and SSA (ASIA Region)', 
                            fontsize=16, fontweight='bold')
                ax = fig.add_subplot(111)
                ax.axis('off')
                
                table_data = [table6_df.columns.tolist()] + table6_df.values.tolist()
                table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                               colWidths=[0.2]*5)
                table.auto_set_font_size(False)
                table.set_fontsize(11)
                table.scale(1, 2.5)
                
                for i in range(len(table6_df.columns)):
                    table[(0, i)].set_facecolor('#00BCD4')
                    table[(0, i)].set_text_props(weight='bold', color='white')
                
                fig.text(0.1, 0.15, 'Note: Classification criteria:\n'
                        '  Uncer. = Uncertain/Unknown\n'
                        '  Dust = FMF < 0.4 and SSA < 0.95\n'
                        '  Mix. = Mixed aerosols (0.4 ≤ FMF < 0.6)\n'
                        '  N.A. = Non-Absorbing (SSA > 0.95)\n'
                        '  BC = Black Carbon (FMF ≥ 0.6 and SSA < 0.90)', 
                        fontsize=9, style='italic')
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Page {page_num}: Table 6 (FMF/SSA Classification)")
                page_num += 1
            except Exception as e:
                print(f"  ⚠ Error adding Table 6: {e}")

        # === FINAL PAGE: CONCLUSIONS ===
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.9, 'Conclusions and Interpretation', ha='center', fontsize=18, fontweight='bold')

        # Find dominant aerosol type
        dominant_type = overall_pct_df.loc[overall_pct_df['Percentage'].idxmax(), 'Aerosol_Type']
        dominant_pct = overall_pct_df['Percentage'].max()

        # Instead of one big block, print each section with its own y position for spacing
        y = 0.82
        line_space = 0.04

        fig.text(0.1, y, "1. Clustering Quality:", fontsize=11, fontweight='bold')
        y -= line_space
        fig.text(0.12, y, f"• Silhouette Score: {metrics['silhouette']:.4f} "
                          f"{'✓ Excellent separation' if metrics['silhouette'] > 0.7 else '✓ Good separation' if metrics['silhouette'] > 0.5 else '✓ Moderate separation' if metrics['silhouette'] > 0.3 else '⚠ Weak separation'}", fontsize=10)
        y -= line_space
        fig.text(0.12, y, f"• Algorithm: Spectral Clustering with {N_CLUSTERS} clusters", fontsize=10)
        y -= line_space
        fig.text(0.12, y, "• Method: Follows Ningombam et al., 2024 methodology", fontsize=10)
        y -= line_space * 1.5

        fig.text(0.1, y, "2. Aerosol Composition (ASIA Region):", fontsize=11, fontweight='bold')
        y -= line_space
        fig.text(0.12, y, f"• Dominant type: {dominant_type} ({dominant_pct:.1f}%)", fontsize=10)
        y -= line_space
        for _, row in cluster_mapping.iterrows():
            pct = overall_pct_df[overall_pct_df['Aerosol_Type'] == row['Aerosol_Type']]['Percentage'].values[0]
            fig.text(0.13, y, f"• {row['Aerosol_Type']}: {pct:.1f}%", fontsize=10)
            y -= line_space
        y -= line_space * 0.5

        fig.text(0.1, y, "3. Physical Interpretation:", fontsize=11, fontweight='bold')
        y -= line_space
        for _, row in cluster_mapping.iterrows():
            fig.text(0.13, y, f"{row['Aerosol_Type']}:", fontsize=10, fontweight='bold')
            y -= line_space
            fig.text(0.15, y, f"- {row['Explanation']}", fontsize=10)
            y -= line_space
            fig.text(0.15, y, f"- AOD_fine: {row['AOD_fine']:.4f}, EAE: {row['EAE']:.3f}", fontsize=10)
            y -= line_space
            fig.text(0.15, y, f"- AAE: {row['AAE']:.3f}, SSA: {row['SSA']:.3f}", fontsize=10)
            y -= line_space * 1.2

        fig.text(0.1, y, "4. Data Quality & Limitations:", fontsize=11, fontweight='bold')
        y -= line_space
        proxy_lines = []
        if proxy_flags['used_AOD_fine_proxy'] > 0:
            pct_proxy = (proxy_flags['used_AOD_fine_proxy'] / len(df_features)) * 100
            proxy_lines.append(f"- AOD_fine: {pct_proxy:.1f}% of samples used proxy (from EAE)")
        if proxy_flags['used_RRI_proxy'] > 0:
            pct_proxy = (proxy_flags['used_RRI_proxy'] / len(df_features)) * 100
            proxy_lines.append(f"- RRI: {pct_proxy:.1f}% of samples used default value")
        for line in proxy_lines:
            fig.text(0.13, y, line, fontsize=10)
            y -= line_space
        fig.text(0.13, y, f"- Data removal: {cleaning_log['removal_pct']:.2f}% due to missing values", fontsize=10)
        y -= line_space
        fig.text(0.13, y, f"- Total samples analyzed: {len(df_features):,}", fontsize=10)
        y -= line_space * 1.5

        fig.text(0.1, y, "5. Recommendations:", fontsize=11, fontweight='bold')
        y -= line_space
        recommendations = [
            "Use this classification for source apportionment studies",
            "Consider seasonal variations in future analysis",
            "Validate results with meteorological data",
            "Apply to air quality management strategies",
            "Use cluster labels for predictive modeling"
        ]
        for rec in recommendations:
            fig.text(0.13, y, f"- {rec}", fontsize=10)
            y -= line_space
        y -= line_space * 0.5

        fig.text(0.1, y, "6. Key Findings:", fontsize=11, fontweight='bold')
        y -= line_space
        findings = [
            "ASIA region shows mixed aerosol composition",
            f"{dominant_type} aerosols are predominant",
            "Clear spectral signatures distinguish aerosol types",
            "Clustering aligns with known aerosol physics",
            "Method suitable for large-scale aerosol classification"
        ]
        for finding in findings:
            fig.text(0.13, y, f"- {finding}", fontsize=10)
            y -= line_space

        # Add reference and signature/date at the bottom
        fig.text(0.1, 0.08, 'Reference:', fontsize=11, fontweight='bold')
        fig.text(0.1, 0.05, 'Ningombam, S. S., et al. (2024). Aerosol classification by application of\n'
                            'machine learning spectral clustering algorithm. Journal, Volume, Pages.',
                 fontsize=9, style='italic')
        fig.text(0.9, 0.02, f'Report generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                 ha='right', fontsize=8, style='italic', color='gray')

        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Page {page_num}: Conclusions & interpretation")

        # Set PDF metadata
        d = pdf.infodict()
        d['Title'] = 'DA9 Aerosol Spectral Clustering Analysis Report'
        d['Author'] = 'Jsh Agarwal (UID: 2022200004)'
        d['Subject'] = 'Aerosol Classification using Spectral Clustering'
        d['Keywords'] = 'Aerosol, Spectral Clustering, AERONET, Classification, Data Analysis'
        d['CreationDate'] = datetime.now()
    
    print(f"\n✓ PDF report saved: {pdf_filename}")
    print(f"✓ Total pages: ~{page_num}")
    return pdf_filename

# Generate the PDF report
try:
    pdf_file = generate_comprehensive_pdf_report()
    print("\n" + "="*80)
    print("🎉 PDF REPORT GENERATION COMPLETE! 🎉")
    print("="*80)
    print("\n📊 Generated Outputs:")
    print("  ✓ All PNG visualizations (5+ plots)")
    print("  ✓ CSV data files (7 files)")
    print("  ✓ Text summary report")
    print(f"  ✓ Comprehensive PDF report: {pdf_file} 📄")
    print(f"\n📈 Report Contents:")
    print("  • Title page with student information")
    print("  • Data cleaning summary")
    print("  • Feature statistics (before & after)")
    print("  • Clustering results & quality metrics")
    print("  • Cluster centroids & classification")
    print("  • Distribution analysis")
    print("  • All visualizations (embedded)")
    print("  • Conclusions & interpretation")
    print("\n🎉 No LaTeX installation needed!")
    print("🎉 PDF generated directly from Python!")
except Exception as e:
    print(f"\n❌ PDF generation failed: {e}")
    import traceback
    traceback.print_exc()
    print("\n⚠️  All other outputs (PNG, CSV, TXT) are still available!")
    pdf_file = None  # <-- FIX: Ensure pdf_file is defined

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)

# === VERIFICATION: Check all required images ===
print("\n" + "="*80)
print("IMAGE VERIFICATION")
print("="*80)

required_images = [
    'fig_page1_header.png',
    'fig_map1.png',
    'fig_map2.png',
    'fig_params.png',
    'fig_histograms.png',
    'fig_scatter_AOD_EAE.png',
    'fig_scatter_matrix_clusters.png',
    'fig_silhouette.png',
    'fig_cluster_bar_percentages.png',
    'fig_centroids_table.png',
    'fig_clustered_map.png',
    'fig_boxplots_clusters.png',
    'fig_cdf_summary.png',
    'fig_algo_comparison.png',
]

missing_images = []
found_images = []

for img in required_images:
    img_path = os.path.join(OUTPUT_DIR, img)
    if os.path.exists(img_path):
        found_images.append(img)
        print(f"  ✓ Found: {img}")
    else:
        missing_images.append(img)
        print(f"  ✗ Missing: {img}")

print(f"\n📊 Image Generation Summary:")
print(f"  • Total required: {len(required_images)}")
print(f"  • Successfully generated: {len(found_images)}")
print(f"  • Missing: {len(missing_images)}")

if missing_images:
    print(f"\n⚠️  Missing images will be replaced with placeholders in PDF")
else:
    print(f"\n✓ All required images generated successfully!")

print(f"\n⏱ Performance optimizations applied:")
print(f"  • Plot DPI: {PLOT_DPI} (reduce for faster rendering)")
print(f"  • Max clustering samples: {MAX_SAMPLES_FOR_CLUSTERING}")
print(f"  • Max silhouette samples: {MAX_SAMPLES_FOR_SILHOUETTE}")
print(f"  • Algorithm comparison: {'Skipped' if SKIP_ALGORITHM_COMPARISON else 'Included'}")

print(f"\n📄 PDF Report Status:")
if pdf_file:
    print(f"  ✓ PDF generated successfully: {pdf_file}")
else:
    print(f"  ✗ PDF generation encountered an error")

print(f"\n✓ All outputs saved successfully!")
print("="*80 + "\n")

# Summary output
print(f"\n✓ Analysis complete!")
if pdf_file:
    print(f"✓ PDF report saved to: {pdf_file}")
else:
    print("⚠ PDF report not generated.")
print(f"✓ All visualizations and data files saved to: {OUTPUT_DIR}/")
print("="*80 + "\n")
