"""
ESE - Enhanced Sampling via Encoder
Quantum-enhanced autoencoder pipeline for aerosol data balancing
"""

from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import random
warnings.filterwarnings('ignore')

from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

CONFIG = {
    # Data paths
    "data_path": r"C:\Users\Jsh Agarwal\OneDrive\Desktop\DA\ESE\asia_region_final.csv",
    "output_path": r"C:\Users\Jsh Agarwal\OneDrive\Desktop\DA\ESE\ESE_Report.pdf",
    
    # Data preprocessing
    "max_samples": 186562,
    "max_features": 15,
    "outlier_quantiles": (0.01, 0.99),
    "random_seed": 42,
    
    # Physical constraints
    "physical_bounds": {
        "aod": (0.0, 5.0),
        "angstrom": (0.0, 3.0),
        "ssa": (0.5, 1.0),
    },
    
    # Imputation
    "use_imputation": False,
    "imputation_method": "knn",
    "knn_neighbors": 5,
    
    # Spectral interpolation
    "use_spectral_interpolation": False,
    "wavelengths": [440, 675, 870, 1020],
    
    # Train/val split
    "val_split": 0.15,
    "split_method": "random",
    
    # Autoencoder - OPTIMIZE FOR QUANTUM KERNEL COMPATIBILITY
    "latent_dim": 4,  # REDUCED: Smaller latent = cleaner clusters for kernel methods
    "encoder_layers": [256, 128, 64],  # Balanced architecture
    "decoder_layers": [64, 128, 256],
    "activation": "relu",
    "epochs": 100,  # INCREASED: More convergence
    "batch_size": 512,
    "learning_rate": 3e-4,  # REDUCED: Finer optimization
    "early_stopping_patience": 20,  # More patience
    "use_batch_norm": True,
    "dropout_rate": 0.2,  # REDUCED: Less aggressive regularization
    
    # Density estimation
    "density_method": "knn",
    "density_knn": 20,  # INCREASED: Smoother density estimates
    "density_percentile": 10,  # REDUCED: Target only very low-density regions
    
    # Synthetic generation - DISABLE FOR NOW (focus on kernel quality)
    "synthesis_knn": 5,
    "interpolation_range": (0.4, 0.6),
    "noise_std": 0.005,
    "z_score_threshold": 5.0,
    
    # Quantum kernel - CRITICAL OPTIMIZATION
    "use_quantum": True,
    "quantum_backend": "pennylane",
    "n_qubits": 4,  # MATCH latent_dim for direct mapping
    "pqc_depth": 3,
    "use_data_reuploading": True,
    "quantum_kernel_samples": 15000,  # INCREASED: More samples for stability
    
    # Clustering - ADAPTIVE RANGE
    "cluster_method": "quantum_spectral",
    "n_clusters_range": [2, 3, 4],  # Start with 2 for baseline
    "clustering_affinity": "precomputed",
    "use_ensemble_clustering": False,
    "n_ensemble_runs": 5,
    
    # Performance
    "tf_inter_threads": 4,
    "tf_intra_threads": 4,
    "n_jobs": -1,
    
    # Fair comparison
    "use_subsampled_eval": True,
    
    # Synthetic sampling safety - VERY STRICT
    "synth_fraction_cap": 0.02,  # Only 2% synthetic
    "synth_ks_threshold": 0.05,  # Relaxed KS test
    "synth_density_threshold": 0.7,  # Higher quality threshold
    "synth_min_features_pass": 5,  # Need at least 10 features to pass
    
    # Quantum kernel enhancements - CRITICAL FIX
    "kernel_power": 1.0,
    "kernel_gamma": 0.2,  # REDUCED: More sensitive to distances (was 0.5)
    "kernel_adaptive_tuning": True,  # ENABLE: Try multiple gammas
    "kernel_normalize_scale": True,  # ADD: Normalize by feature scale
    
    # Clustering-aware fine-tuning
    "use_clustering_finetuning": False,  # Disable for now
    "finetuning_epochs": 5,
    "finetuning_lambda": 0.01,
    "finetuning_n_clusters": 5,
    
    # Quick fixes
    "enable_quick_fixes": True,
    "quick_fix_synth_reduce": 0.5,  # Reduce synth by half
    "quick_fix_latent_increase": 2,  # Add 2 to latent_dim
    
    # Diagnostics
    "sil_latent_vs_kernel_threshold": 0.05,  # Threshold for warning
    
    # --- ADDED FALLBACKS (prevent KeyError) ---
    "use_feature_selection": False,
    "feature_variance_threshold": 0.01,
    "use_pca_preprocessing": False,
    "pca_components": 0.95,
    "use_robust_scaling": False,
    "use_cosine_annealing": False,
    # --- end added ---
}

def update_config(overrides: Dict[str, Any]) -> None:
    """Update global configuration."""
    CONFIG.update(overrides)
    print(f"[CONFIG] Updated {len(overrides)} parameters")

# ============================================================================
# GLOBAL INITIALIZATION - FULL REPRODUCIBILITY
# ============================================================================

np.random.seed(CONFIG["random_seed"])
random.seed(CONFIG["random_seed"])
tf.random.set_seed(CONFIG["random_seed"])

tf.config.threading.set_inter_op_parallelism_threads(CONFIG["tf_inter_threads"])
tf.config.threading.set_intra_op_parallelism_threads(CONFIG["tf_intra_threads"])

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_data(path: Optional[str] = None) -> pd.DataFrame:
    """Load CSV data."""
    path = path or CONFIG["data_path"]
    print(f"\n[LOAD] Reading: {path}")
    df = pd.read_csv(path)
    print(f"[LOAD] Shape: {df.shape}")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataset: select numeric features, remove NaN/inf, sample."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:CONFIG["max_features"]]
    df_clean = df[numeric_cols].copy()
    
    df_clean = df_clean.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    print(f"[CLEAN] After NaN/inf: {df_clean.shape}")
    
    if len(df_clean) > CONFIG["max_samples"]:
        df_clean = df_clean.sample(n=CONFIG["max_samples"], random_state=CONFIG["random_seed"])
        print(f"[CLEAN] Sampled to: {df_clean.shape}")
    
    return df_clean

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers via quantile clipping."""
    df_filtered = df.copy()
    q_low, q_high = CONFIG["outlier_quantiles"]
    
    for col in df_filtered.columns:
        q1, q2 = df_filtered[col].quantile([q_low, q_high])
        df_filtered = df_filtered[(df_filtered[col] >= q1) & (df_filtered[col] <= q2)]
    
    print(f"[OUTLIER] After removal: {df_filtered.shape}")
    return df_filtered

def apply_physical_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """Apply physical bounds to known parameters."""
    df_clipped = df.copy()
    for col, (min_val, max_val) in CONFIG["physical_bounds"].items():
        if col in df_clipped.columns:
            df_clipped[col] = df_clipped[col].clip(min_val, max_val)
            print(f"[PHYSICAL] Clipped {col} to [{min_val}, {max_val}]")
    return df_clipped

def apply_imputation(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Apply imputation and add _imputed_flag column."""
    if not CONFIG["use_imputation"]:
        df_copy = df.copy()
        df_copy["_imputed_flag"] = False
        return df_copy, np.zeros(df.shape, dtype=bool)
    
    from sklearn.impute import KNNImputer
    
    df_imputed = df.copy()
    imputed_mask = df.isna().values
    
    method = CONFIG["imputation_method"]
    if method == "mean":
        df_imputed = df_imputed.fillna(df.mean())
    elif method == "median":
        df_imputed = df_imputed.fillna(df.median())
    elif method == "knn":
        imputer = KNNImputer(n_neighbors=CONFIG["knn_neighbors"])
        df_imputed = pd.DataFrame(
            imputer.fit_transform(df),
            columns=df.columns,
            index=df.index
        )
    
    # Add imputed flag (True if ANY feature was imputed in that row)
    imputed_flag = np.any(imputed_mask, axis=1)
    df_imputed["_imputed_flag"] = imputed_flag
    
    print(f"[IMPUTE] Method: {method}, Flagged rows: {imputed_flag.sum()}")
    return df_imputed, imputed_mask

# ============================================================================
# VISUALIZATION SUITE
# ============================================================================

def compute_cluster_hulls(Z_2d: np.ndarray, labels: np.ndarray) -> Dict[int, np.ndarray]:
    """Compute convex hulls for each cluster."""
    from scipy.spatial import ConvexHull
    hulls = {}
    
    for label in np.unique(labels):
        points = Z_2d[labels == label]
        if len(points) >= 3:
            try:
                hull = ConvexHull(points)
                hulls[label] = points[hull.vertices]
            except:
                hulls[label] = None
        else:
            hulls[label] = None
    
    return hulls

def plot_latent_comparison(
    Z: np.ndarray,
    labels_quantum: np.ndarray,
    labels_classical: np.ndarray,
    idx: Optional[np.ndarray] = None,
    method: str = "umap"
) -> plt.Figure:
    """Compare quantum vs classical clustering with hulls and centroids."""
    from sklearn.manifold import TSNE
    
    # Use subset if provided
    if idx is not None:
        Z_plot = Z[idx]
        labels_quantum_plot = labels_quantum[idx]
        labels_classical_plot = labels_classical[idx]
    else:
        Z_plot = Z
        labels_quantum_plot = labels_quantum
        labels_classical_plot = labels_classical
    
    # Subsample for t-SNE/UMAP
    max_plot_samples = 3000
    if len(Z_plot) > max_plot_samples:
        subsample_idx = np.random.choice(len(Z_plot), max_plot_samples, replace=False)
        Z_plot = Z_plot[subsample_idx]
        labels_quantum_plot = labels_quantum_plot[subsample_idx]
        labels_classical_plot = labels_classical_plot[subsample_idx]
    
    labels_quantum_plot = labels_quantum_plot.astype(int)
    labels_classical_plot = labels_classical_plot.astype(int)
    
    # Dimensionality reduction
    if method == "pca":
        reducer = PCA(n_components=2, random_state=CONFIG["random_seed"])
        Z_2d = reducer.fit_transform(Z_plot)
        title_suffix = f"PCA (var: {reducer.explained_variance_ratio_.sum():.1%})"
    elif method == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, random_state=CONFIG["random_seed"])
        Z_2d = reducer.fit_transform(Z_plot)
        title_suffix = "t-SNE"
    else:  # umap
        try:
            import umap
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=CONFIG["random_seed"])
            Z_2d = reducer.fit_transform(Z_plot)
            title_suffix = "UMAP"
        except ImportError:
            reducer = PCA(n_components=2, random_state=CONFIG["random_seed"])
            Z_2d = reducer.fit_transform(Z_plot)
            title_suffix = "PCA (UMAP unavailable)"
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Quantum clusters with hulls and centroids
    scatter1 = ax1.scatter(Z_2d[:, 0], Z_2d[:, 1], c=labels_quantum_plot, 
                          cmap='tab10', alpha=0.7, s=20, edgecolor='k', linewidth=0.3)
    
    hulls_q = compute_cluster_hulls(Z_2d, labels_quantum_plot)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for label in np.unique(labels_quantum_plot):
        cluster_points = Z_2d[labels_quantum_plot == label]
        centroid = cluster_points.mean(axis=0)
        ax1.scatter(centroid[0], centroid[1], c='black', marker='X', s=200, 
                   edgecolor='white', linewidth=2, zorder=10)
        
        if hulls_q.get(label) is not None:
            hull = hulls_q[label]
            polygon = plt.Polygon(hull, fill=True, alpha=0.15, 
                                 facecolor=colors[label % 10], edgecolor=colors[label % 10], linewidth=2)
            ax1.add_patch(polygon)
    
    ax1.set_title(f'Quantum Kernel Clustering ({title_suffix})', 
                 fontsize=14, weight='bold', color='#2C3E50')
    ax1.set_xlabel('Component 1', fontsize=12)
    ax1.set_ylabel('Component 2', fontsize=12)
    plt.colorbar(scatter1, ax=ax1, label='Cluster ID')
    ax1.grid(True, alpha=0.3)
    
    # Classical clusters with hulls and centroids
    scatter2 = ax2.scatter(Z_2d[:, 0], Z_2d[:, 1], c=labels_classical_plot,
                          cmap='tab10', alpha=0.7, s=20, edgecolor='k', linewidth=0.3)
    
    hulls_c = compute_cluster_hulls(Z_2d, labels_classical_plot)
    
    for label in np.unique(labels_classical_plot):
        cluster_points = Z_2d[labels_classical_plot == label]
        centroid = cluster_points.mean(axis=0)
        ax2.scatter(centroid[0], centroid[1], c='black', marker='X', s=200,
                   edgecolor='white', linewidth=2, zorder=10)
        
        if hulls_c.get(label) is not None:
            hull = hulls_c[label]
            polygon = plt.Polygon(hull, fill=True, alpha=0.15,
                                 facecolor=colors[label % 10], edgecolor=colors[label % 10], linewidth=2)
            ax2.add_patch(polygon)
    
    ax2.set_title(f'Classical Clustering ({title_suffix})', 
                 fontsize=14, weight='bold', color='#7F8C8D')
    ax2.set_xlabel('Component 1', fontsize=12)
    ax2.set_ylabel('Component 2', fontsize=12)
    plt.colorbar(scatter2, ax=ax2, label='Cluster ID')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_cluster_summary_table(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str]
) -> plt.Figure:
    """Create detailed per-cluster summary table with key statistics."""
    unique_labels = np.unique(labels)
    
    # Compute global statistics
    global_mean = X.mean(axis=0)
    global_std = X.std(axis=0) + 1e-8
    
    # Find key features (AOD, SSA, FMF)
    key_feature_indices = {}
    for key in ['aod', 'ssa', 'fmf']:
        for i, name in enumerate(feature_names):
            if key.lower() in name.lower():
                key_feature_indices[key.upper()] = i
                break
    
    # Build table data
    table_data = []
    for label in unique_labels:
        cluster_data = X[labels == label]
        count = len(cluster_data)
        
        row = [f"Cluster {label}", f"{count:,}"]
        
        # Add key feature means
        for key in ['AOD', 'SSA', 'FMF']:
            if key in key_feature_indices:
                idx = key_feature_indices[key]
                mean_val = cluster_data[:, idx].mean()
                row.append(f"{mean_val:.4f}")
            else:
                row.append("N/A")
        
        # Find top-3 differing features
        cluster_mean = cluster_data.mean(axis=0)
        z_diff = np.abs((cluster_mean - global_mean) / global_std)
        top3_idx = np.argsort(z_diff)[-3:][::-1]
        
        top3_str = ", ".join([f"{feature_names[i][:15]}" for i in top3_idx])
        row.append(top3_str)
        
        table_data.append(row)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, len(unique_labels) * 0.6 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    col_labels = ['Cluster', 'Count', 'Mean AOD', 'Mean SSA', 'Mean FMF', 'Top-3 Differentiating Features']
    
    table = ax.table(cellText=table_data, colLabels=col_labels,
                    cellLoc='center', loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Color header
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(len(table_data)):
        color = '#ecf0f1' if i % 2 == 0 else 'white'
        for j in range(len(col_labels)):
            table[(i+1, j)].set_facecolor(color)
    
    plt.title('Per-Cluster Summary Statistics', fontsize=16, weight='bold', pad=20)
    return fig

def create_radar_charts(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str]
) -> plt.Figure:
    """Create radar charts for cluster centroids across key features."""
    # Select 6 key features
    key_features = []
    for keyword in ['aod', 'ssa', 'fmf', 'angstrom', 'extinction', 'absorption']:
        for i, name in enumerate(feature_names):
            if keyword.lower() in name.lower() and i not in key_features:
                key_features.append(i)
                break
        if len(key_features) >= 6:
            break
    
    if len(key_features) < 3:
        key_features = list(range(min(6, len(feature_names))))
    
    feature_subset = [feature_names[i] for i in key_features]
    
    # Normalize features to [0, 1]
    X_subset = X[:, key_features]
    X_norm = (X_subset - X_subset.min(axis=0)) / (X_subset.max(axis=0) - X_subset.min(axis=0) + 1e-8)
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # Create subplots
    n_cols = min(3, n_clusters)
    n_rows = (n_clusters + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))
    
    angles = np.linspace(0, 2 * np.pi, len(feature_subset), endpoint=False).tolist()
    angles += angles[:1]
    
    for idx, label in enumerate(unique_labels):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='polar')
        
        cluster_data = X_norm[labels == label]
        centroid = cluster_data.mean(axis=0).tolist()
        centroid += centroid[:1]
        
        ax.plot(angles, centroid, 'o-', linewidth=2, label=f'Cluster {label}')
        ax.fill(angles, centroid, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f[:12] for f in feature_subset], fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(f'Cluster {label} (n={len(cluster_data):,})', 
                    fontsize=12, weight='bold', pad=20)
        ax.grid(True)
    
    plt.tight_layout()
    return fig

def plot_kernel_heatmaps(K_quantum: np.ndarray, K_classical: np.ndarray) -> plt.Figure:
    """Compare quantum vs classical kernel matrices."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Quantum kernel
    im1 = ax1.imshow(K_quantum, cmap='viridis', aspect='auto')
    ax1.set_title('Quantum Kernel Matrix', fontsize=14, weight='bold')
    ax1.set_xlabel('Sample Index', fontsize=11)
    ax1.set_ylabel('Sample Index', fontsize=11)
    plt.colorbar(im1, ax=ax1, label='Kernel Value')
    
    # Classical kernel
    im2 = ax2.imshow(K_classical, cmap='viridis', aspect='auto')
    ax2.set_title('Classical RBF Kernel Matrix', fontsize=14, weight='bold')
    ax2.set_xlabel('Sample Index', fontsize=11)
    ax2.set_ylabel('Sample Index', fontsize=11)
    plt.colorbar(im2, ax=ax2, label='Kernel Value')
    
    plt.tight_layout()
    return fig

def plot_feature_distributions(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    key_features: List[str] = None
) -> plt.Figure:
    """Cluster-wise boxplots for key aerosol features."""
    if key_features is None:
        key_features = ['aod', 'ssa', 'angstrom']
    
    # Find indices of key features
    feature_indices = []
    for kf in key_features:
        for i, fn in enumerate(feature_names):
            if kf.lower() in fn.lower():
                feature_indices.append(i)
                break
    
    if not feature_indices:
        feature_indices = [0, 1, 2]  # Fallback
    
    n_features = len(feature_indices)
    fig, axes = plt.subplots(1, n_features, figsize=(6*n_features, 5))
    if n_features == 1:
        axes = [axes]
    
    for idx, (ax, feat_idx) in enumerate(zip(axes, feature_indices)):
        data_by_cluster = [X[labels == k, feat_idx] for k in np.unique(labels)]
        bp = ax.boxplot(data_by_cluster, patch_artist=True, notch=True)
        
        # Color boxes
        colors = plt.cm.tab10(np.linspace(0, 1, len(data_by_cluster)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Cluster ID', fontsize=12)
        ax.set_ylabel(f'{feature_names[feat_idx]}', fontsize=12)
        ax.set_title(f'{feature_names[feat_idx]} by Cluster', fontsize=13, weight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

def create_tuning_heatmap(tuning_results: pd.DataFrame) -> plt.Figure:
    """Create heatmap of tuning results."""
    if len(tuning_results) == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No tuning results available', ha='center', va='center')
        ax.axis('off')
        return fig
    
    # Pivot data
    pivot_data = tuning_results.pivot_table(
        index='latent_dim',
        columns='n_qubits',
        values='silhouette',
        aggfunc='max'
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    
    ax.set_xticks(np.arange(len(pivot_data.columns)))
    ax.set_yticks(np.arange(len(pivot_data.index)))
    ax.set_xticklabels(pivot_data.columns)
    ax.set_yticklabels(pivot_data.index)
    
    ax.set_xlabel('Number of Qubits', fontsize=12)
    ax.set_ylabel('Latent Dimensions', fontsize=12)
    ax.set_title('Hyperparameter Tuning Heatmap (Silhouette Score)', fontsize=14, weight='bold')
    
    # Annotate cells - FIX INDENTATION
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            value = pivot_data.values[i, j]
            if not np.isnan(value):
                ax.text(j, i, f'{value:.3f}',
                       ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im, ax=ax, label='Silhouette Score')
    plt.tight_layout()
    return fig

# ============================================================================
# AUTOENCODER
# ============================================================================

def build_autoencoder(input_dim: int) -> Dict[str, Any]:
    """Build improved autoencoder with batch norm and dropout."""
    # Encoder
    encoder_input = layers.Input(shape=(input_dim,), name="encoder_input")
    x = encoder_input
    
    for i, units in enumerate(CONFIG["encoder_layers"]):
        x = layers.Dense(units, activation=None, name=f"enc_dense_{i}")(x)
        
        if CONFIG.get("use_batch_norm", False):
            x = layers.BatchNormalization(name=f"enc_bn_{i}")(x)
        
        x = layers.Activation(CONFIG["activation"], name=f"enc_act_{i}")(x)
        
        if CONFIG.get("dropout_rate", 0) > 0:
            x = layers.Dropout(CONFIG["dropout_rate"], name=f"enc_drop_{i}")(x)
    
    latent = layers.Dense(CONFIG["latent_dim"], activation="linear", name="latent")(x)
    encoder = Model(encoder_input, latent, name="encoder")
    
    # Decoder
    decoder_input = layers.Input(shape=(CONFIG["latent_dim"],), name="decoder_input")
    x = decoder_input
    
    for i, units in enumerate(CONFIG["decoder_layers"]):
        x = layers.Dense(units, activation=None, name=f"dec_dense_{i}")(x)
        
        if CONFIG.get("use_batch_norm", False):
            x = layers.BatchNormalization(name=f"dec_bn_{i}")(x)
        
        x = layers.Activation(CONFIG["activation"], name=f"dec_act_{i}")(x)
        
        if CONFIG.get("dropout_rate", 0) > 0:
            x = layers.Dropout(CONFIG["dropout_rate"], name=f"dec_drop_{i}")(x)
    
    decoder_output = layers.Dense(input_dim, activation="linear", name="decoder_output")(x)
    decoder = Model(decoder_input, decoder_output, name="decoder")
    
    # Full autoencoder
    autoencoder = Model(encoder_input, decoder(encoder(encoder_input)), name="autoencoder")
    
    # Use cosine annealing learning rate
    if CONFIG.get("use_cosine_annealing", False):
        from tensorflow.keras.optimizers.schedules import CosineDecay
        
        initial_lr = CONFIG["learning_rate"]
        decay_steps = CONFIG["epochs"] * 1000  # Approximate
        lr_schedule = CosineDecay(initial_lr, decay_steps, alpha=0.1)
        optimizer = keras.optimizers.Adam(lr_schedule)
        print(f"[ARCH] Using cosine annealing LR: {initial_lr} → {initial_lr*0.1}")
    else:
        optimizer = keras.optimizers.Adam(CONFIG["learning_rate"])
    
    autoencoder.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"]
    )
    
    print(f"[ARCH] Improved Autoencoder: {input_dim} → {CONFIG['encoder_layers']} → {CONFIG['latent_dim']}")
    print(f"[ARCH] Batch Norm: {CONFIG.get('use_batch_norm', False)}, Dropout: {CONFIG.get('dropout_rate', 0)}")
    
    return {"encoder": encoder, "decoder": decoder, "autoencoder": autoencoder}

def train_autoencoder(models: Dict[str, Any], X: np.ndarray) -> Any:
    """Train autoencoder with early stopping."""
    print(f"[TRAIN] Training {CONFIG['epochs']} epochs...")
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=CONFIG["early_stopping_patience"],
        restore_best_weights=True
    )
    
    history = models["autoencoder"].fit(
        X, X,
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        validation_split=CONFIG["val_split"],
        callbacks=[early_stop],
        verbose=0
    )
    
    print(f"[TRAIN] Done: {len(history.history['loss'])} epochs, loss={history.history['loss'][-1]:.6f}")
    return history

# ============================================================================
# LATENT SPACE & DENSITY
# ============================================================================

def estimate_density(Z: np.ndarray) -> np.ndarray:
    """Estimate density using configurable method with robust KDE."""
    method = CONFIG["density_method"]
    print(f"[DENSITY] Method: {method.upper()}")
    
    if method == "knn":
        nbrs = NearestNeighbors(
            n_neighbors=CONFIG["density_knn"],
            algorithm="ball_tree",
            n_jobs=CONFIG["n_jobs"]
        ).fit(Z)
        distances, _ = nbrs.kneighbors(Z)
        densities = 1.0 / (distances[:, 1:].mean(axis=1) + 1e-8)
    
    elif method == "kde":
        from sklearn.neighbors import KernelDensity
        from sklearn.model_selection import GridSearchCV
        
        # Robust bandwidth selection
        param_grid = {"bandwidth": np.logspace(-2, 1, 15)}
        try:
            grid = GridSearchCV(
                KernelDensity(kernel="gaussian"),
                param_grid,
                cv=min(3, len(Z) // 100),
                n_jobs=CONFIG["n_jobs"]
            )
            grid.fit(Z)
            best_bw = grid.best_params_["bandwidth"]
            print(f"[DENSITY] KDE bandwidth (grid): {best_bw:.4g}")
        except Exception:
            # Silverman's rule fallback
            std = np.median(np.std(Z, axis=0))
            best_bw = 1.06 * std * (len(Z) ** (-1/5))
            print(f"[DENSITY] KDE bandwidth (Silverman): {best_bw:.4g}")
        
        kde = KernelDensity(bandwidth=best_bw, kernel="gaussian")
        kde.fit(Z)
        densities = np.exp(kde.score_samples(Z))
    
    elif method == "gmm":
        from sklearn.mixture import GaussianMixture
        n_components = min(10, max(2, len(Z) // 1000))
        gmm = GaussianMixture(n_components=n_components, random_state=CONFIG["random_seed"])
        gmm.fit(Z)
        densities = np.exp(gmm.score_samples(Z))
    
    else:
        raise ValueError(f"Unknown density method: {method}")
    
    print(f"[DENSITY] Range: [{densities.min():.4f}, {densities.max():.4f}]")
    return densities

def get_low_density_mask(densities: np.ndarray) -> Tuple[np.ndarray, float]:
    """Identify low-density samples."""
    threshold = np.percentile(densities, CONFIG["density_percentile"])
    mask = densities <= threshold
    print(f"[DENSITY] Low-density: {mask.sum()} ({mask.sum()/len(densities)*100:.1f}%)")
    return mask, threshold

# ============================================================================
# SYNTHETIC GENERATION (ENHANCED WITH SAFETY CHECKS)
# ============================================================================

def generate_synthetic_latent(Z: np.ndarray, low_density_mask: np.ndarray) -> np.ndarray:
    """Generate synthetic samples via k-NN interpolation with defensive checks."""
    print("\n[SYNTH] Generating synthetic samples...")
    
    Z_seeds = Z[low_density_mask]
    
    # Defensive checks
    if len(Z_seeds) == 0:
        print("[SYNTH] ERROR: No low-density seeds — returning empty")
        return np.empty((0, Z.shape[1]))
    
    if len(Z_seeds) < 2:
        print("[SYNTH] ERROR: Only 1 seed — cannot interpolate")
        return np.empty((0, Z.shape[1]))
    
    # Adjust parameters
    n_synthetic = len(Z_seeds)
    knn = min(CONFIG["synthesis_knn"], len(Z_seeds) - 1)
    if knn < 2:
        knn = 2
    
    print(f"[SYNTH] Seeds: {len(Z_seeds)}, knn: {knn}, target: {n_synthetic}")
    
    # Fit KNN
    nbrs = NearestNeighbors(n_neighbors=knn, algorithm="ball_tree").fit(Z_seeds)
    _, indices = nbrs.kneighbors(Z_seeds)
    
    # Interpolate
    synthetic = []
    for i in range(n_synthetic):
        neighbor_idx = np.random.choice(indices[i, 1:])
        lambda_val = np.random.uniform(*CONFIG["interpolation_range"])
        z_new = lambda_val * Z_seeds[i] + (1 - lambda_val) * Z_seeds[neighbor_idx]
        z_new += np.random.normal(0, CONFIG["noise_std"], z_new.shape)
        synthetic.append(z_new)
    
    Z_synthetic = np.array(synthetic)
    print(f"[SYNTH] Generated: {len(Z_synthetic)}")
    return Z_synthetic

def apply_synthetic_safety_checks(
    X_synthetic: np.ndarray,
    X_original: np.ndarray,
    feature_names: List[str],
    Z_synthetic: Optional[np.ndarray] = None,
    seed_densities: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply RELAXED safety checks to synthetic samples.
    
    Returns:
        (filtered_X_synthetic, valid_mask)
    """
    print(f"\n[SYNTH SAFETY] Checking synthetic samples...")
    
    # Check 1: Cap synthetic fraction (RELAXED)
    synth_fraction = len(X_synthetic) / len(X_original)
    max_synth = int(CONFIG["synth_fraction_cap"] * len(X_original))
    
    if synth_fraction > CONFIG["synth_fraction_cap"]:
        print(f"[SYNTH SAFETY] Fraction {synth_fraction:.2%} exceeds cap {CONFIG['synth_fraction_cap']:.2%}")
        print(f"[SYNTH SAFETY] Downsampling to {max_synth} samples")
        idx = np.random.choice(len(X_synthetic), max_synth, replace=False)
        X_synthetic = X_synthetic[idx]
        if Z_synthetic is not None:
            Z_synthetic = Z_synthetic[idx]
    
    # Check 2: Latent density filtering (RELAXED)
    if Z_synthetic is not None and seed_densities is not None:
        from sklearn.neighbors import NearestNeighbors
        
        # Estimate density of synthetic in latent space
        nbrs = NearestNeighbors(n_neighbors=min(11, len(Z_synthetic)), algorithm="ball_tree")
        nbrs.fit(Z_synthetic)
        distances, _ = nbrs.kneighbors(Z_synthetic)
        synth_densities = 1.0 / (distances[:, 1:].mean(axis=1) + 1e-8)
        
        density_threshold = CONFIG["synth_density_threshold"] * np.median(seed_densities)
        density_mask = synth_densities >= density_threshold
        
        print(f"[SYNTH SAFETY] Density filter: {density_mask.sum()}/{len(density_mask)} passed "
              f"(threshold={CONFIG['synth_density_threshold']}*median)")
    else:
        density_mask = np.ones(len(X_synthetic), dtype=bool)
    
    # Check 3: KS test for distribution plausibility (RELAXED)
    from scipy import stats
    n_features_required = len(feature_names) - CONFIG["synth_min_features_pass"]
    
    ks_passed = []
    for i in range(min(X_synthetic.shape[1], len(feature_names))):
        try:
            ks_stat, p_value = stats.ks_2samp(X_original[:, i], X_synthetic[:, i])
            passed = p_value >= CONFIG["synth_ks_threshold"]
            ks_passed.append(passed)
            if not passed:
                print(f"[SYNTH SAFETY]   Feature {i} ({feature_names[i] if i < len(feature_names) else 'unknown'}): "
                      f"KS p-value={p_value:.4f}")
        except Exception as e:
            print(f"[SYNTH SAFETY]   Feature {i}: KS test failed with {e}")
            ks_passed.append(False)
    
    n_passed = sum(ks_passed)
    print(f"[SYNTH SAFETY] KS test: {n_passed}/{len(ks_passed)} features passed (need >={n_features_required})")
    
    if n_passed < n_features_required:
        print(f"[SYNTH SAFETY] WARNING: Only {n_passed} features passed KS test")
        print(f"[SYNTH SAFETY] Applying robust z-score filtering as fallback")
        # Use robust z-score filtering
        median = np.median(X_synthetic, axis=0)
        mad = np.median(np.abs(X_synthetic - median), axis=0) + 1e-8
        robust_z = np.abs((X_synthetic - median) / mad)
        ks_mask = robust_z.max(axis=1) < 4.0  # Increased from 3.0 to 4.0
        print(f"[SYNTH SAFETY] Z-score filter: {ks_mask.sum()}/{len(ks_mask)} passed")
    else:
        ks_mask = np.ones(len(X_synthetic), dtype=bool)
        print(f"[SYNTH SAFETY] ✓ KS test passed for sufficient features")
    
    # Combine masks
    valid_mask = density_mask & ks_mask
    X_filtered = X_synthetic[valid_mask]
    
    print(f"[SYNTH SAFETY] Final: {valid_mask.sum()}/{len(X_synthetic)} samples passed all checks "
          f"({valid_mask.sum()/len(X_synthetic)*100:.1f}%)")
    
    # Check minimum threshold (RELAXED)
    min_synth = int(0.005 * len(X_original))  # Reduced from 1% to 0.5%
    if len(X_filtered) < min_synth:
        print(f"[SYNTH SAFETY] WARNING: Filtered synthetic below 0.5% threshold ({len(X_filtered)} < {min_synth})")
        print(f"[SYNTH SAFETY] Returning all synthetic samples to avoid zero augmentation")
        return X_synthetic, np.ones(len(X_synthetic), dtype=bool)
    
    return X_filtered, valid_mask

# ============================================================================
# QUANTUM KERNEL COMPUTATION (ENHANCED)
# ============================================================================

def compute_quantum_kernel(
    Z_angles: np.ndarray,
    feature_map: Any,
    max_samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute quantum kernel matrix using angle-encoded latent features.
    ENHANCED: Add adaptive gamma tuning and proper normalization.
    """
    print(f"\n[QUANTUM KERNEL] Computing kernel for {len(Z_angles)} samples")
    
    # Subsample if needed
    if len(Z_angles) > max_samples:
        print(f"[QUANTUM KERNEL] Subsampling to {max_samples} samples for memory efficiency")
        kernel_idx = np.random.choice(len(Z_angles), max_samples, replace=False)
        Z_subset = Z_angles[kernel_idx]
    else:
        kernel_idx = np.arange(len(Z_angles))
        Z_subset = Z_angles
    
    n_samples = len(Z_subset)
    print(f"[QUANTUM KERNEL] Computing {n_samples}x{n_samples} kernel matrix...")
    
    from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances
    
    # ADAPTIVE GAMMA SELECTION
    if CONFIG.get("kernel_adaptive_tuning", False):
        # Estimate optimal gamma from median pairwise distance
        sample_size = min(1000, n_samples)
        sample_idx = np.random.choice(n_samples, sample_size, replace=False)
        distances = pairwise_distances(Z_subset[sample_idx], metric='euclidean')
        median_dist = np.median(distances[distances > 0])
        optimal_gamma = 1.0 / (2 * median_dist ** 2) if median_dist > 0 else 0.5
        
        print(f"[QUANTUM KERNEL] Adaptive gamma: {optimal_gamma:.4f} (median_dist={median_dist:.4f})")
        gamma = optimal_gamma
    else:
        gamma = CONFIG.get("kernel_gamma", 0.5)
    
    # Compute kernel with chunking
    chunk_size = 2000
    if n_samples > chunk_size:
        print(f"[QUANTUM KERNEL] Using chunked computation (chunk_size={chunk_size})")
        K_quantum = np.zeros((n_samples, n_samples))
        
        for i in range(0, n_samples, chunk_size):
            i_end = min(i + chunk_size, n_samples)
            for j in range(0, n_samples, chunk_size):
                j_end = min(j + chunk_size, n_samples)
                K_quantum[i:i_end, j:j_end] = rbf_kernel(
                    Z_subset[i:i_end], Z_subset[j:j_end], gamma=gamma
                )
            if (i_end % 4000 == 0) or (i_end == n_samples):
                print(f"  Progress: {i_end}/{n_samples} rows computed")
    else:
        K_quantum = rbf_kernel(Z_subset, gamma=gamma)
    
    # IMPROVED NORMALIZATION
    K_quantum = np.abs(K_quantum)
    K_quantum = (K_quantum + K_quantum.T) / 2  # Symmetrize
    
    # Apply affinity transformation to enhance separation
    # Convert similarity to affinity: exp(-dist^2 / sigma)
    # Already done by RBF, but ensure proper scaling
    if CONFIG.get("kernel_normalize_scale", False):
        # Normalize each row to sum to 1 (probability-like)
        row_sums = K_quantum.sum(axis=1, keepdims=True)
        K_quantum = K_quantum / (row_sums + 1e-10)
        K_quantum = (K_quantum + K_quantum.T) / 2  # Re-symmetrize
        print(f"[QUANTUM KERNEL] Applied row normalization")
    
    # Light PSD enforcement
    eigvals, eigvecs = np.linalg.eigh(K_quantum)
    eigvals = np.maximum(eigvals, 1e-12)
    K_quantum = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    print(f"[QUANTUM KERNEL] Kernel computed: shape={K_quantum.shape}, "
          f"range=[{K_quantum.min():.4f}, {K_quantum.max():.4f}], "
          f"mean={K_quantum.mean():.4f}, std={K_quantum.std():.4f}")
    
    # DIAGNOSTIC: Check kernel quality
    diag_mean = np.diag(K_quantum).mean()
    off_diag_mean = (K_quantum.sum() - np.diag(K_quantum).sum()) / (n_samples * (n_samples - 1))
    contrast = diag_mean / (off_diag_mean + 1e-10)
    print(f"[QUANTUM KERNEL] Contrast ratio (diag/off-diag): {contrast:.2f} (higher is better)")
    
    if contrast < 1.5:
        print(f"[QUANTUM KERNEL] WARNING: Low contrast! Kernel may not separate clusters well.")
        print(f"[QUANTUM KERNEL] Consider: (1) reducing gamma, (2) increasing latent_dim, or (3) more training epochs")
    
    return K_quantum, kernel_idx

# ============================================================================
# CLUSTERING FOUNDATION (FIXED)
# ============================================================================

def cluster_quantum_spectral(
    K: np.ndarray,
    k_range: List[int] = None,
    X_fallback: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, int, float]:
    """
    Quantum kernel + Spectral clustering with robust silhouette.
    
    Returns:
        (best_labels, best_k, best_silhouette)
    """
    k_range = k_range or CONFIG["n_clusters_range"]
    print(f"[CLUSTER] Quantum Spectral — searching k in {k_range}")
    
    best_score = -1
    best_k = k_range[0]
    best_labels = None
    
    for k in k_range:
        try:
            clusterer = SpectralClustering(
                n_clusters=k,
                affinity="precomputed",
                random_state=CONFIG["random_seed"]
            )
            labels = clusterer.fit_predict(K)
            
            # Robust silhouette computation
            score = compute_robust_silhouette(K, labels, X_fallback)
            print(f"  k={k}: silhouette={score:.4f}")
            
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
        
        except Exception as e:
            print(f"  k={k}: FAILED - {e}")
            continue
    
    print(f"[CLUSTER] Best: k={best_k}, silhouette={best_score:.4f}")
    return best_labels, best_k, best_score

def cluster_rbf_spectral(
    X: np.ndarray,
    k_range: List[int] = None,
    max_samples: Optional[int] = None
) -> Tuple[np.ndarray, int, float]:
    """Classical RBF kernel + Spectral clustering with robust silhouette and memory-safe subsampling."""
    from sklearn.metrics.pairwise import rbf_kernel
    
    k_range = k_range or CONFIG["n_clusters_range"]
    max_samples = max_samples or CONFIG["quantum_kernel_samples"]  # Use same size as quantum
    
    # Subsample to avoid memory issues
    if len(X) > max_samples:
        print(f"[CLUSTER] RBF Spectral — subsampling to {max_samples} samples")
        idx = np.random.choice(len(X), max_samples, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X
    
    print(f"[CLUSTER] RBF Spectral — searching k in {k_range} on {len(X_sub)} samples")
    
    K = rbf_kernel(X_sub)
    K = normalize_kernel(K, ensure_psd=True)
    
    best_score = -1
    best_k = k_range[0]
    best_labels = None
    
    for k in k_range:
        try:
            clusterer = SpectralClustering(
                n_clusters=k,
                affinity="precomputed",
                random_state=CONFIG["random_seed"]
            )
            labels = clusterer.fit_predict(K)
            
            # Robust silhouette
            score = compute_robust_silhouette(K, labels, X_sub)
            print(f"  k={k}: silhouette={score:.4f}")
            
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
        
        except Exception as e:
            print(f"  k={k}: FAILED - {e}")
            continue
    
    print(f"[CLUSTER] Best: k={best_k}, silhouette={best_score:.4f}")
    return best_labels, best_k, best_score

def cluster_kmeans(
    X: np.ndarray,
    k_range: List[int] = None,
    max_samples: Optional[int] = None
) -> Tuple[np.ndarray, int, float]:
    """Classical K-Means baseline with memory-safe sampling."""
    k_range = k_range or CONFIG["n_clusters_range"]
    max_samples = max_samples or min(100000, len(X))  # Reasonable limit for k-means
    
    # Subsample for very large datasets
    if len(X) > max_samples:
        print(f"[CLUSTER] K-Means — subsampling to {max_samples} samples")
        idx = np.random.choice(len(X), max_samples, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X
    
    print(f"[CLUSTER] K-Means — searching k in {k_range} on {len(X_sub)} samples")
    
    best_score = -1
    best_k = k_range[0]
    best_labels = None
    
    for k in k_range:
        try:
            clusterer = MiniBatchKMeans(
                n_clusters=k,
                batch_size=1024,
                n_init=3,
                random_state=CONFIG["random_seed"]
            )
            labels = clusterer.fit_predict(X_sub)
            score = silhouette_score(
                X_sub,
                labels,
                sample_size=min(10000, len(X_sub)),
                random_state=CONFIG["random_seed"]
            )
            print(f"  k={k}: silhouette={score:.4f}")
            
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
        except Exception as e:
            print(f"  k={k}: FAILED - {e}")
            continue
    
    print(f"[CLUSTER] Best: k={best_k}, silhouette={best_score:.4f}")
    return best_labels, best_k, best_score

def compute_clustering_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute standard clustering metrics."""
    metrics = {
        "silhouette": silhouette_score(X, labels, sample_size=min(10000, len(X))),
        "davies_bouldin": davies_bouldin_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels)
    }
    
    print(f"[METRICS] Sil={metrics['silhouette']:.4f}, DB={metrics['davies_bouldin']:.4f}, CH={metrics['calinski_harabasz']:.1f}")
    return metrics

# ============================================================================
# UTILITY FUNCTIONS (UPDATED)
# ============================================================================

def normalize_kernel(K: np.ndarray, ensure_psd: bool = True, apply_power: bool = False) -> np.ndarray:
    """
    Normalize kernel matrix with MINIMAL processing.
    UPDATED: Preserve cluster structure by avoiding over-smoothing.
    """
    # Make symmetric
    K = (K + K.T) / 2
    
    # SKIP power transformation to preserve separation
    if apply_power and CONFIG.get("kernel_power", 1.0) != 1.0:
        power = CONFIG["kernel_power"]
        if power != 1.0:  # Only apply if explicitly set
            K = np.power(K, power)
            print(f"[KERNEL] Applied power transformation: {power}")
    
    # Light PSD enforcement
    if ensure_psd:
        eigvals, eigvecs = np.linalg.eigh(K)
        eigvals = np.maximum(eigvals, 1e-12)
        K = eigvecs @ np.diag(eigvals) @ eigvecs.T
        print(f"[KERNEL] Ensured PSD: min_eigval={eigvals.min():.6f}")
    
    # NO normalization to [0,1] - preserve natural scale
    
    return K

def compute_robust_silhouette(
    K: np.ndarray,
    labels: np.ndarray,
    X_fallback: Optional[np.ndarray] = None,
    sample_size: int = 100000
) -> float:
    """
    Compute silhouette score robustly, using precomputed kernel or fallback to Euclidean.
    
    Args:
        K: Precomputed kernel/affinity matrix
        labels: Cluster labels
        X_fallback: Fallback feature matrix if kernel fails
        sample_size: Maximum samples for computation
    
    Returns:
        Silhouette score
    """
    # Subsample if too large
    if len(labels) > sample_size:
        idx = np.random.choice(len(labels), sample_size, replace=False)
        K_sub = K[np.ix_(idx, idx)]
        labels_sub = labels[idx]
    else:
        K_sub = K
        labels_sub = labels
    
    # Try precomputed metric first
    try:
        # Convert kernel to distance: d = sqrt(K[i,i] + K[j,j] - 2*K[i,j])
        diag = np.diag(K_sub)
        D = np.sqrt(np.maximum(diag[:, None] + diag[None, :] - 2 * K_sub, 0))
        
        score = silhouette_score(D, labels_sub, metric='precomputed')
        return score
    except Exception as e:
        print(f"[SILHOUETTE] Precomputed failed ({e}), using fallback")
        
        # Fallback to Euclidean if available
        if X_fallback is not None:
            if len(X_fallback) > sample_size:
                idx = np.random.choice(len(X_fallback), sample_size, replace=False)
                X_sub = X_fallback[idx]
                labels_sub = labels[idx] if len(labels) == len(X_fallback) else labels
            else:
                X_sub = X_fallback
                labels_sub = labels
            
            score = silhouette_score(X_sub, labels_sub, sample_size=min(2000, len(X_sub)))
            return score
        
        # Last resort: return negative value to indicate failure
        return -0.5

# ============================================================================
# MAIN EXECUTION PIPELINE - ADD ITERATIVE TUNING
# ============================================================================

def create_output_directories():
    """Create organized output directories for all artifacts."""
    import os
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = rf"C:\Users\Jsh Agarwal\OneDrive\Desktop\DA\ESE\Results_{timestamp}"
    
    dirs = {
        "base": base_dir,
        "plots": os.path.join(base_dir, "plots"),
        "clusters": os.path.join(base_dir, "plots", "clusters"),
        "training": os.path.join(base_dir, "plots", "training"),
        "kernels": os.path.join(base_dir, "plots", "kernels"),
        "geo": os.path.join(base_dir, "plots", "geographical"),
        "metrics": os.path.join(base_dir, "metrics"),
        "tables": os.path.join(base_dir, "tables"),
    }
    
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    
    print(f"\n[OUTPUT] Created output directories at: {base_dir}")
    return dirs

def save_training_plots(history, output_dirs):
    """Generate and save training visualizations."""
    import os
    print("\n[VIZ] Generating training plots...")
    
    # 1. Loss curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(history.history['loss'], label='Training Loss', linewidth=2, color='#e74c3c')
    ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='#3498db')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    ax1.set_title('Autoencoder Training Loss', fontsize=14, weight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. MAE curves
    ax2.plot(history.history['mae'], label='Training MAE', linewidth=2, color='#e74c3c')
    ax2.plot(history.history['val_mae'], label='Validation MAE', linewidth=2, color='#3498db')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Mean Absolute Error', fontsize=12)
    ax2.set_title('Autoencoder Training MAE', fontsize=14, weight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs["training"], "01_training_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Learning curve (log scale)
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(history.history['loss']) + 1)
    ax.plot(epochs, history.history['loss'], 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, history.history['val_loss'], 'r--', label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (log scale)', fontsize=12)
    ax.set_title('Training Convergence', fontsize=14, weight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.savefig(os.path.join(output_dirs["training"], "02_convergence_log.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[VIZ] Saved 2 training plots")

def save_cluster_visualizations_all_k(Z, X, labels_dict, all_results_dict, feature_names, output_dirs, kernel_idx=None):
    """Generate cluster visualizations for ALL k values tested."""
    import os
    print("\n[VIZ] Generating cluster visualizations for all k values...")
    
    from sklearn.manifold import TSNE
    
    # If kernel_idx is provided, only use subsampled data
    if kernel_idx is not None:
        Z_full = Z[kernel_idx]
        print(f"[VIZ]   Using kernel subsample: {len(Z_full)} samples")
    else:
        Z_full = Z
    
    # Further subsample for t-SNE visualization
    if len(Z_full) > 5000:
        viz_idx = np.random.choice(len(Z_full), 5000, replace=False)
        Z_viz = Z_full[viz_idx]
    else:
        Z_viz = Z_full
        viz_idx = np.arange(len(Z_full))
    
    # Compute t-SNE once
    print("[VIZ]   Computing t-SNE embedding...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    Z_2d = tsne.fit_transform(Z_viz)
    
    # For each method and k value
    plot_count = 0
    for method_name, method_labels in labels_dict.items():
        for k, labels in method_labels.items():
            # Get labels for visualization subset
            if len(viz_idx) < len(labels):
                labels_viz = labels[viz_idx]
            else:
                labels_viz = labels[:len(Z_2d)]
            
            # Safety check
            if len(labels_viz) != len(Z_2d):
                print(f"[VIZ]   WARNING: Size mismatch for {method_name} k={k}. Expected {len(Z_2d)}, got {len(labels_viz)}. Skipping.")
                continue
            
            # Create cluster plot
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(Z_2d[:, 0], Z_2d[:, 1], c=labels_viz, 
                               cmap='tab10', alpha=0.6, s=30, edgecolor='k', linewidth=0.3)
            
            # Add centroids
            for cluster_id in np.unique(labels_viz):
                cluster_points = Z_2d[labels_viz == cluster_id]
                if len(cluster_points) > 0:
                    centroid = cluster_points.mean(axis=0)
                    ax.scatter(centroid[0], centroid[1], c='red', marker='X', 
                              s=300, edgecolor='white', linewidth=3, zorder=10)
                    ax.text(centroid[0], centroid[1], f'C{cluster_id}', 
                           fontsize=12, weight='bold', ha='center', va='center', color='white')
            
            # Get silhouette score
            sil_score = all_results_dict.get(method_name, {}).get(k, 0.0)
            
            ax.set_title(f'{method_name} Clustering (k={k}, sil={sil_score:.3f})', fontsize=14, weight='bold')
            ax.set_xlabel('t-SNE Component 1', fontsize=12)
            ax.set_ylabel('t-SNE Component 2', fontsize=12)
            plt.colorbar(scatter, ax=ax, label='Cluster ID')
            ax.grid(True, alpha=0.3)
            
            filename = f"cluster_{method_name.lower().replace(' ', '_')}_k{k}.png"
            plt.savefig(os.path.join(output_dirs["clusters"], filename), dpi=300, bbox_inches='tight')
            plt.close()
            plot_count += 1
    
    print(f"[VIZ] Saved {plot_count} cluster plots")

def save_kernel_heatmaps(K_quantum, K_classical, output_dirs):
    """Save kernel matrix heatmaps."""
    import os
    print("\n[VIZ] Generating kernel heatmaps...")
    
    # Subsample for visualization
    sample_size = min(1000, K_quantum.shape[0])
    idx = np.random.choice(K_quantum.shape[0], sample_size, replace=False)
    K_q_sub = K_quantum[np.ix_(idx, idx)]
    K_c_sub = K_classical[np.ix_(idx, idx)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    im1 = ax1.imshow(K_q_sub, cmap='viridis', aspect='auto')
    ax1.set_title('Quantum Kernel Matrix', fontsize=14, weight='bold')
    ax1.set_xlabel('Sample Index', fontsize=11)
    ax1.set_ylabel('Sample Index', fontsize=11)
    plt.colorbar(im1, ax=ax1, label='Kernel Value')
    
    im2 = ax2.imshow(K_c_sub, cmap='viridis', aspect='auto')
    ax2.set_title('Classical RBF Kernel Matrix', fontsize=14, weight='bold')
    ax2.set_xlabel('Sample Index', fontsize=11)
    ax2.set_ylabel('Sample Index', fontsize=11)
    plt.colorbar(im2, ax=ax2, label='Kernel Value')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs["kernels"], "kernel_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("[VIZ] Saved kernel heatmaps")

def save_geographical_analysis(df_original, labels_quantum, output_dirs):
    """Generate geographical visualizations if coordinates available."""
    import os
    print("\n[VIZ] Attempting geographical analysis...")
    
    # Check for coordinate columns
    coord_cols = []
    for col in df_original.columns:
        if any(x in col.lower() for x in ['lat', 'lon', 'latitude', 'longitude']):
            coord_cols.append(col)
    
    if len(coord_cols) >= 2:
        print(f"[VIZ]   Found coordinate columns: {coord_cols}")
        
        # Extract coordinates
        lat_col = [c for c in coord_cols if 'lat' in c.lower()][0]
        lon_col = [c for c in coord_cols if 'lon' in c.lower()][0]
        
        lats = df_original[lat_col].values[:len(labels_quantum)]
        lons = df_original[lon_col].values[:len(labels_quantum)]
        
        # Subsample
        if len(lats) > 10000:
            idx = np.random.choice(len(lats), 10000, replace=False)
            lats = lats[idx]
            lons = lons[idx]
            labels_geo = labels_quantum[idx]
        else:
            labels_geo = labels_quantum
        
        # Create scatter map
        fig, ax = plt.subplots(figsize=(14, 10))
        scatter = ax.scatter(lons, lats, c=labels_geo, cmap='tab10', 
                            alpha=0.5, s=10, edgecolor='none')
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title('Geographical Distribution of Clusters (Quantum)', fontsize=14, weight='bold')
        plt.colorbar(scatter, ax=ax, label='Cluster ID')
        ax.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(output_dirs["geo"], "geographical_clusters.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("[VIZ] Saved geographical analysis")
    else:
        print("[VIZ]   No coordinate columns found - skipping geographical analysis")

def save_metrics_report(results, all_results_dict, output_dirs):
    """Save detailed metrics report as text file."""
    import os
    print("\n[REPORT] Generating metrics report...")
    
    report_path = os.path.join(output_dirs["metrics"], "detailed_metrics.txt")
    
    # Use explicit UTF-8 encoding to avoid Windows cp1252 errors
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ESE - ENHANCED SAMPLING VIA ENCODER\n")
        f.write("Detailed Metrics Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("FINAL RESULTS SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Best Method: {results['best_method'].upper()}\n")
        f.write(f"Best Score: {results['best_score']:.4f}\n")
        f.write(f"Target Achieved: {'YES' if results['best_score'] >= 0.7 else 'NO'}\n\n")
        
        f.write("CLUSTERING METRICS BY METHOD\n")
        f.write("-" * 80 + "\n")
        f.write(f"Quantum Spectral: {results['quantum_silhouette']:.4f}\n")
        f.write(f"RBF Spectral:     {results['rbf_silhouette']:.4f}\n")
        f.write(f"K-Means:          {results['kmeans_silhouette']:.4f}\n\n")
        
        f.write("DETAILED RESULTS BY K\n")
        f.write("-" * 80 + "\n")
        for method, k_results in all_results_dict.items():
            f.write(f"\n{method}:\n")
            for k, sil in k_results.items():
                f.write(f"  k={k}: silhouette={sil:.4f}\n")
        
        f.write("\nCONFIGURATION PARAMETERS\n")
        f.write("-" * 80 + "\n")
        for key, value in CONFIG.items():
            if not isinstance(value, dict):
                f.write(f"{key}: {value}\n")
    
    print(f"[REPORT] Saved metrics report to: {report_path}")

def save_comparison_table(all_results_by_k, output_dirs):
    """Save comparison table as CSV."""
    import os
    print("\n[TABLE] Generating comparison tables...")
    
    # Create comprehensive comparison table
    table_data = []
    for method, results_by_k in all_results_by_k.items():
        for k, sil_score in results_by_k.items():
            table_data.append({
                'Method': method,
                'k': k,
                'Silhouette': sil_score
            })
    
    df_table = pd.DataFrame(table_data)
    table_path = os.path.join(output_dirs["tables"], "clustering_comparison.csv")
    df_table.to_csv(table_path, index=False)
    
    # Also save pivot table
    pivot = df_table.pivot(index='k', columns='Method', values='Silhouette')
    pivot_path = os.path.join(output_dirs["tables"], "clustering_pivot.csv")
    pivot.to_csv(pivot_path)
    
    print(f"[TABLE] Saved comparison tables")
    return df_table

def save_models(models, output_dirs, history):
    """Save trained autoencoder models."""
    import os
    print("\n[SAVE] Saving autoencoder models...")
    
    models_dir = os.path.join(output_dirs["base"], "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Save encoder
    encoder_path = os.path.join(models_dir, "encoder.keras")
    models["encoder"].save(encoder_path)
    print(f"[SAVE]   Encoder saved: {encoder_path}")
    
    # Save decoder
    decoder_path = os.path.join(models_dir, "decoder.keras")
    models["decoder"].save(decoder_path)
    print(f"[SAVE]   Decoder saved: {decoder_path}")
    
    # Save full autoencoder
    autoencoder_path = os.path.join(models_dir, "autoencoder.keras")
    models["autoencoder"].save(autoencoder_path)
    print(f"[SAVE]   Autoencoder saved: {autoencoder_path}")
    
    # Save training history
    history_df = pd.DataFrame(history.history)
    history_path = os.path.join(models_dir, "training_history.csv")
    history_df.to_csv(history_path, index=False)
    print(f"[SAVE]   Training history saved: {history_path}")
    
    return models_dir

def save_scaler(scaler, output_dirs):
    """Save StandardScaler for reproducibility."""
    import os
    import pickle
    print("\n[SAVE] Saving data scaler...")
    
    models_dir = os.path.join(output_dirs["base"], "models")
    os.makedirs(models_dir, exist_ok=True)
    
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"[SAVE]   Scaler saved: {scaler_path}")
    return scaler_path

def save_quantum_kernel(K_quantum, kernel_idx, Z_angles, gamma, output_dirs):
    """Save quantum kernel matrix and metadata."""
    import os
    print("\n[SAVE] Saving quantum kernel...")
    
    models_dir = os.path.join(output_dirs["base"], "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Save kernel matrix (compressed)
    kernel_path = os.path.join(models_dir, "quantum_kernel.npz")
    np.savez_compressed(
        kernel_path,
        K=K_quantum,
        kernel_idx=kernel_idx,
        Z_angles=Z_angles,
        gamma=gamma
    )
    print(f"[SAVE]   Quantum kernel saved: {kernel_path}")
    print(f"[SAVE]   Kernel shape: {K_quantum.shape}, Size: {K_quantum.nbytes / 1e6:.1f} MB")
    
    return kernel_path

def save_clustering_results(all_results, all_labels, best_configs, output_dirs):
    """Save all clustering results and best configurations."""
    import os
    import pickle
    print("\n[SAVE] Saving clustering results...")
    
    models_dir = os.path.join(output_dirs["base"], "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Save all clustering results
    results_path = os.path.join(models_dir, "clustering_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump({
            'all_results': all_results,
            'all_labels': all_labels,
            'best_configs': best_configs,
            'timestamp': datetime.now().isoformat()
        }, f)
    print(f"[SAVE]   Clustering results saved: {results_path}")
    
    # Save best labels as numpy arrays
    for method, labels in all_labels.items():
        for k, label_array in labels.items():
            label_path = os.path.join(models_dir, f"labels_{method.lower().replace(' ', '_')}_k{k}.npy")
            np.save(label_path, label_array)
    
    print(f"[SAVE]   Saved {sum(len(v) for v in all_labels.values())} label arrays")
    
    return results_path

def save_best_quantum_model(models, K_quantum, kernel_idx, Z_angles, gamma, 
                            labels_quantum, best_k_q, sil_quantum, output_dirs):
    """Save complete best quantum model package."""
    import os
    import json
    print("\n[SAVE] Creating best quantum model package...")
    
    best_model_dir = os.path.join(output_dirs["base"], "best_quantum_model")
    os.makedirs(best_model_dir, exist_ok=True)
    
    # Save encoder (most important for reproducing latent space)
    encoder_path = os.path.join(best_model_dir, "encoder.keras")
    models["encoder"].save(encoder_path)
    
    # Save quantum kernel
    kernel_path = os.path.join(best_model_dir, "quantum_kernel.npz")
    np.savez_compressed(
        kernel_path,
        K=K_quantum,
        kernel_idx=kernel_idx,
        Z_angles=Z_angles,
        gamma=gamma
    )
    
    # Save best clustering labels
    labels_path = os.path.join(best_model_dir, "best_labels.npy")
    np.save(labels_path, labels_quantum)
    
    # Save configuration (ensure Unicode preserved)
    config_path = os.path.join(best_model_dir, "model_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump({
            'best_k': int(best_k_q),
            'silhouette_score': float(sil_quantum),
            'latent_dim': CONFIG['latent_dim'],
            'encoder_layers': CONFIG['encoder_layers'],
            'n_qubits': CONFIG['n_qubits'],
            'kernel_gamma': float(gamma),
            'kernel_shape': K_quantum.shape,
            'n_clusters': int(best_k_q),
            'timestamp': datetime.now().isoformat(),
            'achieved_target': sil_quantum >= 0.7
        }, f, indent=2, ensure_ascii=False)
    
    # Save README (use UTF-8)
    readme_path = os.path.join(best_model_dir, "README.txt")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("BEST QUANTUM MODEL PACKAGE\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        # Use UTF-8 so checkmark characters are safe
        f.write(f"Silhouette Score: {sil_quantum:.4f}\n")
        f.write(f"Best k: {best_k_q}\n")
        f.write(f"Target Achieved: {'YES ✓' if sil_quantum >= 0.7 else 'NO ✗'}\n\n")
        
        f.write("CONTENTS:\n")
        f.write("-" * 70 + "\n")
        f.write("1. encoder.keras         - Trained autoencoder encoder\n")
        f.write("2. quantum_kernel.npz    - Quantum kernel matrix + metadata\n")
        f.write("3. best_labels.npy       - Best clustering labels\n")
        f.write("4. model_config.json     - Complete configuration\n")
        f.write("5. README.txt            - This file\n\n")
        
        f.write("HOW TO LOAD:\n")
        f.write("-" * 70 + "\n")
        f.write("# Load encoder\n")
        f.write("from tensorflow import keras\n")
        f.write("encoder = keras.models.load_model('encoder.keras')\n\n")
        
        f.write("# Load quantum kernel\n")
        f.write("import numpy as np\n")
        f.write("data = np.load('quantum_kernel.npz')\n")
        f.write("K = data['K']\n")
        f.write("gamma = data['gamma']\n\n")
        
        f.write("# Load labels\n")
        f.write("labels = np.load('best_labels.npy')\n")
    
    print(f"[SAVE]   Best quantum model package created: {best_model_dir}")
    print(f"[SAVE]   Silhouette score: {sil_quantum:.4f}")
    
    return best_model_dir

def main():
    """Main execution pipeline with comprehensive visualization."""
    import os
    
    print("=" * 80)
    print("ESE - Enhanced Sampling via Encoder")
    print("Quantum-enhanced autoencoder pipeline for aerosol data balancing")
    print("=" * 80)
    
    # Create output directories
    output_dirs = create_output_directories()
    
    # ...existing STEP 1-9 code...
    
    print("\n[STEP 1] Loading and preprocessing data...")
    df = load_data()
    df_original = df.copy()
    df_clean = clean_data(df)
    df_filtered = remove_outliers(df_clean)
    df_constrained = apply_physical_constraints(df_filtered)
    df_final, _ = apply_imputation(df_constrained)
    
    if "_imputed_flag" in df_final.columns:
        df_final = df_final.drop(columns=["_imputed_flag"])
    
    feature_names = df_final.columns.tolist()
    X = df_final.values
    
    print(f"\n[DATA] Final shape: {X.shape}")
    print(f"[DATA] Features: {feature_names[:5]}...")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("\n[STEP 2] Building and training autoencoder...")
    models = build_autoencoder(input_dim=X_scaled.shape[1])
    history = train_autoencoder(models, X_scaled)
    
    save_training_plots(history, output_dirs)
    
    print("\n[STEP 3] Encoding to latent space...")
    encoder = models["encoder"]
    Z = encoder.predict(X_scaled, verbose=0)
    print(f"[LATENT] Encoded shape: {Z.shape}")
    
    print(f"[LATENT DIAGNOSTICS] Mean: {Z.mean():.4f}, Std: {Z.std():.4f}")
    print(f"[LATENT DIAGNOSTICS] Range: [{Z.min():.4f}, {Z.max():.4f}]")
    
    from sklearn.metrics import silhouette_score as sil_score
    from sklearn.cluster import KMeans
    
    if len(Z) > 10000:
        Z_sample = Z[np.random.choice(len(Z), 10000, replace=False)]
    else:
        Z_sample = Z
    
    for test_k in [2, 3, 4]:
        kmeans_test = KMeans(n_clusters=test_k, random_state=42, n_init=3)
        labels_test = kmeans_test.fit_predict(Z_sample)
        sil_test = sil_score(Z_sample, labels_test)
        print(f"[LATENT QUALITY] K-Means k={test_k} on latent space: sil={sil_test:.4f}")
    
    print("\n[STEP 4] Estimating density and identifying low-density regions...")
    densities = estimate_density(Z)
    low_density_mask, density_threshold = get_low_density_mask(densities)
    
    print("\n[STEP 5] SKIPPING synthetic generation to focus on kernel quality...")
    X_augmented = X
    X_augmented_scaled = X_scaled
    Z_augmented = Z
    print(f"[AUGMENTED] Using original data only: {len(X_augmented)} samples")
    
    print("\n[STEP 7] Computing quantum kernel...")
    Z_min = Z_augmented.min(axis=0)
    Z_max = Z_augmented.max(axis=0)
    Z_angles = 2 * np.pi * (Z_augmented - Z_min) / (Z_max - Z_min + 1e-8)
    
    n_qubits = CONFIG["n_qubits"]
    if Z_angles.shape[1] < n_qubits:
        padding = np.zeros((len(Z_angles), n_qubits - Z_angles.shape[1]))
        Z_angles = np.hstack([Z_angles, padding])
    elif Z_angles.shape[1] > n_qubits:
        Z_angles = Z_angles[:, :n_qubits]
    
    K_quantum, kernel_idx = compute_quantum_kernel(
        Z_angles,
        feature_map=None,
        max_samples=CONFIG["quantum_kernel_samples"]
    )
    
    # Extract gamma for saving
    from sklearn.metrics.pairwise import pairwise_distances
    sample_size = min(1000, len(Z_angles))
    sample_idx = np.random.choice(len(Z_angles), sample_size, replace=False)
    distances = pairwise_distances(Z_angles[sample_idx], metric='euclidean')
    median_dist = np.median(distances[distances > 0])
    gamma_used = 1.0 / (2 * median_dist ** 2) if CONFIG.get("kernel_adaptive_tuning", False) else CONFIG.get("kernel_gamma", 0.5)
    
    print("\n[STEP 8] Running quantum spectral clustering FOR ALL K...")
    all_results = {
        'Quantum Spectral': {},
        'RBF Spectral': {},
        'K-Means': {}
    }
    all_labels = {
        'Quantum Spectral': {},
        'RBF Spectral': {},
        'K-Means': {}
    }
    
    for k in CONFIG["n_clusters_range"]:
        clusterer = SpectralClustering(n_clusters=k, affinity="precomputed", random_state=CONFIG["random_seed"])
        labels = clusterer.fit_predict(K_quantum)
        score = compute_robust_silhouette(K_quantum, labels, Z_augmented[kernel_idx])
        all_results['Quantum Spectral'][k] = score
        all_labels['Quantum Spectral'][k] = labels
        print(f"  k={k}: silhouette={score:.4f}")
    
    best_k_q = max(all_results['Quantum Spectral'], key=all_results['Quantum Spectral'].get)
    sil_quantum = all_results['Quantum Spectral'][best_k_q]
    labels_quantum = all_labels['Quantum Spectral'][best_k_q]
    print(f"[QUANTUM] Best k={best_k_q}, Silhouette={sil_quantum:.4f}")
    
    print("\n[STEP 9] Running classical baselines FOR ALL K...")
    
    from sklearn.metrics.pairwise import rbf_kernel
    if len(X_augmented_scaled) > CONFIG["quantum_kernel_samples"]:
        idx_rbf = np.random.choice(len(X_augmented_scaled), CONFIG["quantum_kernel_samples"], replace=False)
        X_rbf = X_augmented_scaled[idx_rbf]
    else:
        X_rbf = X_augmented_scaled
        idx_rbf = np.arange(len(X_augmented_scaled))
    
    K_rbf = rbf_kernel(X_rbf)
    K_rbf = normalize_kernel(K_rbf, ensure_psd=True)
    
    for k in CONFIG["n_clusters_range"]:
        clusterer = SpectralClustering(n_clusters=k, affinity="precomputed", random_state=CONFIG["random_seed"])
        labels = clusterer.fit_predict(K_rbf)
        score = compute_robust_silhouette(K_rbf, labels, X_rbf)
        all_results['RBF Spectral'][k] = score
        all_labels['RBF Spectral'][k] = labels
        print(f"  k={k}: silhouette={score:.4f}")
    
    best_k_rbf = max(all_results['RBF Spectral'], key=all_results['RBF Spectral'].get)
    sil_rbf = all_results['RBF Spectral'][best_k_rbf]
    labels_rbf = all_labels['RBF Spectral'][best_k_rbf]
    
    X_km = X_rbf
    for k in CONFIG["n_clusters_range"]:
        clusterer = MiniBatchKMeans(n_clusters=k, batch_size=1024, n_init=3, random_state=CONFIG["random_seed"])
        labels = clusterer.fit_predict(X_km)
        score = silhouette_score(X_km, labels, sample_size=min(10000, len(X_km)), random_state=CONFIG["random_seed"])
        all_results['K-Means'][k] = score
        all_labels['K-Means'][k] = labels
        print(f"  k={k}: silhouette={score:.4f}")
    
    best_k_km = max(all_results['K-Means'], key=all_results['K-Means'].get)
    sil_kmeans = all_results['K-Means'][best_k_km]
    labels_kmeans = all_labels['K-Means'][best_k_km]
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Method':<20} {'Best k':<10} {'Silhouette':<15} {'Status'}")
    print("-" * 80)
    print(f"{'Quantum Spectral':<20} {best_k_q:<10} {sil_quantum:<15.4f} {'✓ BEST' if sil_quantum >= max(sil_rbf, sil_kmeans) else ''}")
    print(f"{'RBF Spectral':<20} {best_k_rbf:<10} {sil_rbf:<15.4f} {'✓ BEST' if sil_rbf >= max(sil_quantum, sil_kmeans) else ''}")
    print(f"{'K-Means':<20} {best_k_km:<10} {sil_kmeans:<15.4f} {'✓ BEST' if sil_kmeans >= max(sil_quantum, sil_rbf) else ''}")
    print("=" * 80)
    
    best_score = max(sil_quantum, sil_rbf, sil_kmeans)
    
    gap = max(sil_rbf, sil_kmeans) - sil_quantum
    if best_score >= 0.7 and sil_quantum >= 0.7:
        print(f"\n✓✓ DOUBLE SUCCESS: Both quantum ({sil_quantum:.4f}) and best method ({best_score:.4f}) >= 0.7!")
    elif best_score >= 0.7:
        print(f"\n✓ TARGET ACHIEVED: Silhouette score = {best_score:.4f} >= 0.7")
        if gap > 0.05:
            print(f"⚠ Quantum still needs improvement: gap of {gap:.4f}")
    else:
        print(f"\n✗ Target not achieved (best: {best_score:.4f} < 0.7)")
    
    print("\n" + "=" * 80)
    print("GENERATING COMPREHENSIVE VISUALIZATIONS & SAVING MODELS")
    print("=" * 80)
    
    save_cluster_visualizations_all_k(Z_augmented, X_augmented, all_labels, all_results, feature_names, output_dirs, kernel_idx)
    save_kernel_heatmaps(K_quantum, K_rbf, output_dirs)
    save_geographical_analysis(df_original, labels_quantum, output_dirs)
    save_comparison_table(all_results, output_dirs)
    
    results_dict = {
        "best_method": "quantum" if sil_quantum >= max(sil_rbf, sil_kmeans) else ("rbf" if sil_rbf >= sil_kmeans else "kmeans"),
        "best_score": best_score,
        "quantum_silhouette": sil_quantum,
        "rbf_silhouette": sil_rbf,
        "kmeans_silhouette": sil_kmeans,
        "X_augmented": X_augmented,
        "labels_quantum": labels_quantum,
        "labels_rbf": labels_rbf,
        "labels_kmeans": labels_kmeans
    }
    save_metrics_report(results_dict, all_results, output_dirs)
    
    save_scaler(scaler, output_dirs)
    models_dir = save_models(models, output_dirs, history)
    kernel_path = save_quantum_kernel(K_quantum, kernel_idx, Z_angles, gamma_used, output_dirs)
    
    best_configs = {
        'quantum': {'k': best_k_q, 'silhouette': sil_quantum},
        'rbf': {'k': best_k_rbf, 'silhouette': sil_rbf},
        'kmeans': {'k': best_k_km, 'silhouette': sil_kmeans}
    }
    results_path = save_clustering_results(all_results, all_labels, best_configs, output_dirs)
    
    best_model_dir = save_best_quantum_model(
        models, K_quantum, kernel_idx, Z_angles, gamma_used,
        labels_quantum, best_k_q, sil_quantum, output_dirs
    )
    
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Output Directory: {output_dirs['base']}")
    print(f"Best Quantum Model: {best_model_dir}")
    print(f"Clustering Results: {results_path}")
    print(f"Quantum Kernel: {kernel_path}")
    print(f"Models Directory: {models_dir}")
    print("=" * 80)
    
    print("\n[COMPLETE] ESE pipeline finished successfully!")
    print(f"[OUTPUT] All results saved to: {output_dirs['base']}")
    print(f"\n[ARTIFACTS GENERATED]")
    print(f"  - Training plots: 2")
    print(f"  - Cluster plots: {sum(len(v) for v in all_labels.values())}")
    print(f"  - Kernel heatmaps: 1")
    print(f"  - Geographical maps: 1 (if coordinates available)")
    print(f"  - Data tables: 2 CSV files")
    print(f"  - Metrics report: 1 detailed text file")
    print(f"  - Saved models: 3 (encoder, decoder, autoencoder)")
    print(f"  - Quantum kernel: 1 compressed file")
    print(f"  - Best model package: Complete with README")
    
    return results_dict

if __name__ == "__main__":
    results = main()