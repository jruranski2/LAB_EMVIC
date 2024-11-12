import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict

def preprocess_eye_tracking_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Preprocess eye tracking data for machine learning with optimized performance.
    
    Args:
        df (pd.DataFrame): Raw eye tracking data
    
    Returns:
        Tuple[pd.DataFrame, np.ndarray, np.ndarray]: (features, encoded_labels, class_names)
    """
    # Constants
    ROWS_PER_USER = 4
    COORDINATE_TYPES = ['x', 'y', 'z', 'w']
    
    # Extract user IDs and features more efficiently
    user_ids = df.iloc[::ROWS_PER_USER, 0].values  # Take every 4th row
    features = df.iloc[:, 1:].astype(np.float32)  # More memory efficient dtype
    
    # Reshape data more efficiently using numpy operations
    num_users = len(df) // ROWS_PER_USER
    num_timepoints = features.shape[1]
    
    # Reshape all at once instead of loop
    X = features.values.reshape(num_users, ROWS_PER_USER * num_timepoints)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(user_ids)
    
    # Create feature names vectorized
    timepoints = np.arange(num_timepoints)
    feature_names = [f'{coord}_t{t}' for t in timepoints for coord in COORDINATE_TYPES]
    
    # Convert to DataFrame
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Add derived features more efficiently
    def add_derived_features(df: pd.DataFrame, coord: str) -> pd.DataFrame:
        # Get base columns for this coordinate
        coord_cols = df.filter(regex=f"^{coord}_t").columns
        coord_data = df[coord_cols]
        
        # Calculate velocity using vectorized operations
        velocity = coord_data.diff(axis=1)
        velocity_cols = [f'{coord}_velocity_t{i}' for i in range(len(coord_cols)-1)]
        df[velocity_cols] = velocity.iloc[:, 1:]
        
        # Calculate acceleration using vectorized operations
        acceleration = velocity.diff(axis=1)
        accel_cols = [f'{coord}_acceleration_t{i}' for i in range(len(coord_cols)-2)]
        df[accel_cols] = acceleration.iloc[:, 2:]
        
        # Calculate statistical features efficiently
        df[f'{coord}_mean'] = coord_data.mean(axis=1)
        df[f'{coord}_std'] = coord_data.std(axis=1)
        df[f'{coord}_max'] = coord_data.max(axis=1)
        df[f'{coord}_min'] = coord_data.min(axis=1)
        df[f'{coord}_range'] = df[f'{coord}_max'] - df[f'{coord}_min']
        
        return df
    
    # Apply derived features for each coordinate type
    for coord in COORDINATE_TYPES:
        X_df = add_derived_features(X_df, coord)
    
    return X_df, y_encoded, le.classes_

def analyze_features(X_df: pd.DataFrame, y: np.ndarray, 
                    class_names: np.ndarray) -> Dict[str, list]:
    """
    Analyze the preprocessed features efficiently.
    
    Args:
        X_df (pd.DataFrame): Preprocessed features
        y (np.ndarray): Encoded labels
        class_names (np.ndarray): Original class names
    
    Returns:
        Dict[str, list]: Feature categories and their columns
    """
    feature_types = {
        'Position': X_df.filter(regex=r'^[xyzw]_t\d+$').columns.tolist(),
        'Velocity': X_df.filter(regex='velocity').columns.tolist(),
        'Acceleration': X_df.filter(regex='acceleration').columns.tolist(),
        'Statistical': X_df.filter(regex='mean|std|max|min|range').columns.tolist()
    }
    
    # Print summary statistics
    print(f"""Dataset Summary:
Number of samples: {len(X_df):,}
Number of features: {len(X_df.columns):,}
Number of classes: {len(np.unique(y)):,}
""")
    
    print("\nFeature categories:")
    for category, features in feature_types.items():
        print(f"{category} features: {len(features):,}")
        if features:
            print(f"Example: {features[0]}")
    
    return feature_types

# Example usage:
if __name__ == "__main__":
    # Load data
    df = pd.read_csv("your_data.csv")
    
    # Process data
    X_df, y_encoded, class_names = preprocess_eye_tracking_data(df)
    
    # Analyze features
    feature_types = analyze_features(X_df, y_encoded, class_names)