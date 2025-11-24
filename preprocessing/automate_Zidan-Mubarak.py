"""
Climate Change Impact on Agriculture - Data Preprocessing Automation
Author: Zidan Mubarak
Description: Automated preprocessing pipeline for climate change agriculture dataset
Target: Advanced (4 pts) - Kriteria 1
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class ClimateAgriculturePreprocessor:
    """
    Automated preprocessing pipeline for Climate Change Agriculture dataset
    
    Features:
    - Data loading and validation
    - Missing value handling
    - Outlier detection and treatment
    - Feature engineering
    - Encoding categorical variables
    - Feature scaling
    - Data splitting
    """
    
    def __init__(self, filepath):
        """Initialize preprocessor with data file path"""
        self.filepath = filepath
        self.df = None
        self.df_processed = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load dataset from CSV file"""
        print("=" * 70)
        print("STEP 1: LOADING DATA")
        print("=" * 70)
        
        self.df = pd.read_csv(self.filepath)
        print(f"✓ Data loaded successfully")
        print(f"  Shape: {self.df.shape}")
        print(f"  Columns: {list(self.df.columns)}")
        print()
        
        return self
    
    def explore_data(self):
        """Explore dataset characteristics"""
        print("=" * 70)
        print("STEP 2: DATA EXPLORATION")
        print("=" * 70)
        
        print("\n📊 Dataset Info:")
        print(f"  Total rows: {len(self.df)}")
        print(f"  Total columns: {len(self.df.columns)}")
        
        print("\n📋 Column Data Types:")
        for col, dtype in self.df.dtypes.items():
            print(f"  {col}: {dtype}")
        
        print("\n🔍 Missing Values:")
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("  ✓ No missing values found")
        else:
            print(missing[missing > 0])
        
        print("\n📈 Numerical Columns Statistics:")
        print(self.df.describe())
        
        print("\n📊 Categorical Columns:")
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            print(f"  {col}: {self.df[col].nunique()} unique values")
        
        print()
        return self
    
    def handle_missing_values(self):
        """Handle missing values in dataset"""
        print("=" * 70)
        print("STEP 3: HANDLING MISSING VALUES")
        print("=" * 70)
        
        missing_before = self.df.isnull().sum().sum()
        
        if missing_before == 0:
            print("✓ No missing values to handle")
        else:
            # Fill numerical columns with median
            numerical_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if self.df[col].isnull().sum() > 0:
                    median_val = self.df[col].median()
                    self.df[col].fillna(median_val, inplace=True)
                    print(f"  ✓ Filled {col} with median: {median_val:.2f}")
            
            # Fill categorical columns with mode
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if self.df[col].isnull().sum() > 0:
                    mode_val = self.df[col].mode()[0]
                    self.df[col].fillna(mode_val, inplace=True)
                    print(f"  ✓ Filled {col} with mode: {mode_val}")
        
        missing_after = self.df.isnull().sum().sum()
        print(f"\n✓ Missing values: {missing_before} → {missing_after}")
        print()
        
        return self
    
    def handle_outliers(self):
        """Detect and handle outliers using IQR method"""
        print("=" * 70)
        print("STEP 4: HANDLING OUTLIERS")
        print("=" * 70)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        outliers_removed = 0
        
        for col in numerical_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            
            if outliers > 0:
                # Cap outliers instead of removing
                self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
                outliers_removed += outliers
                print(f"  ✓ {col}: {outliers} outliers capped")
        
        print(f"\n✓ Total outliers handled: {outliers_removed}")
        print()
        
        return self
    
    def feature_engineering(self):
        """Create new features from existing ones"""
        print("=" * 70)
        print("STEP 5: FEATURE ENGINEERING")
        print("=" * 70)
        
        # 1. Temperature categories
        self.df['Temperature_Category'] = pd.cut(
            self.df['Average_Temperature_C'],
            bins=[-np.inf, 10, 20, 30, np.inf],
            labels=['Cold', 'Moderate', 'Warm', 'Hot']
        )
        print("  ✓ Created Temperature_Category")
        
        # 2. Precipitation categories
        self.df['Precipitation_Category'] = pd.cut(
            self.df['Total_Precipitation_mm'],
            bins=[-np.inf, 500, 1500, 2500, np.inf],
            labels=['Low', 'Medium', 'High', 'Very_High']
        )
        print("  ✓ Created Precipitation_Category")
        
        # 3. Yield categories (for classification)
        self.df['Yield_Category'] = pd.cut(
            self.df['Crop_Yield_MT_per_HA'],
            bins=[-np.inf, 1.5, 2.5, 3.5, np.inf],
            labels=['Low', 'Medium', 'High', 'Very_High']
        )
        print("  ✓ Created Yield_Category")
        
        # 4. Climate stress indicator
        self.df['Climate_Stress'] = (
            (self.df['Average_Temperature_C'] > 30).astype(int) +
            (self.df['Total_Precipitation_mm'] < 500).astype(int) +
            (self.df['Extreme_Weather_Events'] > 5).astype(int)
        )
        print("  ✓ Created Climate_Stress indicator")
        
        # 5. Resource efficiency
        self.df['Resource_Efficiency'] = (
            self.df['Crop_Yield_MT_per_HA'] / 
            (self.df['Fertilizer_Use_KG_per_HA'] + self.df['Pesticide_Use_KG_per_HA'] + 1)
        )
        print("  ✓ Created Resource_Efficiency")
        
        # 6. Environmental impact score
        self.df['Environmental_Impact'] = (
            self.df['CO2_Emissions_MT'] * 0.4 +
            self.df['Pesticide_Use_KG_per_HA'] * 0.3 +
            self.df['Fertilizer_Use_KG_per_HA'] * 0.3
        )
        print("  ✓ Created Environmental_Impact")
        
        print(f"\n✓ Total features after engineering: {len(self.df.columns)}")
        print()
        
        return self
    
    def encode_categorical(self):
        """Encode categorical variables"""
        print("=" * 70)
        print("STEP 6: ENCODING CATEGORICAL VARIABLES")
        print("=" * 70)
        
        categorical_cols = ['Country', 'Region', 'Crop_Type', 'Adaptation_Strategies',
                           'Temperature_Category', 'Precipitation_Category', 'Yield_Category']
        
        for col in categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[f'{col}_Encoded'] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
                print(f"  ✓ Encoded {col} ({self.df[col].nunique()} categories)")
        
        print(f"\n✓ Total categorical columns encoded: {len(categorical_cols)}")
        print()
        
        return self
    
    def select_features(self):
        """Select final features for modeling"""
        print("=" * 70)
        print("STEP 7: FEATURE SELECTION")
        print("=" * 70)
        
        # Select numerical features
        numerical_features = [
            'Year',
            'Average_Temperature_C',
            'Total_Precipitation_mm',
            'CO2_Emissions_MT',
            'Extreme_Weather_Events',
            'Irrigation_Access_%',
            'Pesticide_Use_KG_per_HA',
            'Fertilizer_Use_KG_per_HA',
            'Soil_Health_Index',
            'Climate_Stress',
            'Resource_Efficiency',
            'Environmental_Impact'
        ]
        
        # Select encoded categorical features
        encoded_features = [
            'Country_Encoded',
            'Region_Encoded',
            'Crop_Type_Encoded',
            'Adaptation_Strategies_Encoded',
            'Temperature_Category_Encoded',
            'Precipitation_Category_Encoded'
        ]
        
        # Target variable
        target = 'Crop_Yield_MT_per_HA'
        
        # Combine all features
        selected_features = numerical_features + encoded_features + [target]
        
        # Create processed dataframe
        self.df_processed = self.df[selected_features].copy()
        
        print(f"  ✓ Selected {len(numerical_features)} numerical features")
        print(f"  ✓ Selected {len(encoded_features)} encoded features")
        print(f"  ✓ Target variable: {target}")
        print(f"\n✓ Total features in processed dataset: {len(selected_features)}")
        print()
        
        return self
    
    def scale_features(self):
        """Scale numerical features"""
        print("=" * 70)
        print("STEP 8: FEATURE SCALING")
        print("=" * 70)
        
        # Features to scale (exclude target)
        features_to_scale = self.df_processed.columns.drop('Crop_Yield_MT_per_HA')
        
        # Fit and transform
        self.df_processed[features_to_scale] = self.scaler.fit_transform(
            self.df_processed[features_to_scale]
        )
        
        print(f"  ✓ Scaled {len(features_to_scale)} features using StandardScaler")
        print(f"  ✓ Mean: ~0, Std: ~1")
        print()
        
        return self
    
    def save_processed_data(self, output_path):
        """Save processed dataset"""
        print("=" * 70)
        print("STEP 9: SAVING PROCESSED DATA")
        print("=" * 70)
        
        self.df_processed.to_csv(output_path, index=False)
        
        print(f"  ✓ Processed data saved to: {output_path}")
        print(f"  ✓ Shape: {self.df_processed.shape}")
        print(f"  ✓ File size: {self.df_processed.memory_usage(deep=True).sum() / 1024:.2f} KB")
        print()
        
        return self
    
    def get_summary(self):
        """Print preprocessing summary"""
        print("=" * 70)
        print("PREPROCESSING SUMMARY")
        print("=" * 70)
        
        print(f"\n📊 Original Dataset:")
        print(f"  Rows: {len(self.df)}")
        print(f"  Columns: {len(self.df.columns)}")
        
        print(f"\n📊 Processed Dataset:")
        print(f"  Rows: {len(self.df_processed)}")
        print(f"  Columns: {len(self.df_processed.columns)}")
        
        print(f"\n✅ Preprocessing Steps Completed:")
        print(f"  1. ✓ Data Loading")
        print(f"  2. ✓ Data Exploration")
        print(f"  3. ✓ Missing Value Handling")
        print(f"  4. ✓ Outlier Treatment")
        print(f"  5. ✓ Feature Engineering (6 new features)")
        print(f"  6. ✓ Categorical Encoding (7 columns)")
        print(f"  7. ✓ Feature Selection")
        print(f"  8. ✓ Feature Scaling")
        print(f"  9. ✓ Data Saving")
        
        print(f"\n🎯 Ready for Model Training!")
        print("=" * 70)
        
        return self


def main():
    """Main preprocessing pipeline"""
    print("\n" + "=" * 70)
    print("CLIMATE CHANGE AGRICULTURE - DATA PREPROCESSING")
    print("=" * 70)
    print("Author: Zidan Mubarak")
    print("Target: Advanced (4 pts) - Kriteria 1")
    print("=" * 70 + "\n")
    
    # File paths
    input_file = "../climate_change_raw.csv"
    output_file = "climate_change_preprocessing.csv"
    
    # Initialize preprocessor
    preprocessor = ClimateAgriculturePreprocessor(input_file)
    
    # Run preprocessing pipeline
    preprocessor.load_data() \
                .explore_data() \
                .handle_missing_values() \
                .handle_outliers() \
                .feature_engineering() \
                .encode_categorical() \
                .select_features() \
                .scale_features() \
                .save_processed_data(output_file) \
                .get_summary()
    
    print("\n✅ Preprocessing completed successfully!")
    print(f"📁 Output file: {output_file}\n")


if __name__ == "__main__":
    main()
