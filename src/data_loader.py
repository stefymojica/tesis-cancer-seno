import pandas as pd
import os

def load_cbis_ddsm_data(data_dir='data'):
    """
    Attempts to load CBIS-DDSM metadata files from the data directory.
    """
    # Ensure we look in the right place relative to the caller
    if not os.path.exists(data_dir):
        return None
        
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and ('mass' in f.lower() or 'calc' in f.lower())]
    if not files:
        return None
    
    dataframes = {}
    for f in files:
        path = os.path.join(data_dir, f)
        df = pd.read_csv(path)
        
        # Estandarizar nombres de columnas comunes
        # Cambiar 'breast density' a 'breast_density' si existe
        if 'breast density' in df.columns:
            df = df.rename(columns={'breast density': 'breast_density'})
            
        dataframes[f] = df
    
    return dataframes

def get_summary_stats(df):
    """
    Returns specific summary stats relevant for CBIS-DDSM.
    """
    stats = {
        'total_records': len(df),
        'unique_patients': df['patient_id'].nunique() if 'patient_id' in df.columns else 0,
        'malignant_count': len(df[df['pathology'] == 'MALIGNANT']) if 'pathology' in df.columns else 0,
        'benign_count': len(df[df['pathology'].str.contains('BENIGN', na=False)]) if 'pathology' in df.columns else 0,
        'avg_breast_density': df['breast_density'].mean() if 'breast_density' in df.columns else None
    }
    return stats

def get_patient_summary(df):
    """
    Groups the dataframe by patient_id and returns counts of images/abnormalities.
    Useful for ensuring patient-level splitting during training.
    """
    if 'patient_id' not in df.columns:
        return None
    
    patient_grp = df.groupby('patient_id').agg({
        'pathology': 'first',
        'image file path': 'count',
        'breast_density': 'mean'
    }).rename(columns={'image file path': 'image_count'})
    
    return patient_grp
