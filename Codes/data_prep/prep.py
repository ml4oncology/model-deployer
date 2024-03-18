"""
Module to prepare data for model consumption
"""
from typing import Optional

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd

from data_prep.constants_postprocess import lab_cols, lab_change_cols, symp_cols, symp_change_cols  

class Imputer:
    """Impute missing data by mean, mode, or median
    """
    def __init__(self):
        self.impute_cols = {
            'mean': lab_cols.copy() + lab_change_cols.copy(), 
            'most_frequent': symp_cols.copy() + symp_change_cols.copy(),
            'median': []
        }
        self.imputer = {'mean': None, 'most_frequent': None, 'median': None}

    def impute(self, data: pd.DataFrame) -> pd.DataFrame:
        # loop through the mean, mode, and median imputer
        for strategy, imputer in self.imputer.items():
            cols = self.impute_cols[strategy]
            if len(cols)==0:
                continue
            
            data[cols] = data[cols].apply(pd.to_numeric)
            
            if imputer is None:
                # create the imputer and impute the data
                imputer = SimpleImputer(strategy=strategy) 
                data[cols] = imputer.fit_transform(data[cols])
                self.imputer[strategy] = imputer # save the imputer
            else:
                # use existing imputer to impute the data
                # print('Imputer Working!')
                data[cols] = imputer.transform(data[cols])
        return data
    

def fill_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing data that can be filled heuristically"""
    # fill the following missing data with 0
    col = 'num_prior_ED_visits_within_5_years'
    df[col] = df[col].fillna(0)

    # fill the following missing data with the maximum value
    for col in ['days_since_last_treatment', 'days_since_prev_ED_visit']:
        df[col] = df[col].fillna(df[col].max())

    return df


def encode_regimens(df, info_data_dir):

    GI_regimen_FeatureList_Full = pd.read_excel(info_data_dir + '/GI_regimen_feature_list.xlsx')
    GI_regimen_FeatureList = list(GI_regimen_FeatureList_Full['Regimen'])
    GI_regimen_rename_FeatureList = list(GI_regimen_FeatureList_Full['Regimen_Rename'])
    
    for iR in range(len(GI_regimen_FeatureList)):
        df[GI_regimen_FeatureList[iR]]=0
    df['regimen_other']=0
    
    for iR2 in range(len(df)):
        if df['regimen'][iR2] in GI_regimen_FeatureList:
            df[df['regimen'][iR2]][iR2] = 1
        else:
            df['regimen_other'][iR2] = 1    
            
    df = df.drop('regimen', axis=1)
    
    for iR in range(len(GI_regimen_FeatureList)):
        df = df.rename(columns={GI_regimen_FeatureList[iR]: GI_regimen_rename_FeatureList[iR]})
    
    return df

def encode_intent(df):

    intent_list = ['PALLIATIVE', 'NEOADJUVANT', 'ADJUVANT', 'CURATIVE']
    
    for iR in range(len(intent_list)):
        df[intent_list[iR]]=0
    
    for iR2 in range(len(df)):
        if df['intent'][iR2] in intent_list:
            df[df['intent'][iR2]][iR2] = 1
            
    df = df.drop('intent', axis=1)
    
    for iR in range(len(intent_list)):
        df = df.rename(columns={intent_list[iR]: 'intent_'+intent_list[iR]})
    
    return df
            

class PrepData:
    """Prepare the data for model training"""
    def __init__(self):
        self.imp = Imputer() # imputer
        self.scaler = None # normalizer
        self.clip_thresh = None # outlier clippers

        self.norm_cols = [
            'height',
            'weight',
            'body_surface_area',
            'cycle_number',
            'age',
            'visit_month_sin',
            'visit_month_cos',
            'line_of_therapy',
            'days_since_starting_treatment',
            'days_since_last_treatment',
            'num_prior_EDs_within_5_years',
            'days_since_prev_ED',
        ] + symp_cols + lab_cols + lab_change_cols + symp_change_cols #drug_cols +
        self.clip_cols = [
            'height',
            'weight',
            'body_surface_area',
        ] + lab_cols + lab_change_cols
    
    def transform_data(
        self, 
        data,
        clip: bool = True, 
        impute: bool = True, 
        normalize: bool = True, 
        ohe_kwargs: Optional[dict] = None,
        data_name: Optional[str] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """Transform (one-hot encode, clip, impute, normalize) the data.
        
        Args:
            ohe_kwargs (dict): a mapping of keyword arguments fed into 
                OneHotEncoder.encode
                
        IMPORTANT: always make sure train data is done first before valid
        or test data
        """
        if ohe_kwargs is None: ohe_kwargs = {}
        if data_name is None: data_name = 'the'
        
        if clip:
            # Clip the outliers based on the train data quantiles
            data = self.clip_outliers(data)

        if impute:
            # Impute missing data based on the train data mode/median/mean
            allNaN_col = data.columns[data.isna().all()].tolist()
            for iC in range(len(allNaN_col)):
                data[allNaN_col[iC]][0] = 0
            data = self.imp.impute(data)
            
        if normalize:
            # Scale the data based on the train data distribution
            data = self.normalize_data(data)
            
        return data
    
    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # use only the columns that exist in the data
        norm_cols = [col for col in self.norm_cols if col in data.columns]
        
        if self.scaler is None:
            self.scaler = StandardScaler()
            data[norm_cols] = self.scaler.fit_transform(data[norm_cols])
        else:
            data[norm_cols] = self.scaler.transform(data[norm_cols])
        return data
    
    def clip_outliers(
        self, 
        data: pd.DataFrame, 
        lower_percentile: float = 0.001, 
        upper_percentile: float = 0.999
    ) -> pd.DataFrame:
        """Clip the upper and lower percentiles for the columns indicated below
        """
        # use only the columns that exist in the data
        cols = [col for col in self.clip_cols if col in data.columns]
        
        data[cols] = data[cols].apply(pd.to_numeric)
        
        if self.clip_thresh is None:
            percentiles = [lower_percentile, upper_percentile]
            self.clip_thresh = data[cols].quantile(percentiles)
            
        data[cols] = data[cols].clip(
            lower=self.clip_thresh.loc[lower_percentile], 
            upper=self.clip_thresh.loc[upper_percentile], 
            axis=1
        )
        return data