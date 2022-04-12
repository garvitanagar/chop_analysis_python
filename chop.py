#!/usr/bin/env python
# coding: utf-8

# In[175]:


import pandas as pd
import numpy as np


# ### Importing files

# In[176]:


patients = pd.read_csv('patients.csv')
encounters = pd.read_csv('encounters.csv')
procedures = pd.read_csv('procedures.csv')
medications = pd.read_csv('medications.csv')
allergies = pd.read_csv('allergies.csv')


# ### Merging Patient and Encounters data
# #### ape is all_patient_encounters

# In[177]:


get_ipython().run_cell_magic('time', '', "ape = pd.merge(patients,encounters,left_on='Id',right_on='PATIENT',how='left')")


# In[178]:


## Renaming columns
column_name = {'Id_x':'PATIENT_ID','Id_y':'ENCOUNTER_ID'}
ape.rename(columns=column_name,inplace=True)


# ### Different Encounters

# In[179]:


#dict(ape['REASONDESCRIPTION'].value_counts())


# ### Converting Birth, Death, Start and Stop date object to datetime type

# In[180]:


ape['BIRTHDATE'] = pd.to_datetime(ape['BIRTHDATE'], format='%Y-%m-%d')
ape['DEATHDATE'] = pd.to_datetime(ape['DEATHDATE'], format='%Y-%m-%d').dt.to_period('D')
ape['START'] =  pd.to_datetime(ape['START'], format='%Y-%m-%d %H:%M:%S.%f')
ape['STOP'] =  pd.to_datetime(ape['STOP'], format='%Y-%m-%d').dt.to_period('D')


# ### Calculating AGE

# In[181]:


#sorted(ape[ape['START']>pd.Timestamp(1999,7,15)]['START'].dt.to_period('M').unique())
ape['AGE_AT_ENCOUNTER'] = (np.floor((pd.to_datetime(ape['START']) - 
             pd.to_datetime(ape['BIRTHDATE'])).dt.days / 365.25)).astype(int)


# ### Filter
# #### Drug Overdose, encounters after July 15,1999 and patients age between 18 and 35
# ###### ape_do - contain all_patient_encouters_drug_overdose

# In[182]:


ape_do = ape[(ape['REASONDESCRIPTION']=='Drug overdose') & (ape['START']>pd.Timestamp(1999,7,15)) & (ape['AGE_AT_ENCOUNTER'].between(18,35))]
ape_do = ape_do[['PATIENT_ID','ENCOUNTER_ID','BIRTHDATE','DEATHDATE','START','STOP','AGE_AT_ENCOUNTER']]


# ### Part 2
# #### Creating DEATH_AT_VISIT_IND variable

# In[183]:


start = ape_do['START'].dt.to_period('D')
stop = ape_do['STOP']
death = ape_do['DEATHDATE']
ape_do['DEATH_AT_VISIT_IND'] = np.where(death.isnull(),'N/A',np.where(death.between(start,stop),1,0))


# ### Medication 
# #### Including Today's date as stop date if N/A (still on medication), ape_do_med - now includes medication data in ape_do
# 

# In[184]:


medications = medications[['START', 'STOP', 'PATIENT', 'ENCOUNTER', 'CODE', 'DESCRIPTION']]
medications['START'] =  pd.to_datetime(medications['START'], format='%Y-%m-%d')
medications['STOP'] =  pd.to_datetime(medications['STOP'], format='%Y-%m-%d')
ape_do_med = pd.merge(ape_do,medications,left_on='PATIENT_ID',right_on='PATIENT',how='left')
ape_do_med['STOP_y'] = pd.to_datetime(ape_do_med['STOP_y'].fillna(pd.Timestamp("today")), format='%Y-%m-%d')


# #### Filtering medications active and not active during the start encounter date 

# In[185]:


start_encounter = ape_do_med['START_x'].dt.to_period('D')
medication_start = ape_do_med['START_y'].dt.to_period('D')
medication_stop = ape_do_med['STOP_y'].dt.to_period('D')
ape_do_med['MED_STATUS_BEFORE_ENC'] = np.where(start_encounter.between(medication_start,medication_stop),1,0)


# In[186]:


## Renaming columns
column_name = {'START_x':'ENCOUNTER_START','START_y':'MEDICATION_START','STOP_x':'ENCOUNTER_STOP','STOP_y':'MEDICATION_STOP'}
ape_do_med.rename(columns=column_name,inplace=True)


# #### Creating CURRENT_OPIOID_IND if MED_STATUS_BEFORE_ENC(medication status before encounter) is True

# In[187]:


hydro = ape_do_med['DESCRIPTION'].str.contains('Hydromorphone 325 MG',na=False)
fen = ape_do_med['DESCRIPTION'].str.contains('Fentanyl 100 MCG',na=False)
oxy = ape_do_med['DESCRIPTION'].str.contains('Oxycodone-acetaminophen 100ML',na=False)
ape_do_med['CURRENT_OPIOID_IND'] = np.where((ape_do_med.MED_STATUS_BEFORE_ENC==1)&(hydro | fen | oxy),1,0)


# #### Distributing Active and Not active patients by MED_STATUS_BEFORE_ENC

# In[188]:


ape_do_med_active = ape_do_med[ape_do_med['MED_STATUS_BEFORE_ENC']==1]
ape_do_med_not_active = ape_do_med[ape_do_med['MED_STATUS_BEFORE_ENC']==0]


# #### Grouping both active and not active patients to COUNT_CURRENT_MEDS

# In[189]:


ape_do_med_active = ape_do_med_active.groupby(['PATIENT_ID', 'ENCOUNTER_ID','ENCOUNTER_START','AGE_AT_ENCOUNTER','DEATH_AT_VISIT_IND'], as_index=False).agg({'MED_STATUS_BEFORE_ENC':'sum','CURRENT_OPIOID_IND':'sum'})
ape_do_med_not_active = ape_do_med_not_active.groupby(['PATIENT_ID', 'ENCOUNTER_ID','ENCOUNTER_START','AGE_AT_ENCOUNTER','DEATH_AT_VISIT_IND'], as_index=False).agg({'MED_STATUS_BEFORE_ENC':'sum','CURRENT_OPIOID_IND':'sum'})
ape_do_med_active.rename(columns={'MED_STATUS_BEFORE_ENC':'COUNT_CURRENT_MEDS'},inplace=True)
ape_do_med_not_active.rename(columns={'MED_STATUS_BEFORE_ENC':'COUNT_CURRENT_MEDS'},inplace=True)


# #### Removing already existing patient_id's in active df from not active df

# In[190]:


ap = ape_do_med_not_active[~ape_do_med_not_active['ENCOUNTER_ID'].isin(ape_do_med_active.ENCOUNTER_ID)]


# #### Stacking both active and not active dataframe
# #### ape_do_both - contains all_patients, encounters, drug_overdose amd medication data

# In[191]:


ape_do_both = pd.concat([ape_do_med_active,ap],ignore_index=True).sort_values(by=['PATIENT_ID','ENCOUNTER_START'])


# ### Creating READMISSION_90_DAY_IND, READMISSION_30_DAY_IND and FIRST_READMISSION_DATE

# In[192]:


previous = ape_do_both['ENCOUNTER_START'].shift(-1)
current = ape_do_both['ENCOUNTER_START']

ape_do_both['READMISSION_90_DAY_IND'] = np.where(
    
    ((previous - current).dt.days <= 90) & (ape_do_both['PATIENT_ID'].shift(-1) == ape_do_both['PATIENT_ID']), 1, 0
)


# In[193]:


ape_do_both['READMISSION_30_DAY_IND'] = np.where(
    ((previous - current).dt.days <= 30) &
    (ape_do_both['PATIENT_ID'].shift(-1) == ape_do_both['PATIENT_ID']), 1, 0
)


# In[194]:


ape_do_both['FIRST_READMISSION_DATE'] = np.where(
    (ape_do_both['READMISSION_90_DAY_IND'] == 1), (previous).dt.strftime('%Y-%m-%d %H:%M:%S'),'N/A'
)


# In[201]:


ape_do_both.rename(columns={'ENCOUNTER_START':'HOSPITAL_ENCOUNTER_DATE','AGE_AT_ENCOUNTER':'AGE_AT_VISIT'},inplace=True)


# ### Exporting into CSV

# In[203]:


ape_do_both.to_csv(r'/Users/hkokat963/Downloads/analyst-take-home-task-master/Solution/garvita_nagar.csv', index=False)

