# CHOP Analysis Python

## Instructions

### Part 1: Assemble the project cohort

The project goal is to identify patients seen for drug overdose, determine if they had an active opioid at the start of the encounter, and if they had any readmissions for drug overdose.

Your task is to assemble the study cohort by identifying encounters that meet the following criteria:

1. The patient’s visit is an encounter for drug overdose
2. The hospital encounter occurs after July 15, 1999
3. The patient’s age at time of encounter is between 18 and 35 (Patient is considered to be 35 until turning 36)

### Part 2: Create additional fields

With your drug overdose encounter, create the following indicators:

1. `DEATH_AT_VISIT_IND`: `1` if patient died during the drug overdose encounter, `0` if the patient died at a different time
2. `COUNT_CURRENT_MEDS`: Count of active medications at the start of the drug overdose encounter
3. `CURRENT_OPIOID_IND`: `1` if the patient had at least one active medication at the start of the overdose encounter that is on the Opioids List (provided below), 0 if not 
4. `READMISSION_90_DAY_IND`: `1` if the visit resulted in a subsequent drug overdose readmission within 90 days, 0 if not 
5. `READMISSION_30_DAY_IND`: `1` if the visit resulted in a subsequent drug overdose readmission within 30 days, 0 if not overdose encounter, `0` if not
6. `FIRST_READMISSION_DATE`: The date of the index visit's first readmission for drug overdose. Field should be left as `N/A` if no readmission for drug overdose within 90 days

### Part 3: Export the data to a `CSV` file

Export a dataset containing these required fields:

| Field name                | Field Description                                                                                                                  | Data Type        |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| `PATIENT_ID`              | Patient identifier                                                                                                                 | Character String |
| `ENCOUNTER_ID`            | Visit identifier                                                                                                                   | Character string |
| `HOSPITAL_ENCOUNTER_DATE` | Beginning of hospital encounter date                                                                                               | Date/time        |
| `AGE_AT_VISIT`            | Patient age at admission                                                                                                           | Num              |
| `DEATH_AT_VISIT_IND`      | Indicator if the patient died during the drug overdose encounter. Leave `N/A` if patient has not died,                             | 0 /1             |
| `COUNT_CURRENT_MEDS`      | Count of active medications at the start of the drug overdose encounter                          | Num              |
| `CURRENT_OPIOID_IND`      | if the patient had at least one active medication at the start of the overdose encounter that is on the Opioids List (provided below)     | 0/1              |
| `READMISSION_90_DAY_IND`  | Indicator if the visit resulted in a subsequent readmission within 90 days     | 0/1              |
| `READMISSION_30_DAY_IND`  | Indicator if the visit resulted in a subsequent readmission within 30 days     | 0/1              |
| `FIRST_READMISSION_DATE`  | Date of the first readmission for drug overdose within 90 days. Leave `N/A` if no readmissions for drug overdose within 90 days. | Date/time        |

## Opioids List:

- Hydromorphone 325Mg
- Fentanyl – 100 MCG
- Oxycodone-acetaminophen 100 Ml
-
