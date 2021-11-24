import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = "/home/denis/Desktop/" \
                 "CSYS 395B - Machine Learning/Project/" \
                 "Data/2019"
PARAMETER_DATA_DIRECTORY = os.path.join(ROOT_DIR, "Parameter Search Data")
PERFORMANCE_DATA_DIRECTORY = os.path.join(ROOT_DIR, "Performance Data")
CLASSIFIER_DIRECTORY = os.path.join(
    "/home/denis/Data/PyCharm Extended Storage/SAMSHA Classifiers"
)
FINAL_DATA = os.path.join(ROOT_DIR, "_final_code", "Figures and Data")


"""
    Feature dictionaries below constructed based on TEDS-D 2019 Codebook

    'Other' and 'Unknown' were treated as equivalent categories in a few places.

    All fields use integers for all data points. Most use sequential
    sequences starting at 1, but a few start at 0, and some have gaps. -9 is
    used to track missing data, although 'other', when present,
    was identified with a sequential positive integer.

    Note that it is possible to duplicate features in these feature
    dictionaries without causing problems. Rather than modifying existing
    dictionaries, please add new, custom dictionaries to meet new needs.
    Let's consider the current 'DEFAULT' dictionaries as starting points for
    discussion that we won't change - unless we find an error.
"""

# DEFAULT DICTIONARY - DO NOT EDIT
# Outputs to Predict - From 2019 Codebook
outputs = {
    'REASON': {
        'Description': 'Reason For Discharge',
        'Number of Categories': 6,
        'Ordinal': False,
        'Unknown': 69626 / 1722503,
        'Values to Exclude': None
    }
}

# DEFAULT DICTIONARY - DO NOT EDIT
# Patient Demographics and History
demo_and_history = {
    'AGE': {
        'Description': 'Age at Admission',
        'Number of Categories': 12,
        'Ordinal': True,
        'Unknown': 0 / 1722503,
        'Values to Exclude': None
    },
    'GENDER': {
        'Description': 'Biological Gender',
        'Number of Categories': 2,
        'Ordinal': False,
        'Unknown': 756 / 1722503,
        'Values to Exclude': None
    },
    'RACE': {
        'Description': 'Race',
        'Number of Categories': 10,
        'Ordinal': False,
        'Unknown': 58924 / 1722503,
        'Values to Exclude': None
    },
    'ETHNIC': {
        'Description': 'Hispanic or Latino Origin',
        'Number of Categories': 6,
        'Ordinal': False,
        'Unknown': 60872 / 1722503,
        'Values to Exclude': None
    },
    'MARSTAT': {
        'Description': 'Marital Status',
        'Number of Categories': 5,
        'Ordinal': False,
        'Unknown': 360937 / 1722503,
        'Values to Exclude': None
    },
    'EDUC': {
        'Description': 'Education',
        'Number of Categories': 6,
        'Ordinal': True,
        'Unknown': 174935 / 1722503,
        'Values to Exclude': None
    },
    'EMPLOY': {
        'Description': 'Employment Status at Admission',
        'Number of Categories': 5,
        'Ordinal': False,
        'Unknown': 152365 / 1722503,
        'Values to Exclude': None
    },
    'EMPLOY_D': {
        'Description': 'Employment Status at Discharge',
        'Number of Categories': 12,
        'Ordinal': False,
        'Unknown': 322858 / 1722503,
        'Values to Exclude': None
    },
    'DETNLF': {
        'Description': 'Detailed Not in Labor Force at Admission',
        'Number of Categories': 6,
        'Ordinal': False,
        'Unknown': 1251911 / 1722503,
        'Values to Exclude': None
    },
    'DETNLF_D': {
        'Description': 'Detailed Not in Labor Force at Discharge',
        'Number of Categories': 6,
        'Ordinal': False,
        'Unknown': 1332282 / 1722503,
        'Values to Exclude': None
    },
    'PREG': {
        'Description': 'Pregnant at Admission',
        'Number of Categories': 3,
        'Ordinal': False,
        'Unknown': 1136767 / 1722503,
        'Values to Exclude': None
    },
    'VET': {
        'Description': 'Veteran Status',
        'Number of Categories': 3,
        'Ordinal': False,
        'Unknown': 174003 / 1722503,
        'Values to Exclude': None
    },
    'LIVARAG': {
        'Description': 'Living Arrangements at Admission',
        'Number of Categories': 4,
        'Ordinal': False,
        'Unknown': 174307 / 1722503,
        'Values to Exclude': None
    },
    'LIVARAG_D': {
        'Description': 'Living Arrangements at Discharge',
        'Number of Categories': 4,
        'Ordinal': False,
        'Unknown': 354754 / 1722503,
        'Values to Exclude': None
    },
    'PRIMINC': {
        'Description': 'Source of Income/Support',
        'Number of Categories': 6,
        'Ordinal': False,
        'Unknown': 680082 / 1722503,
        'Values to Exclude': None
    },
    'ARRESTS': {
        'Description': 'Arrests in Past 30 Days Prior to Admission',
        'Number of Categories': 4,
        'Ordinal': True,
        'Unknown': 182116 / 1722503,
        'Values to Exclude': None
    },
    'ARRESTS_D': {
        'Description': 'Arrests in Past 30 Days Prior to Discharge',
        'Number of Categories': 4,
        'Ordinal': True,
        'Unknown': 340163 / 1722503,
        'Values to Exclude': None
    },
    'DAYWAIT': {
        'Description': 'Days Waiting to Enter Substance Use Treatment',
        'Number of Categories': 5,
        'Ordinal': True,
        'Unknown': 920756 / 1722503,
        'Values to Exclude': None
    },
    'PSOURCE': {
        'Description': 'Referral Source',
        'Number of Categories': 8,
        'Ordinal': False,
        'Unknown': 120477 / 1722503,
        'Values to Exclude': None
    },
    'DETCRIM': {
        'Description': 'Detailed Criminal Justice Referral',
        'Number of Categories': 9,
        'Ordinal': False,
        'Unknown': (32175 + 1397909) / 1722503,
        'Values to Exclude': None
    },
    'NOPRIOR': {
        'Description': 'Previous Substance Use Treatment Episodes',
        'Number of Categories': 3,
        'Ordinal': False,
        'Unknown': 136613 / 1722503,
        'Values to Exclude': None
    },
    'DSMCRIT': {
        'Description': 'DSM Diagnosis',
        'Number of Categories': 20,
        'Ordinal': False,
        'Unknown': 410311 / 1722503,
        'Values to Exclude': None
    },
    'PSYPROB': {
        'Description': 'Co-Occurring Mental and Substance Use Disorders',
        'Number of Categories': 3,
        'Ordinal': False,
        'Unknown': 225375 / 1722503,
        'Values to Exclude': None
    },
    'FREQ_ATND_SELF_HELP': {
        'Description': 'Attendance at Substance Use Self-Help Groups in Past '
                       '30 Days Prior to Admission',
        'Number of Categories': 6,
        'Ordinal': True,
        'Unknown': 327054 / 1722503,
        'Values to Exclude': None
    },
    'FREQ_ATND_SELF_HELP_D': {
        'Description': 'Attendance at Substance Use Self-Help Groups in Past '
                       '30 Days Prior to Discharge',
        'Number of Categories': 6,
        'Ordinal': True,
        'Unknown': 376995 / 1722503,
        'Values to Exclude': None
    },
    'HLTHINS': {
        'Description': 'Health Insurance',
        'Number of Categories': 5,
        'Ordinal': False,
        'Unknown': 868565 / 1722503,
        'Values to Exclude': None
    },
    'PRIMPAY': {
        'Description': 'Payment Source, Primary (Expected or Actual)',
        'Number of Categories': 8,
        'Ordinal': False,
        'Unknown': 989974 / 1722503,
        'Values to Exclude': None
    },

}

# DEFAULT DICTIONARY - DO NOT EDIT
# Specific To Treatment Type / Facility Type
facility = {
    'SERVICES': {
        'Description': 'Type of Treatment/Service Setting at Admission',
        'Number of Categories': 9,
        'Ordinal': False,
        'Unknown': 0 / 1722503,
        'Values to Exclude': None
    },
    'METHUSE': {
        'Description': 'Medication-Assisted Opioid Therapy',
        'Number of Categories': 3,
        'Ordinal': False,
        'Unknown': 178805 / 1722503,
        'Values to Exclude': None
    },

}

# DEFAULT DICTIONARY - DO NOT EDIT
# Geographic Data Points
geographic = {
    'STFIPS': {
        'Description': 'Census State FIPS Code',
        'Number of Categories': 49,
        'Ordinal': False,
        'Unknown': 0 / 1722503,
        'Values to Exclude': None,
        'Notes': 'Not all states included? Count includes Puerto Rico and '
                 'District of Columbia. Does not include Oregon, Washington, '
                 'West Virginia.'
    },
    'REGION': {
        'Description': 'Census Region',
        'Number of Categories': 5,
        'Ordinal': False,
        'Unknown': 0 / 1722503,
        'Values to Exclude': None,
        'Notes': 'The country chopped up coarsely.'
    },
    'DIVISION': {
        'Description': 'Census Division',
        'Number of Categories': 10,
        'Ordinal': False,
        'Unknown': 0 / 1722503,
        'Values to Exclude': None,
        'Notes': 'Slightly finer tuned region of country than Region.'
    },
    'CBSA2010': {
        'Description': 'Core Based Statistical Area',
        'Number of Categories': 'Many',
        'Ordinal': False,
        'Unknown': 0 / 1722503,
        'Values to Exclude': None,
        'Notes': 'Numerical index of areas anchored to urban areas of greater '
                 '10,000 people.'
    },
}

# DEFAULT DICTIONARY - DO NOT EDIT
# Drug and Psych Data, Categorical
drug_cat = {
    'SUB1': {
        'Description': 'Substance Use at Admission (Primary)',
        'Number of Categories': 20,
        'Ordinal': False,
        'Unknown': 99426 / 1722503,
        'Values to Exclude': None
    },
    'SUB1_D': {
        'Description': 'Substance Use at Discharge (Primary).',
        'Number of Categories': 20,
        'Ordinal': False,
        'Unknown': 217220 / 1722503,
        'Values to Exclude': None
    },
    'ROUTE1': {
        'Description': 'Route of Administration (Primary)',
        'Number of Categories': 6,
        'Ordinal': False,
        'Unknown': 142828 / 1722503,
        'Values to Exclude': None
    },
    'FREQ1': {
        'Description': 'Frequency of Use at Admission (Primary)',
        'Number of Categories': 4,
        'Ordinal': True,
        'Unknown': 196872 / 1722503,
        'Values to Exclude': None
    },
    'FREQ1_D': {
        'Description': 'Frequency of Use at Discharge (Primary)',
        'Number of Categories': 4,
        'Ordinal': True,
        'Unknown': 537124 / 1722503,
        'Values to Exclude': None
    },
    'FRSTUSE1': {
        'Description': 'Age at First Use (Primary)',
        'Number of Categories': 8,
        'Ordinal': True,
        'Unknown': 150579 / 1722503,
        'Values to Exclude': None
    },
    'SUB2': {
        'Description': 'Substance Use at Admission (Secondary)',
        'Number of Categories': 20,
        'Ordinal': False,
        'Unknown': 62316 / 1722503,
        'Values to Exclude': None
    },
    'SUB2_D': {
        'Description': 'Substance Use at Discharge (Secondary).',
        'Number of Categories': 20,
        'Ordinal': False,
        'Unknown': 123173 / 1722503,
        'Values to Exclude': None
    },
    'ROUTE2': {
        'Description': 'Route of Administration (Secondary)',
        'Number of Categories': 6,
        'Ordinal': False,
        'Unknown': 817221 / 1722503,
        'Values to Exclude': None
    },
    'FREQ2': {
        'Description': 'Frequency of Use at Admission (Secondary)',
        'Number of Categories': 4,
        'Ordinal': True,
        'Unknown': 818003 / 1722503,
        'Values to Exclude': None
    },
    'FREQ2_D': {
        'Description': 'Frequency of Use at Discharge (Secondary)',
        'Number of Categories': 4,
        'Ordinal': True,
        'Unknown': 1183470 / 1722503,
        'Values to Exclude': None
    },
    'FRSTUSE2': {
        'Description': 'Age at First Use (Secondary)',
        'Number of Categories': 8,
        'Ordinal': True,
        'Unknown': 844104 / 1722503,
        'Values to Exclude': None
    },
    'SUB3': {
        'Description': 'Substance Use at Admission (Tertiary)',
        'Number of Categories': 20,
        'Ordinal': False,
        'Unknown': 185870 / 1722503,
        'Values to Exclude': None
    },
    'SUB3_D': {
        'Description': 'Substance Use at Discharge (Tertiary).',
        'Number of Categories': 20,
        'Ordinal': False,
        'Unknown': 82543 / 1722503,
        'Values to Exclude': None
    },
    'ROUTE3': {
        'Description': 'Route of Administration (Tertiary)',
        'Number of Categories': 6,
        'Ordinal': False,
        'Unknown': 1362519 / 1722503,
        'Values to Exclude': None
    },
    'FREQ3': {
        'Description': 'Frequency of Use at Admission (Tertiary)',
        'Number of Categories': 4,
        'Ordinal': True,
        'Unknown': 1372001 / 1722503,
        'Values to Exclude': None
    },
    'FREQ3_D': {
        'Description': 'Frequency of Use at Discharge (Tertiary)',
        'Number of Categories': 4,
        'Ordinal': True,
        'Unknown': 1441000 / 1722503,
        'Values to Exclude': None
    },
    'FRSTUSE3': {
        'Description': 'Age at First Use (Tertiary)',
        'Number of Categories': 8,
        'Ordinal': True,
        'Unknown': 1371909 / 1722503,
        'Values to Exclude': None
    },
    'ALCDRUG': {
        'Description': 'Substance Use Type',
        'Number of Categories': 4,
        'Ordinal': False,
        'Unknown': 0 / 1722503,
        'Values to Exclude': None
    },

}

# DEFAULT DICTIONARY - DO NOT EDIT
# Drug and Psych Data, True/False
drug_flags = {
    'IDU': {
        'Description': 'Current IV Drug Use Reported at Admission',
        'Number of Categories': 3,
        'Ordinal': False,
        'Unknown': 0 / 1722503,
        'Values to Exclude': None
    },
    'ALCFLG': {
        'Description': 'Alcohol Reported at Admission',
        'Number of Categories': 2,
        'Ordinal': False,
        'Unknown': 0 / 1722503,
        'Values to Exclude': None
    },
    'COKEFLG': {
        'Description': 'Cocaine/Crack Reported at Admission',
        'Number of Categories': 2,
        'Ordinal': False,
        'Unknown': 0 / 1722503,
        'Values to Exclude': None
    },
    'MARFLG': {
        'Description': 'Marijuana/Hashish Reported at Admission',
        'Number of Categories': 2,
        'Ordinal': False,
        'Unknown': 0 / 1722503,
        'Values to Exclude': None
    },
    'HERFLG': {
        'Description': 'Heroin Reported at Admission',
        'Number of Categories': 2,
        'Ordinal': False,
        'Unknown': 0 / 1722503,
        'Values to Exclude': None
    },
    'METHFLG': {
        'Description': 'Non-Prescribed Methadone Reported at Admission',
        'Number of Categories': 2,
        'Ordinal': False,
        'Unknown': 0 / 1722503,
        'Values to Exclude': None
    },
    'OPSYNFLG': {
        'Description': 'Other Opiates/Synthetics Reported at Admission',
        'Number of Categories': 2,
        'Ordinal': False,
        'Unknown': 0 / 1722503,
        'Values to Exclude': None
    },
    'PCPFLG': {
        'Description': 'PCP Reported at Admission',
        'Number of Categories': 2,
        'Ordinal': False,
        'Unknown': 0 / 1722503,
        'Values to Exclude': None
    },
    'HALLFLG': {
        'Description': 'Hallucinogens Reported at Admission',
        'Number of Categories': 2,
        'Ordinal': False,
        'Unknown': 0 / 1722503,
        'Values to Exclude': None
    },
    'MTHAMFLG': {
        'Description': 'Methamphetamines/Speed Reported at Admission',
        'Number of Categories': 2,
        'Ordinal': False,
        'Unknown': 0 / 1722503,
        'Values to Exclude': None
    },
    'AMPHFLG': {
        'Description': 'Other Amphetamines Reported at Admission',
        'Number of Categories': 2,
        'Ordinal': False,
        'Unknown': 0 / 1722503,
        'Values to Exclude': None
    },
    'STIMFLG': {
        'Description': 'Other Stimulants Reported at Admission',
        'Number of Categories': 2,
        'Ordinal': False,
        'Unknown': 0 / 1722503,
        'Values to Exclude': None
    },
    'BENZFLG': {
        'Description': 'Benzodiazepines Reported at Admission',
        'Number of Categories': 2,
        'Ordinal': False,
        'Unknown': 0 / 1722503,
        'Values to Exclude': None
    },
    'TRNQFLG': {
        'Description': 'Other Tranquilizers Reported at Admission',
        'Number of Categories': 2,
        'Ordinal': False,
        'Unknown': 0 / 1722503,
        'Values to Exclude': None
    },
    'BARBFLG': {
        'Description': 'Barbiturates Reported at Admission',
        'Number of Categories': 2,
        'Ordinal': False,
        'Unknown': 0 / 1722503,
        'Values to Exclude': None
    },
    'SEDHPFLG': {
        'Description': 'Other Sedatives/Hypnotics Reported at Admission',
        'Number of Categories': 2,
        'Ordinal': False,
        'Unknown': 0 / 1722503,
        'Values to Exclude': None
    },
    'INHFLG': {
        'Description': 'Inhalants Reported at Admission',
        'Number of Categories': 2,
        'Ordinal': False,
        'Unknown': 0 / 1722503,
        'Values to Exclude': None
    },
    'OTCFLG': {
        'Description': 'Over-the-Counter Medication Reported at Admission',
        'Number of Categories': 2,
        'Ordinal': False,
        'Unknown': 0 / 1722503,
        'Values to Exclude': None
    },
    'OTHERFLG': {
        'Description': 'Other Drug Reported at Admission',
        'Number of Categories': 2,
        'Ordinal': False,
        'Unknown': 0 / 1722503,
        'Values to Exclude': None
    },

}

# DEFAULT DICTIONARY - DO NOT EDIT
# Excluded Data
excluded = {
    'CASEID': {
        'Description': 'Case Identification Number',
        'Number of Categories': 1722503,
        'Ordinal': False,
        'Unknown': 0 / 1722503,
        'Values to Exclude': None
    },
    'DISYR': {
        'Description': 'Year of Discharge',
        'Number of Categories': 1,
        'Ordinal': True,
        'Unknown': 0 / 1722503,
        'Values to Exclude': None
    },
    'SERVICES_D': {
        'Description': 'Types of Treatment/Settings at Discharge',
        'Number of Categories': 8,
        'Ordinal': False,
        'Unknown': 0 / 1722503,
        'Values to Exclude': None,
        'Notes': 'Identical to SERVICES.'
    },
    'LOS': {
        'Description': 'Length of Stay in Treatment',
        'Number of Categories': 37,
        'Ordinal': True,
        'Unknown': 0 / 1722503,
        'Values to Exclude': None
    }

}