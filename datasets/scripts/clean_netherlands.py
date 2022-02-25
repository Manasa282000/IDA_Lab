import pandas as pd
import pathlib


original_datasets = pathlib.Path("original_datasets")

path_netherlands = original_datasets / "netherlands"

raw_input = path_netherlands / "netherlands_raw.csv.gz"
output_csv = path_netherlands / "netherlands.csv.gz"

data = pd.read_csv(raw_input, sep=";")

print(f"read input data: {data.shape}")
print(f"Coluns: {data.columns}")

# drop the samp column, that just indicated whether it was training or test set
data = data.drop('samp', axis=1)

# drop the dichtheid column, as it was not defined anywhere.
data = data.drop('dichtheid', axis=1)

# rename and order of columns
colmap = [
    # original              # new columns
    ("sekse",               "sex=female"),
    ("cgebland",            "country_of_birth"),
    ("lnvgalguz",           "log_#_of_previous_penal_cases"),
    ("leeftijd",            "age_in_years"),
    ("lftinsz1inclvtt",     "age_at_first_penal_case"),
    ("delcuziv",            "offence_type"),
    ("dum1120",             "11-20_previous_case"),
    ("dum21plus",           ">20_previous_case"),
    ("lft2",                "age_squared"),
    ("rec4",                "recidivism_in_4y"),
]

# do the renaming of the columns
print("renaming columns:")
for c in colmap:
    print(f"  - {c[0]:15}  =>  {c[1]}")
data.rename(columns={ x[0] : x[1] for x in colmap}, inplace = True)

# reorder the columns
new_columns = [x[1] for x in colmap]
print(f'Changing column order: {new_columns}')
data = data.reindex(columns=new_columns)

# save the dataset
print(f"saving to '{output_csv}'")
data.to_csv(output_csv, sep=";", index=False)

print("dataset cleaned")