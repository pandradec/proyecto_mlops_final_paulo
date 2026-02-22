import requests

url = "http://127.0.0.1:5001/invocations"

columns = [
    'Unnamed: 0',
    'Age',
    'Credit amount',
    'Duration',
    'Sex_male',
    'Job_1',
    'Job_2',
    'Job_3',
    'Housing_own',
    'Housing_rent',
    'Saving accounts_little',
    'Saving accounts_moderate',
    'Saving accounts_quite rich',
    'Saving accounts_rich',
    'Checking account_little',
    'Checking account_moderate',
    'Checking account_rich',
    'Purpose_car',
    'Purpose_domestic appliances',
    'Purpose_education',
    'Purpose_furniture/equipment',
    'Purpose_radio/TV',
    'Purpose_repairs',
    'Purpose_vacation/others'
]

# Creamos fila con todo en 0
row = [0] * len(columns)

# Ahora activamos valores correctos
row[0] = 0          # Unnamed: 0 (puede ser 0)
row[1] = 35         # Age
row[2] = 3000       # Credit amount
row[3] = 24         # Duration

# Sexo: male
row[4] = 1          # Sex_male

# Job = 2
row[6] = 1          # Job_2

# Housing = own
row[8] = 1          # Housing_own

# Saving accounts = little
row[10] = 1         # Saving accounts_little

# Checking account = moderate
row[15] = 1         # Checking account_moderate

# Purpose = car
row[17] = 1         # Purpose_car

data = {
    "dataframe_split": {
        "columns": columns,
        "data": [row]
    }
}

response = requests.post(url, json=data)

print("Status code:", response.status_code)
print("Response:", response.json())