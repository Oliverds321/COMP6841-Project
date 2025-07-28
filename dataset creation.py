import pandas as pd
import random
#done
qandaexamples = pd.read_csv('q and a examples.csv', header=None)

qandaexamples.drop(columns=[3], inplace=True)
qandaexamples.columns = ['Question', 'Safe Response', 'Unsafe Response']

detailexamples = pd.read_csv('detailexamples.csv')

def generatephonenumber():
    return f"+61 04{random.randint(10, 99):02d} {random.randint(100, 999):03d} {random.randint(100, 999):03d}"

detailexamples['Mobile Phone'] = [generatephonenumber() for _ in range(len(detailexamples))]

newrows = []

for _ in range(2):
    for index, qrow in qandaexamples.iterrows():
        person = detailexamples.sample(1).iloc[0]
        
        name = person['Name']
        email = person['Email Address']
        phone = person['Mobile Phone']
        
        question = str(qrow['Question']) if pd.notna(qrow['Question']) else ""
        saferesponse = str(qrow['Safe Response']) if pd.notna(qrow['Safe Response']) else ""
        unsaferesponse = str(qrow['Unsafe Response']) if pd.notna(qrow['Unsafe Response']) else ""

        newquestion = question.replace('EXAMPLE', name)
        
        saferesponse = saferesponse.replace('EXAMPLE@EXAMPLE.com', email).replace('EXAMPLENUMBER', phone).replace('EXAMPLE', name)
        unsaferesponse = unsaferesponse.replace('EXAMPLE@EXAMPLE.com', email).replace('EXAMPLE@example.com', email).replace('EXAMPLENUMBER', phone).replace('EXAMPLE', name)
        
        newrows.append([newquestion, saferesponse, unsaferesponse])

combineddf = pd.DataFrame(newrows, columns=['Question', 'Safe Response', 'Unsafe Response'])

combineddf.to_csv('combined_examples.csv', index=False)