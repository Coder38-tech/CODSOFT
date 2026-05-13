import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('titanic.csv')

df.columns = df.columns.str.strip().str.capitalize()

df['Age'] = df['Age'].fillna(df['Age'].median())

if 'Sex' in df.columns:
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# This list matches the capitalized names and handles missing columns safely
to_drop = [c for c in ['Name', 'Ticket', 'Cabin', 'Passengerid'] if c in df.columns]
df.drop(to_drop, axis=1, inplace=True)

# Drop any remaining rows with missing values (like Embarked)
df.dropna(inplace=True)

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Model Accuracy:", accuracy_score(y_test, model.predict(X_test)))
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

custom_palette = {0: "#C0392B", 1: "#27AE60"} 


ax = sns.countplot(x='Survived', data=df, hue='Survived', palette=custom_palette, legend=False)

ax.set_xticks([0, 1])
ax.set_xticklabels(['Died', 'Survived'])

plt.title('Titanic Survival Analysis')
plt.xlabel('Passenger Outcome')
plt.ylabel('Number of Passengers')

plt.tight_layout()
plt.show()
