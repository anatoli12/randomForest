from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Зареждане на набора от данни за Ирис
iris = load_iris()
X = iris.data
y = iris.target

# Разделяне на набора от данни на обучение и тестове
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Създаване на класификатор Random Forest
# n_estimators е броят на дърветата в гората
# random_state се използва за възпроизводимост на резултатите
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Обучение на класификатора с обучаващите данни
clf.fit(X_train, y_train)

# Предсказване на етикетите на тестовите данни
y_pred = clf.predict(X_test)

# Изчисляване на точността на предсказанията
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
