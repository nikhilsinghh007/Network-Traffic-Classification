import pandas as pd    #a powerful library for data manipulation and analysis. We use 'pd' as a shorthand to call its functions.
from sklearn.model_selection import train_test_split   #This function from sklearn splits your data into training and testing sets.
from sklearn.preprocessing import StandardScaler, LabelEncoder   #StandardScaler : Used to standardize features (i.e., scale them so                                                                       they have a mean of 0 and a standard deviation of 1).  
                                                        #labelEncoder : Converts categorical labels (like text) into numerical labels.
from sklearn.svm import SVC                             #Support Vector Classifier, a type of machine learning model.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  #These functions help evaluate the performance of                                                                                         your model.

# Load the testing set
df = pd.read_csv("UNSW_NB15_testing-set.csv")     #df = dataframe, like a table where you can store & manipulate your data.

# Encode categorical features
categorical_columns = df.select_dtypes(include=['object']).columns  #This selects all columns in the DataFrame that contain categorical                                                                       data (i.e., text).

for column in categorical_columns:
    le = LabelEncoder()            #LabelEncoder: Converts categorical labels (like text) into numerical labels.
    df[column] = le.fit_transform(df[column])  #Converts the text in each column into numbers. e.g, "tcp" become 1, "udp" become 2, etc

# Preprocess the data
X = df.drop('label', axis=1)  #This drops the label column from the Df, leaving only the features in X. We don’t want to use the labels                                 as input to the model.
y = df['label']               #This is the label column, which we want to use as the target variable for the model.

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   #This splits the data into training and testing sets. test_size=0.2 means 20% of the data is used for testing, random_state=42 ensures that the split is reproducible.

# Standardize the features (optional, but recommended for SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   #Fits the scaler to the training data and then transforms it(i.e., scales it).
X_test = scaler.transform(X_test)        #Same transformation to the testing data. It’s important to use scaling for both training & testing data.    
# Train an SVM(Support Vector Classifier) model
model = SVC(kernel='linear')  #Creates an SVM model with a linear kernel, a simple and fast way to find a decision boundary between classes.
model.fit(X_train, y_train)   #Trains the SVM model using the training data (X_train and y_train).

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)    #Calculates the accuracy, which is the percentage of correct predictions.
precision = precision_score(y_test, y_pred)  #Calculates the precision, which is the percentage of true positives out of all predicted positives.
recall = recall_score(y_test, y_pred)        #Calculates the recall, which is the percentage of true positives out of all actual positives.
f1 = f1_score(y_test, y_pred)                #Calculates the F1-score, which is a balance between precision and recall.

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
