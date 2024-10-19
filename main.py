import pandas as pd
from NaiveBayes import NaiveBayes
from LaplaceSmoothing import LaplaceSmoothing

# For Laplace Smoothing:
# In the data preparation step, add the information about the number of levels. 
# This means that for each data column you should add the number of possible different values for that column.
def get_num_levels(df):
    num_levels = []
    for col in df.columns[:-1]:
        num_levels.append(df[col].nunique())
    return num_levels

def pre_processing(df):
    # Convert 'Windy' column to string for consistency
    df['Windy'] = df['Windy'].astype(str)
	
    # Select 10 random rows -> train set
    #random_state = int(time.time())
    df_train = df.sample(n=10, random_state=42)

    # test set
    df_test = df.drop(df_train.index)
    
    # Ensure test set has same features as training set or one less
    assert set(df_train.columns) == set(df_test.columns) or set(df_train.columns) == set(df_test.columns - 1), "Test set features do not match training set features."

    # partition data into features and target
    X_train = df_train.iloc[:,:-1]  # features
    y_train = df_train.iloc[:, -1]   # target
    
    X_test = df_test.iloc[:,:-1]  # features
    y_test = df_test.iloc[:, -1]   # target
    
    num_levels = get_num_levels(df)
    
    return X_train, X_test, y_train, y_test, num_levels

def accuracy_score(y_true, y_pred):

	"""	score = (y_true - y_pred) / len(y_true) """

	return round(float(sum(y_pred == y_true))/float(len(y_true)) * 100 ,2)



if __name__=='__main__':
    
    # preprocessing
    df = pd.read_csv("weather.data", delim_whitespace=True)    
    X_train, X_test, y_train, y_test, num_levels = pre_processing(df)
    
    # Naive Bayes classifier
    nb_clf = NaiveBayes()
    # Train the model -> calculate all probabilities
    nb_clf.fit(X_train, y_train)  
    
    # Test model with the extracted test set (X_test, y_test)
    predictions = nb_clf.predict(X_test)  # Get predictions on the test set

    print("Naive Bayes Classifier:")
    # Print predicted and actual label
    for i, (true_label, predicted_label) in enumerate(zip(y_test, predictions), 1):
        print(f"Test {i}: Actual label = {true_label}, Predicted label = {predicted_label}")
   
    # Calculate accuracy on the test set
    test_accuracy = accuracy_score(y_test, predictions)

    # Print results
    print(f"Test Accuracy: {test_accuracy}%")
    
    
    ########################################################################################
    # Task 3: Make the classifier robust to missing data with Laplace (additive) smoothing #
    ########################################################################################
    
    ls_clf = LaplaceSmoothing()
    # Train the model -> calculate all probabilities
    ls_clf.fit(X_train, y_train, num_levels) 
    
    # Test model with the extracted test set (X_test, y_test)
    predictions = ls_clf.predict(X_test)  # Get predictions on the test set

    print("\nAfter Laplace smoothing:")
    # Print predicted and actual label
    for i, (true_label, predicted_label) in enumerate(zip(y_test, predictions), 1):
        print(f"Test {i}: Actual label = {true_label}, Predicted label = {predicted_label}")
   
    # Calculate accuracy on the test set
    test_accuracy = accuracy_score(y_test, predictions)

    # Print results
    print(f"Test Accuracy: {test_accuracy}%")