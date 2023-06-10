import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import customtkinter
from CTkMessagebox import CTkMessagebox
#import graphviz
#from sklearn import tree

# Load the data from the four Excel files
age_gender_df = pd.read_excel('age_gender.xlsx')
treatments_df = pd.read_excel('treatments.xlsx')
diagnoses_df = pd.read_excel('diagnoses.xlsx')
symptoms_df = pd.read_excel('symptoms.xlsx')

# Merge the data based on 'patient_id'
merged_df = age_gender_df.merge(treatments_df, on='patient_id', how='inner')
merged_df = merged_df.merge(diagnoses_df, on='patient_id', how='inner')
merged_df = merged_df.merge(symptoms_df, on='patient_id', how='inner')

# To visualize merged data, convert to csv and open
#merged_df.to_csv('merged_data_new.csv', index=False)

# Extract the features (symptoms, previous diagnoses, age, gender), which is in order of decreasing significance
# Extract the target variable (current treatments column in merged_df)
X = merged_df[['symptoms', 'previous_diagnoses', 'age', 'gender']]
y = merged_df['current_treatments']

# Handle missing values in the target variable (N/A in treatments.xlsx)
# Missing values in the treatments are replaced with the treatment that appears the most frequently
target_imputer = SimpleImputer(strategy='most_frequent')
y_imputed = target_imputer.fit_transform(y.values.reshape(-1, 1)).flatten()

# Preprocess categorical variables (gender, symptoms, previous diagnoses)
# Categorical variables are variables that cannot be represented by numbers
categorical_columns = ['gender', 'symptoms', 'previous_diagnoses']
X_categorical = X[categorical_columns]

# Handle missing values in categorical variables (N/A)
# Missing values are replaced with the string 'Unknown'
categorical_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
X_categorical_imputed = categorical_imputer.fit_transform(X_categorical)

# One-hot encode categorical variables
# Convert categorical variables into a numerical representation to be used by machine learning algorithms
# Note that each unique string in the categorical variables is considered a new category
# Therefore is case-sensitive and does not understand similarities between strings
# Thus consider improving by cleaning up dataset entries
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X_categorical_imputed)

# Preprocess numerical variable (age)
# Numerical variables are variables that can be represented by numbers
numerical_columns = ['age']
X_numerical = X[numerical_columns]

# Handle missing values in numerical variables (N/A)
# Missing values are replaced with the mean of all numerical variables
numerical_imputer = SimpleImputer(strategy='mean')
X_numerical_imputed = numerical_imputer.fit_transform(X_numerical)

# Combine categorical and numerical variables
X_processed = np.concatenate((X_numerical_imputed, X_encoded), axis=1)

# Create the decision tree classifier
clf = DecisionTreeClassifier()

# Train the classifier using X_processed (pre-processed features: symptoms, previous diagnoses, age, gender)
# and y_imputed (pre-processed data on target variable: from treatments spreadsheet)
# Symptoms are at the top of the decision tree, being the most important feature
# A flowchart-like structure with various nodes that eventually lead up to a decision at the end (recommended treatment in this case)
# The decision tree is constructed by the program and each recommended treatment is determined by traversing the decision tree
clf.fit(X_processed, y_imputed)

# Graphical User Interface (GUI) appearance
customtkinter.set_appearance_mode("light")
customtkinter.set_default_color_theme("green")
root = customtkinter.CTk()
root.geometry("500x450")

# Function to get inputs from user, preprocess them, generate predicted recommended treatments, and display them to the user
def treatment_search():
    # Get input data from entry boxes in GUI
    input_age = int(entry1.get())
    input_gender = entry2.get()
    input_symptoms = entry3.get()
    input_previous_diagnoses = entry4.get()
    
    # Preprocess input data
    # Similar process before with categorical and numerical variables
    input_categorical = [[input_gender, input_symptoms, input_previous_diagnoses]]
    input_categorical_imputed = categorical_imputer.transform(input_categorical)
    input_encoded = encoder.transform(input_categorical_imputed)
    input_numerical = [[input_age]]
    input_numerical_imputed = numerical_imputer.transform(input_numerical)
    
    # Combine input variables
    input_processed = np.concatenate((input_numerical_imputed, input_encoded), axis=1)
    
    # Make predictions for the input data (based on decision tree classifier which was trained)
    prediction = clf.predict(input_processed.reshape(1, -1))
    
    # Get recommended treatments based on predicted values
    recommended_treatments = treatments_df[treatments_df['current_treatments'].isin(prediction)]['current_treatments'].values

    # Remove duplicate treatments
    recommended_treatments = np.unique(recommended_treatments)

    # Display the recommended treatments in box that pops up after the "Get Recommended Treatments" button is clicked
    CTkMessagebox(title="Recommended Treatments", message=", ".join(recommended_treatments))

# Labels, entry boxes, and button for GUI
frame = customtkinter.CTkFrame(master=root)
frame.pack(pady=20, padx=60, fill="both", expand=True)

label1 = customtkinter.CTkLabel(master=frame, text="Treatment Searcher", font=("Roboto", 24))
label1.pack(pady=12, padx=10)

label2 = customtkinter.CTkLabel(master=frame, text="WARNING: This program is not a replacement for a medical professional.\nRecommended treatments generated by this program must \nbe confirmed by a medical professional before patient use.\n\nNote: Separate multiple symptoms and previous diagnoses with a comma only.", font=("Roboto", 10))
label2.pack(pady=3, padx=10)

entry1 = customtkinter.CTkEntry(master=frame, placeholder_text="Age")
entry1.pack(pady=12, padx=10)

entry2 = customtkinter.CTkEntry(master=frame, placeholder_text="Male or Female")
entry2.pack(pady=12, padx=10)

entry3 = customtkinter.CTkEntry(master=frame, placeholder_text="Symptoms")
entry3.pack(pady=12, padx=10)

entry4 = customtkinter.CTkEntry(master=frame, placeholder_text="Previous Diagnoses")
entry4.pack(pady=12, padx=10)

# Call treatment_search function to allow for pop-up with recommended treatments
button = customtkinter.CTkButton(master=frame, text="Get Recommended Treatments", command=treatment_search)
button.pack(pady=12, padx=10)

# Run GUI
root.mainloop()

'''
# To visualize decision tree
# Symptoms are at the top of the decision tree, thus is the most important feature

# Select a subset of features (first 10) for visualization
selected_features = X.columns[:10] 

# Get the indices of the selected features in X_processed
selected_feature_indices = [X.columns.get_loc(feature) for feature in selected_features]

# Create a new decision tree classifier with the selected features
selected_clf = DecisionTreeClassifier(max_depth=3)
selected_clf.fit(X_processed[:, selected_feature_indices], y_imputed)

# Visualize the decision tree
dot_data = tree.export_graphviz(selected_clf, out_file=None, feature_names=selected_features)
graph = graphviz.Source(dot_data)
graph.view()
'''
