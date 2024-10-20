# THEME
MACHINE LEARNING MODEL TO PREDICT PCOS
## PROBLEM STATEMENT
Polycystic Ovary Syndrome (PCOS) The syndrome is named after cysts which form on the ovaries of some women with this condition, that affects a significant percentage of women of reproductive age,  leading to complications such as infertility, metabolic syndrome, and an increased risk of developing diabetes and cardiovascular diseases. Early detection of PCOS can greatly improve treatment outcomes and quality of life.

## WHAT AND WHY WE SOLVE THIS PROBLEM
PCOS has become increasingly common in today's society, primarily due to lifestyle and health habits. This condition often leads to significant fertility issues for women. In response, we have taken the initiative to support women in adopting a healthier lifestyle and preventing potential diseases in the future.
## Solution approach
Our solution, AVAL.AI, addresses the challenges associated with PCOS by employing advanced machine learning models, including Random Forest and logistic regression, to predict the likelihood of PCOS in users. This optimized approach enhances accuracy in diagnosis and provides actionable insights. On the front end, we prioritize user experience by incorporating interactive input methods and presenting personalized lifestyle recommendations tailored to at-risk individuals. The wellness plans include dietary suggestions and exercise routines, ensuring a comprehensive approach to managing PCOS. This makes AVAL.AI both predictive and proactive, ultimately leading to improved health outcomes for users.

## MACHINE LEARNING MODEL TO PREDICT PCOS USING RANDOM FOREST

### IMPORING THE NECESSARY LIBRARIES
```
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from tabulate import tabulate
```

### Load the datasets

```data_path = 'CLEAN- PCOS SURVEY SPREADSHEET.csv'
df = pd.read_csv(data_path)

recipes_path = 'recipes.csv - Sheet1.csv'
recipes_df = pd.read_csv(recipes_path, encoding='latin-1')

exercise_path = 'exercise.csv - sheet1.csv'
exercise_df = pd.read_csv(exercise_path, encoding='latin-1')
```


### Fill missing values AND TRAIN THE TEST SPLIT
```
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)


target_column = 'Have you been diagnosed with PCOS/PCOD?'  # Adjust this as needed
X = df_imputed.drop(columns=[target_column])
y = df_imputed[target_column]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
### TRAIN THE MODEL
```# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))
```

### GENERATE THE RECOMMENDATIONS
```
def generate_diet_and_exercise_plan():
    days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    meals = ["Breakfast", "Lunch", "Dinner"]

    diet_table_data = [["Days", "Breakfast", "Lunch", "Dinner"]]
    exercise_table_data = [["Days", "Morning Exercise", "Evening Exercise"]]

    recipe_name_column = recipes_df.columns[0]
    exercise_name_column = exercise_df.columns[0]

    for day in days:
        diet_row = [day]
        for meal in meals:
            recipe = recipes_df.sample(1)
            diet_row.append(recipe[recipe_name_column].values[0])
        diet_table_data.append(diet_row)

        exercise_row = [day]
        morning_exercise = exercise_df.sample(1)
        evening_exercise = exercise_df.sample(1)
        exercise_row.extend([morning_exercise[exercise_name_column].values[0], evening_exercise[exercise_name_column].values[0]])
        exercise_table_data.append(exercise_row)

    print("\n--- Personalized Weekly Diet Plan ---")
    print(tabulate(diet_table_data, headers="firstrow", tablefmt="grid"))
    print("\n--- Personalized Weekly Exercise Plan ---")
    print(tabulate(exercise_table_data, headers="firstrow", tablefmt="grid"))
    print("\nPlease follow the meal and exercise plans to manage PCOS effectively.")```

### TAKING USER CHOICE
```
def pcos_chatbot():
    print("Welcome to the PCOS detection chatbot!")
    
    while True:
        print("\nYou can enter your request (e.g., 'predict PCOS', 'get diet and exercise recommendations', or 'quit').")
        
        user_input = input("\nWhat would you like to do? ").strip().lower()

       
        if "predict" in user_input or "diagnosis" in user_input:
            print("\nPlease answer the following questions for PCOS prediction:")
            user_data = []
            blood_group_mapping = {
                'A+': 11, 'A-': 12,
                'B+': 13, 'B-': 14,
                'O+': 15, 'O-': 16,
                'AB+': 17, 'AB-': 18
            }

            for feature in X.columns:
                if 'blood group' in feature.lower():
                    while True:
                        blood_group = input("Enter your blood group (e.g., A+, O-): ").upper()
                        if blood_group in blood_group_mapping:
                            value = blood_group_mapping[blood_group]
                            break
                        else:
                            print("Invalid blood group entered. Please try again.")
                else:
                    while True:
                        try:
                            value = float(input(f"Enter value for {feature} (numeric input expected): "))
                            break
                        except ValueError:
                            print("Invalid input. Please enter a numeric value.")
                user_data.append(value)

           
            user_data_scaled = scaler.transform([user_data])
            prediction = model.predict(user_data_scaled)

            if prediction == 1:
                print("\nThe chatbot predicts that you might have PCOS.")
            else:
                print("\nThe chatbot predicts that you likely do not have PCOS. However, consult a doctor if you have concerns.")

      
        elif "recommendation" in user_input or "recommend" in user_input:
            print("\nLet's collect more information for a personalized lifestyle plan.")
            busy_morning = input("Do you usually have a busy morning schedule? (yes/no): ").strip().lower()
            work_hours = int(input("On average, how many hours do you spend at work or school each day? (0-12): ").strip())
            exercise_freq = input("How often do you exercise each week? (never/1-2 times/3-5 times/daily): ").strip().lower()
            preferred_exercise_type = input("Do you prefer high-intensity or low-intensity exercises? (high/low): ").strip().lower()
            time_of_day_exercise = input("When do you prefer to exercise? (morning/evening/both): ").strip().lower()
            exercise_duration = int(input("How much time can you dedicate to each exercise session? (in minutes): ").strip())

            generate_diet_and_exercise_plan()

    
        elif "quit" in user_input:
            print("\nThank you for using the PCOS chatbot. Goodbye!")
            break

        else:
            print("\nInvalid input. Please enter a valid command (e.g., 'predict PCOS', 'get diet and exercise recommendations', or 'quit').")


pcos_chatbot()
```

# OUPUT

