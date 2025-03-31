Predicting heart disease using machine learning
This notebook looks into various Python-based machine learning and data science libraries in an atempt to build a machine learning model capable of predicting whether or not someone has heart disease or not based on their medical attributes.

We're going to take the following approach:

Problem definition
Data
Evaluation
Features
Modelling
Experimentation
1. Problem Definition
In a statement,

Given clinical parameters about a patient, can we predict whether they have heart disease?

2. Data
The original data came from the Cleaveland data from the UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/45/heart+disease There is also a version of it available on Kaggle. https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data/data

3. Evaluation
If we can reach 95% accuracy at predicting whether or not a patient has heart disease during the proof of concept, then we'll pursue the project.

4. Features
This is where you'll get different information about each of the features in your data. You can do this via your own research (such as looking at the links above) or by talking to a subject matter expert (someone who knows about the dataset).

Create a data dictionary

age - age in years
sex - (1 = male; 0 = female)
cp - chest pain type
Typical angina: chest pain related decrease blood supply to the heart
Atypical angina: chest pain not related to heart
Non-anginal pain: typically esophageal spasms (non heart related)
Asymptomatic: chest pain not showing signs of disease
trestbps - resting blood pressure (in mm Hg on admission to the hospital) anything above 130-140 is typically cause for concern
chol - serum cholestoral in mg/dl *serum = LDL + HDL + .2 * triglycerides *above 200 is cause for concern
fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false) *'>126' mg/dL signals diabetes
restecg - resting electrocardiographic results
0: Nothing to note
1: ST-T Wave abnormality
can range from mild symptoms to severe problems
signals non-normal heart beat *2: Possible or definite left ventricular hypertrophy
Enlarged heart's main pumping chamber
thalach - maximum heart rate achieved
exang - exercise induced angina (1 = yes; 0 = no)
oldpeak - ST depression induced by exercise relative to rest looks at stress of heart during excercise unhealthy heart will stress more
slope - the slope of the peak exercise ST segment
0: Upsloping: better heart rate with excercise (uncommon)
1: Flatsloping: minimal change (typical healthy heart)
2: Downslopins: signs of unhealthy heart
ca - number of major vessels (0-3) colored by flourosopy
colored vessel means the doctor can see the blood passing through
the more blood movement the better (no clots)
thal - thallium stress result
1,3: normal
6: fixed defect: used to be defect but ok now
7: reversable defect: no proper blood movement when excercising
target - have disease or not (1=yes, 0=no) (= the predicted attribute)
5. Modelling
We'll use a few machine learning models to make the predictions on the data we have, which are: LogisticRegressor() (despite the name, it's a linear classification model often used for classification problems), K-Neighbors() and RandomForestClassifier(). First, we'll test those three as they are; if the models can't reach 95% accuracy (project's goal) after being trained and tested, we'll tune them to get a better score. Only the best-performing model will be used as reference to our predictions.

6. Experimenting
After some EDA and hyperparameter tuning, it's time to put the models to work and rate their performance. For this, we're going to use all of the cross-validated classification metrics below for evaluation completeness:

Accuracy:
How much of the data the models predicted right between classes 0 and 1. This one is better used for balanced-class data frames, such as ours, but it doesn't take into account how much each feature the model predicts correctly.
Precision:
How much of the positive predictions were actually correct. Thus, it's a weighted measure of the model's capacity to point out the correct positive values; however, this metric doesn't consider the false negatives.
Recall:
Also known as "true positive rate", which calculates the model's score over the true cases it identifies and the true cases it misses in percentage (%). It doesn't measure the false positive rate though.
F1-score:
The F1-score is a harmonic mean of precision and recall, meaning it gives equal weight for both metrics. This metric can tell the reliability of the predictions made.
