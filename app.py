# app.py
from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.cluster import AgglomerativeClustering

app = Flask(__name__)

# Load models and accuracy files
with open('model1.pkl', 'rb') as file:
    classification_model = pickle.load(file)

with open('classification_accuracy-2.pkl', 'rb') as file:
    accuracy_classification = pickle.load(file)

with open('clustering_accuracy-2.pkl', 'rb') as file:
    accuracy_clustering = pickle.load(file)

with open('model2-2.pkl', 'rb') as file:
    clustering_model = pickle.load(file)

# Load the dataset
df = pd.read_csv('bank.csv')
X = df.drop('deposit', axis=1)
y = df['deposit']

# One-hot encode the input features
one_hot_encoder = OneHotEncoder()
X_encoded = one_hot_encoder.fit_transform(X)

# Load the label encoder for inverse transformation of class labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets (optional, if needed)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)


# Route to the home page

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html', accuracy_classification=accuracy_classification,
                           accuracy_clustering=accuracy_clustering)


# Route to handle user input and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        input1 = int(request.form.get('age'))
        input2 = request.form.get('job')
        input3 = request.form.get('marital')
        input4 = request.form.get('education')
        input5 = request.form.get('default')

        input6 = int(request.form.get('balance'))
        # input6 = int(input6) if input6 is not None else 0
        #
        input7 = request.form.get('housing')
        input8 = request.form.get('loan')
        input9 = request.form.get('contact')
        input10 = int(request.form.get('day'))
        input11 = request.form.get('month')
        input12 = int(request.form.get('duration'))
        input13 = int(request.form.get('campaign'))
        input14 = int(request.form.get('pdays'))

        # The 'previous' input might be empty, so handle it accordingly
        input15 = int(request.form.get('previous'))
        # input15 = int(previous_input) if previous_input else 0

        input16 = request.form.get('poutcome')

        # Create a DataFrame for user input
        # user_input_df = pd.DataFrame({
        #     'age': [input1],
        #     'job': [input2],
        #     'marital': [input3],
        #     'education': [input4],
        #     'default': [input5],
        #     'balance': [input6],
        #     'housing': [input7],
        #     'loan': [input8],
        #     'contact': [input9],
        #     'day': [input10],
        #     'month': [input11],
        #     'duration': [input12],
        #     'campaign': [input13],
        #     'pdays': [input14],
        #     'previous': [input15],
        #     'poutcome': [input16]
        # })
        user_input_df = pd.DataFrame([[input1, input2, input3, input4, input5, input6, input7, input8, input9, input10,
                                       input11, input12, input13, input14, input15, input16]],
                                     columns=X.columns)
        user_input_df = pd.DataFrame([[input1, input2, input3, input4, input5, input6, input7, input8, input9, input10,
                                       input11, input12, input13, input14, input15, input16]], columns=X.columns)
        user_input_encoded = one_hot_encoder.transform(user_input_df)

        # Convert user input data to a dense numpy array
        user_input_dense = user_input_encoded.toarray()

        # Fit the classification model with the training data (X_train should be dense)
        classification_model.fit(X_train.toarray(), y_train)
        clustering_model = AgglomerativeClustering(n_clusters=2)
        # Fit the clustering model with the data (X_encoded should be dense)
        clustering_model.fit(X_encoded.toarray())

        # Make predictions using dense data
        classification_prediction = classification_model.predict(user_input_dense)
        clustering_prediction = clustering_model.labels_


        #
        # # Fit the classification model with the training data
        # classification_model.fit(X_train, y_train)
        #
        # # Fit the clustering model with the data (excluding the target labels)
        # clustering_model.fit(X_encoded)
        #
        # # Make predictions
        # classification_prediction = classification_model.predict(user_input_encoded)
        # clustering_prediction = clustering_model.predict(user_input_encoded)

        # Convert predictions to original class label and cluster label using inverse_transform
        classification_prediction = label_encoder.inverse_transform(classification_prediction)
        clustering_prediction = label_encoder.inverse_transform(clustering_prediction)

        if classification_prediction[0] == '0':
            ans = 'No,customer is not likely to make term deposits'
        else:
            ans = 'yes, customer is more likely to make term deposits'

        if clustering_prediction[0] == '0':
            ans2 = 'In the cluster of not making term deposit'
        else:
            ans2 = 'In the cluster of making term deposit'

        return render_template('output.html', classification_prediction=ans,
                               clustering_prediction=ans2)


@app.route('/report', methods=['GET'])
def report():
    return render_template('Jaimeen_Bank_Pandas_Profiling.html')


if __name__ == '__main__':
    app.run(debug=True)
