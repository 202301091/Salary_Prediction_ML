from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load model files
model = pickle.load(open('model.pkl', 'rb'))
encoder = pickle.load(open('ordinal_encoder.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))


@app.route('/', methods=['GET', 'POST'])
def home():

    prediction = None

    if request.method == 'POST':

        data = {
            'job_title': request.form['job_title'],
            'experience_years': int(request.form['experience_years']),
            'education_level': request.form['education_level'],
            'skills_count': int(request.form['skills_count']),
            'industry': request.form['industry'],
            'company_size': request.form['company_size'],
            'location': request.form['location'],
            'certifications': int(request.form['certifications'])
        }

        df = pd.DataFrame([data])

        df[['education_level', 'company_size']] = encoder.transform(
            df[['education_level', 'company_size']]
        )

        df = pd.get_dummies(
            df,
            columns=['industry', 'job_title', 'location'],
            drop_first=True,
            dtype=int
        )

        df = df.reindex(
            columns=columns,
            fill_value=0
        )

        salary = model.predict(df)[0]

        prediction = f"${salary:,.0f} / Year"

    return render_template(
        'index.html',
        prediction=prediction
    )


if __name__ == '__main__':
    app.run(debug=True)