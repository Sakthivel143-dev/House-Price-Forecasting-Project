from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('model.joblib')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    if request.method == 'POST':
        try:
            size = float(request.form['size'])
            bedrooms = int(request.form['bedrooms'])
            age = float(request.form['age'])

            prediction = model.predict([[size, bedrooms, age]])[0]
            prediction_text = f"Estimated Property Value: ₹{prediction:,.0f}"
        except Exception as e:
            prediction_text = f"Error: {e}"

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
