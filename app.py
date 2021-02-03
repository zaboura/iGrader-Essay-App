from flask import Flask, render_template, request    
from tensorflow.keras.models import load_model 
from essay_grading_sys import pipeline_
from src import *
import tensorflow as tf



app = Flask(__name__)

model = load_model('models/model.h5', compile = False)

#model._make_predict_function()

scaler=load('models/std_scaler.bin')


@app.route('/predict',methods=['POST'])

def predict():
    
    text = request.form['essay']
    prompt = request.form['prompt']
    final_features, word_counts, prompt_simm, nbr_mistakes = pipeline_(text, prompt)
    
    sc_final_features = scaler.transform(final_features)
     
    prediction = model.predict(sc_final_features).round()
    
    final_output = inverse_class_labels_reassign(prediction[0].tolist())
    
    return render_template('index.html', essay_grade='{}/5'.format(final_output),
                           Prompt = prompt, Essay = text, Word_count= word_counts, Nbr_mistakes = nbr_mistakes,
                           Prompt_sim ='{}' .format(round(prompt_simm*100)))



@app.route("/")
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
    