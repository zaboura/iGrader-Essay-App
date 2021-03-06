# iGrader Essay

<div style="text-align:center"><img src="static/img/logo.png" alt="logo" width="200"/></div>

<sub><sup><sup style="color:gray">* Logo designed by *Mohammed EL Aziz*</sup></sub></sub>
## Automated Grading System for everyone

IGrader Essay is an Automated Essay Grading systems platform based on AI, it is a tool that can help you grade essays in English. It is used in conjunction or even independently of a human grader. Handy for busy teachers, or to students having doubts. It needs your essay and its prompt, to give your deserved grade and some feedback. Released in 2021, by motivated students in France, it is a friend who is never tired to help with your essays.

<div style="text-align:center">
<h3>Input section</h3>  
<img src="static/img/Front.png" alt="logo"/></div> 
<br><br>




<div style="text-align:center">
<h3>Results section</h3>
<img src="static/img/Results.png" alt="logo"/></div>


<br><br>

## Why *__iGrader Essay__* is different?

In all reviewed automated grading systems, only the embedding technique was adopted, however, in our application, we followed the more complex path. We extract features that a real teacher based grading on to train our multi-layer perceptron.
Find the list of some extracted features:

 * Number of mistakes.
 * Number of words.
 * Lexical diversity.
 * Average sparse tree height
 * Inner similarities: We used the *BERT* model to embed essays' sentences and compute *cosine similarity distance* between them.
 * Text coherence Score: GPT-2 language model used to compute the probability.
 * Number of transition words: Count of linking words.
 * Prompt and essay relevance: Universal-sentence-encoder-large (v5) used to compute similarity.
 * .
 * .
 * etc 
## Running locally

A step by step series of examples that tell you how to get a development env running

After cloning the repository in your local machine, create a conda environment by running the following line:

```
conda create --name igrader-essay-env python=3.6
```

And then install the required dependencies within the ```igrader-essay-env``` environment by running the following line

```
pip3 install -r requirements.txt
```

At the end run the application locally:

```
python app.py
```

The default development UR is [```http://127.0.0.1:5000/```](http://127.0.0.1:5000/)

## Built With

* [Flask](https://flask.palletsprojects.com/en/1.1.x/) - The web framework used
* [Docker](https://docs.docker.com/) - Dependency Management & Envirenment
* [Azure Microsoft](https://docs.microsoft.com/en-us/azure/?product=ai-machine-learning) - Used to host the application

## Contributing

If you want to contribute to a project and make it better, your help is very welcome. Contributing is also a great way to learn more about social coding on Github, new technologies and their ecosystems, how to make constructive, helpful bug reports, feature requests, and the noblest of all contributions: a good, clean pull request.

## Authors

* **Abdelhak Zabour**
* **Jehona Kryeziu**
* **Meziane Bellahmer**

See also the list of [contributors](https://github.com/zaboura/iGrader-Essay-App/graphs/contributors) who participated in this project.

## License

This project is licensed under Attribution-NonCommmercial-ShareAlike 4.0 International.

[![License: CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/80x15.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
