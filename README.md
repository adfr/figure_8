# Disaster Response Pipeline Project



### Objective
Provide classification of disaster messages using a multilayer perceptron.


### Folders
<pre>
Figure8/
├── README.md  
├── app/  
		├──	run.py # Flask file that runs app  
		├── templates/  
                ├──	master.html  
				└── go.html  
├── data/  
		├── categories.csv  
		├──	messages.csv   
		├── process_data.py  
		└── DisasterResponse.db    
└── models/  
		└── train_classifier.py  
</pre>

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Go to http://0.0.0.0:3001/

### Results

2 Algorithms were tested out:
- Multilayer perceptron
- Random forest 
	Both can be used to for multiouput classifcation
	Accurracy reached 93.81%.
	The features used were TF-IDF and verb extractors.

