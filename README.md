# My-project-report-about-Spider
The improved SyntaxSQLNet model

I. Improvements
Based on SyntaxSQLNet model set up by Yu, and Yasunaga et al, my model contains some improvements.
(1) Add a module to predict DISTINCT.
(2) Add a operator called BETWEEN.
(3) Improve the module predicting COLUMN.
(4) Predicting WHERE, HAVING, LIMIT.

II. Testing Environgment:
(1) Python == 3.6 64-bit
(2) Pytorch == 1.5.0
(3) After Python and Pytorch being set, open the "cmd" window, and input "pip install -r requiements.txt"

III. Data selecting and pre-model
(1) The data for the model can be downloaded from [Spider Dataset website](https://yale-lily.github.io/spider). 
(2)  "wikisql_tables.json" needs to be downloaded from https://drive.google.com/file/d/13I_EqnAR4v2aE-CWhJ0XQ8c-UlGS9oic/view?usp=sharing
(3)  "glove.6B.300d.txt" needs to be downloaded from https://nlp.stanford.edu/projects/glove/

IV. Train Model
(1) I have trained the model, and you just need to test this model.
(2) Before each train, you should change the train objective from ["agg", "andor", "column", "desasc", "distinct", "having", "keyword", "limit", "op", "value"].
(3) Before each train, you should change the train objective corresponding the model, like epochs and so on.
(4) Use the operating line "python train.py" or run the Python module "train.py" directly in the IDLE.

V. Test Model
Use the operating line "python test.py" or run the Python module "test.py" directly in the IDLE.
