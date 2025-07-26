# This points out how to run the cross site validation i.e leave one out. 

1. Navigate to the directory, Modelling 
`cd Modelling `

2. To run the experiments and get plots. You can directly use the notebook, `cross_site_validation.ipynb`

OR 

3. (more preffered for quick testing) Launch the launch_cross.sh script in the terminal.

a. Create and activate the a virtual environment: 
` cd Modelling`

`python -m venv virtualenv`

`source virtualenv/bin/activate`

# Install the required packages 
`pip install scikit-learn pandas numpy matplotlib xgboost seaborn openpyxl statsmodels hmmlearn`


# Launch the cross-validation python script
- This script performs cross-site validation for generator data.
- It allows for hyperparameter tuning of Random Forest and XGBoost models.
- To run this script, you can use the command line with the following options:
- If you want to run this script with suboptimal hyperparameter tuning, use:

```
python cross_site_validation.py --tune_hyperparams
```

- If you want to run the best model without hyperparameter tuning, use:(Faster execution time)

```
python cross_site_validation.py
```

# To plot the results.

```
python cross_plots.py 
```


