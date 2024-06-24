"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import time
from Multivalued_Dependencies.mvd_algorithms import analyze_mvd, setup_logging, top_down_algorithm, bottom_up_algorithm, print_mvd_tree
from Multivalued_Dependencies.data_processing import replace_missing_with_mean,convert_to_categorical
from . import app
import logging

@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
    )

@app.route('/results')
def results():
    """Renders the results page."""
    return render_template(
        'results.html',
        title='Results Page',
        year=datetime.now().year,
        message='Mining Results.'
    )

@app.route('/index', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file and file.filename.endswith('.csv'):
        data = pd.read_csv(file)
        name = file.filename 
        
        # Logging mechanism
        setup_logging()
        
        # Data Cleaning
        data = replace_missing_with_mean(data)
        data = convert_to_categorical(data) 

        # Top Down Algorithm
        start_time = time.time()
        top_down_tree = analyze_mvd(data, list(data.columns),True)
        top_down_result = print_mvd_tree(top_down_tree.root)
        top_down_processing_time = time.time() - start_time
        
        # Bottom Up Algorithm
        start_time = time.time()
        bottom_up_tree = analyze_mvd(data, list(data.columns),False)
        bottom_up_result = print_mvd_tree(top_down_tree.root)
        bottom_up_processing_time = time.time() - start_time
        
        # Logging
        logging.info("Data" + str(name))
        logging.info(top_down_result)
        logging.info("topdown processing time - " + str(top_down_processing_time))
        logging.info(bottom_up_result)
        logging.info("bottomup processing time - " + str(bottom_up_processing_time))

        # Render the results template with all results and processing times
        return render_template(
            'results.html',
            top_down_result=top_down_result,
            bottom_up_result=bottom_up_result,
            top_down_time=top_down_processing_time,
            bottom_up_time=bottom_up_processing_time
        )
    return redirect(url_for('home'))  # Redirect to home if file is not valid