# Automated Data Analysis Project

This project involves the creation of an automated analysis script that leverages a large language model (LLM) to process a dataset, perform exploratory data analysis, visualize the results, and narrate the insights discovered.

## Project Overview

The Python script - `autolysis.py`, performs the following:

- Accepts a CSV file containing any dataset.
- Performs generic analysis on the dataset, such as summary statistics, correlation matrices, outlier detection, clustering etc.
- Uses an LLM to assist in generating Python code for analysis.
- Creates visualizations (charts) to support the analysis.
- Uses an LLM to provide a narrative summarizing the results.
- Outputs a Markdown file (`README.md`) that includes the analysis summary and narrative.

## Running the Script

To run the script, use the following command in your terminal:

```bash
uv run autolysis.py dataset.csv

