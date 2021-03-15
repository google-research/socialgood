from flask import Flask, render_template, request
import pandas as pd
import textwrap
import altair as alt
import io
import logging
import sqlalchemy

app = Flask(__name__)
logger = logging.getLogger()

alt.data_transformers.disable_max_rows() 

db_user = 'root'
db_pass = 'wdKvwgkAb1GLgxvF'
db_name = 'sample_db'
db_connection_name = 'ai4sg-template-1:us-central1:ai4sg-example-db'
engine = sqlalchemy.create_engine(F'mysql+pymysql://{db_user}:{db_pass}@/{db_name}?unix_socket=/cloudsql/{db_connection_name}')

@app.route('/')
def index():
    # Get request dropdown selection if available.
    selected_country = request.args.get('country', 'France')

    # Load data from DB
    sql = 'SELECT country, year, co2 FROM `owid`'
    df_sql = pd.read_sql(sql, engine)
    # Generate dropdown data
    country_dropdown = df_sql.country.unique().tolist()
    # Log something to validate connection to DB
    logger.info(F'df size = {df_sql.shape}')

    # Generate a plot to display
    df_plot = df_sql[df_sql.country == selected_country].copy()
    plot = alt.Chart(df_plot).encode(x='year:Q',y='co2:Q', color=alt.Color('country:N')).mark_line().encode()
    json_str = plot.to_json()
    
    # Display it using a simple html template. Flask uses jinja2 for generating html files used for rendering. 
    return render_template('ai4sg_template.html', dropdown=country_dropdown, json_str=json_str, selected_country=selected_country)


if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)