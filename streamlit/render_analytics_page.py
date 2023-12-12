import streamlit as st
import pandas as pd
import snowflake.connector
import plotly.express as px
import os
from dotenv import load_dotenv
from geopy.distance import geodesic
import math
import toml  # Install using pip install toml

# Load Snowflake credentials from .toml file
snowflake_credentials = toml.load("snowflake_credentials.toml")

# Database connection function
def load_data(query):
    conn = snowflake.connector.connect(
        user=snowflake_credentials["snowflake"]["user"],
        password=snowflake_credentials["snowflake"]["password"],
        account=snowflake_credentials["snowflake"]["account"],
        role=snowflake_credentials["snowflake"]["role"],
        warehouse=snowflake_credentials["snowflake"]["warehouse"],
        database=snowflake_credentials["snowflake"]["database"],
        schema=snowflake_credentials["snowflake"]["schema"],
    )
    cur = conn.cursor()
    cur.execute(query)
    df = pd.DataFrame(cur.fetchall(), columns=[col[0] for col in cur.description])
    cur.close()
    conn.close()
    return df

def fetch_distinct_column_values(column_name):
    query = f"SELECT DISTINCT {column_name} FROM BUSINESS LIMIT 10"
    df = load_data(query)
    return df[column_name].tolist()

def clean_and_process_categories(df):
    # Remove newline characters, brackets, and quotes, then split into a list
    df['CATEGORIES'] = df['CATEGORIES'].str.replace(r'[\n\[\]"]', '', regex=True)
    df['CATEGORIES'] = df['CATEGORIES'].apply(lambda x: x.split(", ") if x else [])
    df_exploded = df.explode('CATEGORIES')
    df_exploded['CATEGORIES'] = df_exploded['CATEGORIES'].str.strip()
    print(df_exploded.head(5))
    return df_exploded

# Function for category popularity analysis with filters
def category_popularity_analysis(city=None, postal_code=None):
    # Form the base query
    base_query = "SELECT CITY, POSTAL_CODE, CATEGORIES FROM BUSINESS"

    # Fetch the raw categories data
    df = load_data(base_query)

    print(df.head(10))
    
    # Clean and process the categories
    df_exploded = clean_and_process_categories(df)

    if city:
        df_exploded = df_exploded[df_exploded['CITY'].str.lower() == city.lower()]
        
    # Apply postal code filter if specified
    if postal_code:
        df_exploded = df_exploded[df_exploded['POSTAL_CODE'] == postal_code]
    
    # Count the occurrences of each category
    category_counts = df_exploded['CATEGORIES'].value_counts().reset_index()
    category_counts.columns = ['CATEGORY', 'BUSINESS_COUNT']
    
    # Plot the top categories
    fig = px.bar(category_counts.head(10), x='CATEGORY', y='BUSINESS_COUNT', title='Top 10 Popular Business Categories')
    st.plotly_chart(fig)



# Analytics Functions
def business_density_analysis():
    query = "SELECT CITY, STATE, LATITUDE, LONGITUDE, COUNT(*) AS NUM_BUSINESSES FROM BUSINESS GROUP BY CITY, STATE, LATITUDE, LONGITUDE"
    df = load_data(query)
    fig = px.scatter_mapbox(df, lat='LATITUDE', lon='LONGITUDE', hover_name='CITY', hover_data=['STATE', 'NUM_BUSINESSES'], color_discrete_sequence=["fuchsia"], zoom=5, height=300)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig)


def average_rating_and_review_count_by_city():
    query = """
    SELECT CITY, AVG(STARS) AS AVERAGE_RATING, COUNT(*) AS REVIEW_COUNT
    FROM BUSINESS
    WHERE STATE = 'PA'
    GROUP BY CITY
    ORDER BY AVERAGE_RATING DESC , REVIEW_COUNT DESC
    LIMIT 100
    """
    df = load_data(query)
    fig = px.bar(df, x='CITY', y=['AVERAGE_RATING', 'REVIEW_COUNT'], barmode='group', title='Average Rating and Review Count by City in PA')
    st.plotly_chart(fig)

def top_users_by_review_counts():
    query = """
    SELECT USER_ID, COUNT(*) AS REVIEW_COUNT
    FROM REVIEW
    GROUP BY USER_ID
    ORDER BY REVIEW_COUNT DESC
    LIMIT 10
    """
    df = load_data(query)
    fig = px.bar(df, x='USER_ID', y='REVIEW_COUNT', title='Top Users by Review Counts in PA')
    st.plotly_chart(fig)

def most_checked_in_restaurants():
    query = """
    SELECT BUSINESS_ID, COUNT(*) AS CHECKIN_COUNT
    FROM CHECKINS
    GROUP BY BUSINESS_ID
    ORDER BY CHECKIN_COUNT DESC
    LIMIT 10
    """
    df = load_data(query)
    fig = px.bar(df, x='BUSINESS_ID', y='CHECKIN_COUNT', title='Most Checked-in Restaurants in PA')
    st.plotly_chart(fig)

def correlation_checkins_reviews():
    query = """
    SELECT B.BUSINESS_ID, COUNT(DISTINCT C.DATE) AS CHECKIN_COUNT, COUNT(DISTINCT R.REVIEW_ID) AS REVIEW_COUNT
    FROM BUSINESS B
    LEFT JOIN CHECKINS C ON B.BUSINESS_ID = C.BUSINESS_ID
    LEFT JOIN REVIEW R ON B.BUSINESS_ID = R.BUSINESS_ID
    WHERE B.STATE = 'PA'
    GROUP BY B.BUSINESS_ID
    """
    df = load_data(query)
    fig = px.scatter(df, x='CHECKIN_COUNT', y='REVIEW_COUNT', title='Correlation between Check-ins and Reviews in PA')
    st.plotly_chart(fig)

# Function to perform sentiment analysis
def sentiment_analysis(state='PA'):
    query = f"""
    SELECT STARS, COUNT(*) AS REVIEW_COUNT
    FROM BUSINESS
    WHERE STATE = '{state}'
    GROUP BY STARS
    """
    df = load_data(query)
    
    # Classify reviews based on star ratings
    df['SENTIMENT'] = pd.cut(df['STARS'], bins=[0, 2, 4, 5], labels=['Negative', 'Neutral', 'Positive'], right=False)
    
    sentiment_counts = df.groupby('SENTIMENT')['REVIEW_COUNT'].sum().reset_index()
    fig = px.pie(sentiment_counts, values='REVIEW_COUNT', names='SENTIMENT', title='Sentiment Distribution in PA State')
    st.plotly_chart(fig)

    star_counts = df.groupby('STARS')['REVIEW_COUNT'].sum().reset_index()
    fig_star = px.pie(star_counts, values='REVIEW_COUNT', names='STARS', title='Star Rating Distribution in PA State')
    st.plotly_chart(fig_star)


# Function to find the latitude and longitude for a given zip code from the Snowflake table
def get_lat_lon_from_zip(zip_code):
    query = f"SELECT LATITUDE, LONGITUDE FROM BUSINESS WHERE POSTAL_CODE = '{zip_code}'"
    result = load_data(query)
    if not result.empty:
        return (result.iloc[0]['LATITUDE'], result.iloc[0]['LONGITUDE'])
    else:
        return (None, None)

def calculate_distance(coords_1, coords_2):
    if any(math.isnan(coord) for coord in coords_1) or any(math.isnan(coord) for coord in coords_2):
        return float('nan')  # Return NaN if any of the coordinates are NaN.
    return geodesic(coords_1, coords_2).miles

def perform_search(zip_code, category, radius):
    user_location = get_lat_lon_from_zip(zip_code)

    if None in user_location:
        st.error("Invalid ZIP code or no data available for this ZIP code.")
        return

    query = f"""
    SELECT *
    FROM BUSINESS,
    LATERAL FLATTEN(INPUT => SPLIT(CATEGORIES[0], ', '))
    WHERE ARRAY_CONTAINS('{category}'::VARIANT, VALUE)
    """
    business_df = load_data(query)

    if business_df.empty:
        st.error("No businesses found for the selected category.")
        return

    business_df['distance'] = business_df.apply(
        lambda row: calculate_distance(user_location, (row['LATITUDE'], row['LONGITUDE'])), axis=1
    )

    nearby_businesses = business_df[business_df['distance'] <= radius]
    
    if not nearby_businesses.empty:
        best_rated_business = nearby_businesses.sort_values(by='STARS', ascending=False).head(1)
        st.write(f"Best rated business in {category}:")
        st.dataframe(best_rated_business[['NAME', 'ADDRESS', 'CITY', 'STATE', 'STARS', 'distance']])
    else:
        st.write("No businesses found within the specified radius.")

    # Display results on the map
    st.map(nearby_businesses[['LATITUDE', 'LONGITUDE']])

def find_best_restaurants():
    # Initialize session state variables if they don't exist
    if 'zip_code' not in st.session_state:
        st.session_state.zip_code = '19107'
    if 'radius' not in st.session_state:
        st.session_state.radius = 5
    if 'category' not in st.session_state:
        st.session_state.category = ''

    # Input fields with session state
    st.session_state.zip_code = st.text_input("Enter your ZIP code:", value=st.session_state.zip_code)
    st.session_state.radius = st.number_input("Enter radius in miles:", min_value=1, max_value=50, value=st.session_state.radius)
    st.session_state.category = st.text_input("Enter a category (e.g., Italian, Chinese):", value=st.session_state.category)

    # Button to find restaurants with a callback
    if st.button("Find Restaurants"):
        perform_search(st.session_state.zip_code, st.session_state.category, st.session_state.radius)




# Dashboard Rendering Function
def render_analytics_dashboard():
    st.title("Business Analytics Dashboard")

    # Initialize session state variables for button clicks
    if 'last_clicked' not in st.session_state:
        st.session_state.last_clicked = None

    # Button Definitions with Callbacks
    if st.button("Find Best Restaurants"):
        st.session_state.last_clicked = 'find_best_restaurants'

    if st.session_state.last_clicked == 'find_best_restaurants':
        find_best_restaurants()

    # Initialize a flag to track if Category Popularity Analysis button is clicked
    if 'category_analysis_clicked' not in st.session_state:
        st.session_state['category_analysis_clicked'] = False
    if 'show_category_analysis' not in st.session_state:
        st.session_state['show_category_analysis'] = False
    
    # Button for Business Density Analysis
    if st.button('Business Density Analysis'):
        # Reset the show_category_analysis state and the category_analysis_clicked flag
        st.session_state['show_category_analysis'] = False
        st.session_state['category_analysis_clicked'] = False
        business_density_analysis()

    # Toggle for Category Popularity Analysis
    if st.button('Category Popularity Analysis'):
        st.session_state['show_category_analysis'] = not st.session_state.get('show_category_analysis', False)
        st.session_state['category_analysis_clicked'] = True

    # Display city and postal code options only if the toggle is active
    if st.session_state.get('show_category_analysis', False):
        cities = fetch_distinct_column_values('CITY')
        postal_codes = fetch_distinct_column_values('POSTAL_CODE')
        selected_city = st.selectbox('Select City', [''] + cities, index=0)
        selected_postal_code = st.selectbox('Select Postal Code', [''] + postal_codes, index=0)

        if selected_city or selected_postal_code:
            if st.button('Filter Search'):
                category_popularity_analysis(selected_city, selected_postal_code)
        else:
            if st.session_state['category_analysis_clicked']:
                # Display default chart when no filters are selected
                category_popularity_analysis()

    # If the toggle is off and Category Popularity Analysis was clicked, show the default chart
    elif 'show_category_analysis' in st.session_state and not st.session_state['show_category_analysis']:
        if st.session_state['category_analysis_clicked']:
            category_popularity_analysis()
    # Hide everything if Category Popularity Analysis was not clicked
    else:
        st.session_state['category_analysis_clicked'] = False


    if st.button("Average Rating and Review Count by City"):
        average_rating_and_review_count_by_city()

    if st.button("Top Users by Review Counts"):
        top_users_by_review_counts()

    if st.button("Most Checked-in Restaurants"):
        most_checked_in_restaurants()

    if st.button("Correlation between Check-ins and Reviews"):
        correlation_checkins_reviews()
    
    if st.button("Sentiment Analysis"):
        # Reset the show_category_analysis state and the category_analysis_clicked flag
        st.session_state['show_category_analysis'] = False
        st.session_state['category_analysis_clicked'] = False
        sentiment_analysis()