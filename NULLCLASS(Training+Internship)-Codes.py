#!/usr/bin/env python
# coding: utf-8

# ### NULLCLASS Training Codes (December, 2024)

# In[1]:


# Importing required libraries
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import webbrowser
import os


# In[2]:


# Downloading NLTK's VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')


# ### **Step 1: Data Loading**

# In[3]:


# Reading the datasets
apps=pd.read_csv('Play Store Data.csv')
reviews=pd.read_csv('User Reviews.csv')


# In[4]:


apps.head()


# In[5]:


reviews.head()


# In[6]:


apps.info()


# In[7]:


reviews.info()


# ### **Step 2: Initial Data Inspection**

# In[8]:


# Checking for missing values in the datasets
print("Missing values in 'apps' dataset:")
print(apps.isnull().sum())
print("\nMissing values in 'reviews' dataset:")
print(reviews.isnull().sum())


# ### **Step 3: Data Cleaning**

# In[9]:


# Dropping rows with missing ratings and duplicates
apps= apps.dropna(subset=['Rating'])

# Filling missing values with the mode (most frequent value) for each column
for column in apps.columns :
    
    apps[column].fillna(apps[column].mode() [0], inplace=True)
    
apps.drop_duplicates(inplace=True)

# Removing invalid ratings (greater than 5)
apps=apps[apps['Rating']<=5]

# Dropping rows with missing reviews in the 'reviews' dataset
reviews.dropna(subset=['Translated_Review'], inplace=True)


# In[10]:


#Converting the Installs columns to numeric by removing commas and + 
apps['Installs']=apps['Installs'].str.replace(',','').str.replace('+','').astype(int)

#Converting Price column to numeric after removing $ 
apps['Price']=apps[ 'Price'].str.replace('$','').astype(float)


# In[11]:


apps.info()


# In[12]:


#combining the apps and reviews dataset into one for easier analysis
combined_data=pd.merge(apps,reviews,on="App",how="inner")
combined_data.head()


# In[13]:


# Function to convert 'Size' column to a uniform numeric format (MB)
def convert_size(size):
    if 'M' in size:
        return float(size.replace('M',''))
    elif 'k' in size:
        return float(size.replace('k',''))/1024
    else:
        return np.nan
apps['Size']=apps['Size'].apply(convert_size)
apps


# ### **Step 4: Feature Engineering**

# In[14]:


# Adding log-transformed features for better scaling in visualizations
apps['Log_Installs']=np.log(apps['Installs'])
apps['Reviews']=apps['Reviews'].astype(int)
apps['Log_Reviews']=np.log(apps['Reviews'])


# In[15]:


apps.info()


# In[16]:


# Calculating estimated revenue for apps
apps['Revenue']=apps['Price']*apps['Installs']
apps


# ### **Step 5: Sentiment Analysis**

# In[17]:


#To measure polarity scores of any sentence or analyze the positive/negative intensity of any sentence
sia=SentimentIntensityAnalyzer()


# In[18]:


# Using VADER for sentiment score computation
reviews['Sentiment Score']=reviews['Translated_Review'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
reviews.head()


# In[19]:


# Adding Time-Based Features
# Converting 'Last Updated' to datetime format and extracting the year
apps['Last Updated']=pd.to_datetime(apps['Last Updated'])

apps['Year']=apps['Last Updated'].dt.year
apps


# ### **Step 6: Data Visualization Setup**

# In[20]:


# Directory for saving HTML plots
html_files_path="./"
if not os.path.exists(html_files_path): 
    os.makedirs(html_files_path)


# In[21]:


# Global container for all plots and insights
plot_containers=""


# In[22]:


# Function to save Plotly figures as interactive HTML files

def save_plot_as_html(fig, filename, insight):
    global plot_containers 
    filepath=os.path.join(html_files_path, filename)
    
    # Converting the figure to HTML content
    html_content = pio.to_html(fig, full_html=False, include_plotlyjs='inline')

    # Appending the plot and its insight to plot_containers
    plot_containers += f"""
    <div class="plot-container" id="(filename)" onclick="openPlot('{filename}')">
        <div class="plot">{html_content}</div>
        <div class="insights">{insight}</div>
    </div>
    """
    fig.write_html (filepath, full_html=False, include_plotlyjs='inline')


# In[23]:


# Visualization layout configuration
plot_width=1200
plot_height=570
plot_bg_color='black'
text_color= 'white'
title_font={'size':16}
axis_font={'size':12}


# ### Step 7: Using Plotly to built various types of graphs (demonstration) 

# In[24]:


#Figure 1

category_counts=apps['Category'].value_counts().nlargest(10)

fig1=px.bar(
    x=category_counts.index,
    y=category_counts.values,
    labels={'x': 'Category','y': 'Count'},
    title='Top Categories on Play Store',
    color=category_counts.index,
    color_discrete_sequence=px.colors.sequential.Plasma,
    width=1200,
    height=570
)

fig1.update_layout(

    plot_bgcolor="black",
    paper_bgcolor="black",
    font_color="white",
    title_font={'size':16},
    xaxis= dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10, r=10, t=30, b=10) 
)

#figl.update_traces(marker=dict(pattern=dict(line=dict(color='white', width=1))))
save_plot_as_html (fig1, "Category Graph 1.html", "The top categories on the Play Store are dominated by tools, entertainment, and productivity apps")


# In[25]:


#Figure 2

type_counts=apps['Type'].value_counts()

fig2=px.pie(

    values=type_counts.values,
    names=type_counts.index,
    title='App Type Distribution',
    color_discrete_sequence=px.colors.sequential.RdBu,
    width=1200,
    height=570
)

fig2.update_layout(

    plot_bgcolor="black",
    paper_bgcolor='black',
    font_color= 'white',
    title_font={"size":16},
    margin=dict(l=10, r=10, t=30,b=10)
)

#fig2.update_traces(marker=dict(pattern=dict(line=dict(color='white', width=1))))
save_plot_as_html (fig2, "Type Graph 2.html", "Most apps on the Playstore are free, indicating a strategy to attract users first and monetize through ads or in app purchases")


# In[26]:


#Figure 3

fig3=px.histogram(

    apps,
    x='Rating',
    nbins=20,
    title='Rating Distribution',
    color_discrete_sequence=['#636EFA'],
    width=1200,
    height=570
)

fig3.update_layout(

    plot_bgcolor='black',
    paper_bgcolor="black",
    font_color="white",
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10, r=10, t=30, b=10)

)

#fig3.update_traces (marker=dict(pattern=dict(Line=dict(color='white', width=1))))
save_plot_as_html(fig3, "Rating Graph 3.html", "Ratings are skewed towards higher values, suggesting that most apps are rated favorably by users")


# In[27]:


#Figure4
sentiment_counts=reviews['Sentiment Score'].value_counts()

fig4=px.bar(
    x=sentiment_counts.index,
    y=sentiment_counts.values,
    labels={'x': "Sentiment Score", 'y': "Count"},
    title='Sentiment Distribution',
    color=sentiment_counts.index,
    color_discrete_sequence=px.colors.sequential.RdPu,
    width=1200,
    height=570

)

fig4.update_layout(

    plot_bgcolor="black",
    paper_bgcolor="black",
    font_color="white",
    title_font={"size":16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10, r=10, t=30,b=10)
)

#fig4.update traces (marker-dict(pattern-dict(line-dict(color="white", width-1))))
save_plot_as_html(fig4, "Sentiment Graph 4.html", "Sentiments in reviews show a mix of positive and negative feedback, with a slight lean towards positive sentiments")


# In[28]:


#Figure 5

installs_by_category=apps.groupby('Category') ['Installs'].sum().nlargest(10)

fig5=px.bar(

    x=installs_by_category.index,
    y=installs_by_category.values,
    orientation="h",
    labels={'x': 'Installs','y': 'Category'},
    title='Installs by Category',
    color=installs_by_category.index,
    color_discrete_sequence=px.colors.sequential.Blues,
    width=1200,
    height=570
)

fig5.update_layout(

    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color="white",
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10, r=10, t=30, b=10)
)

#fig5.update_traces (marker-dict(pattern-dict(Line-dict(color='white', width=1))))
save_plot_as_html(fig5, "Installs Graph 5.html", "The categories with the most installs are social and communication apps, reflecting their broad appeal and daily usgae")


# In[29]:


# Updates Per Year Plot

updates_per_year = apps[ 'Last Updated'].dt.year.value_counts().sort_index()

fig6=px.line(

    x=updates_per_year.index,
    y=updates_per_year.values, labels={'x': 'Year', 'y': 'Number of Updates'},
    title='Number of Updates Over the Years',
    color_discrete_sequence=['#AB63FA'],
    width=1200,
    height=570
)

fig6.update_layout(

    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)

)

save_plot_as_html(fig6, "updates_per_year 6.html", "Updates have been increasing over the years, showing that developers are actively maintaining and improving their apps")


# In[30]:


#Figure 7

revenue_by_category=apps.groupby('Category') ['Revenue'].sum().nlargest (10)

fig7=px.bar(

    x=installs_by_category.index,
    y=installs_by_category.values,
    labels={'x': 'Category','y': 'Revenue'},
    title="Revenue by Category",
    color= installs_by_category.index,
    color_discrete_sequence=px.colors.sequential.Greens,
    width=1200,
    height=570
)

fig7.update_layout(

    plot_bgcolor="black",
    paper_bgcolor="black",
    font_color="white",
    title_font={'size':16},
    xaxis=dict(title_font={"size":12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict( l = 18 ,r = 10 ,t = 3 , b = 10 )

)
#fig.7update_traces (marker=dict(pattern=dict(line=dict(color="white", width=1))))

save_plot_as_html(fig7, "Revenue Graph 7.html", "Categories such as Business and Productivity lead in revenue generation, indicating their monetization potential")


# In[31]:


#Figure 8

genre_counts=apps['Genres'].str.split(';', expand=True).stack().value_counts().nlargest(10)

fig8=px.bar(

    x=genre_counts.index,
    y=genre_counts.values,
    labels={'x': 'Genre','y': 'Count'},
    title='Top Genres',
    color=installs_by_category.index,
    color_discrete_sequence=px.colors.sequential.OrRd,
    width=1200,
    height=570
)

fig8.update_layout(

    plot_bgcolor='black',
    paper_bgcolor="black",
    font_color="white",
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}), 
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)
                                                        

#fig8.update_traces(marker=dict(pattern=dict(Line=dict(color="white", width-1))))
save_plot_as_html(fig8,"Genre Graph 8.html", "Action and Casual genres are the most common, reflecting users' preference for engaging and easy-to-play games")


# In[32]:


#Figure 9

fig9=px.scatter(

    apps,
    x= 'Last Updated',
    y='Rating',
    color='Type',
    title='Impact of Last Update on Rating',
    color_discrete_sequence=px.colors.qualitative.Vivid,
    width=1200,
    height=570
)

fig9.update_layout(

    plot_bgcolor='black',
    paper_bgcolor="black",
    font_color="white",
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10, r=10, t=30,b=10)

)

#fig9.update_traces (marker=dict(pattern=dict(line=dict(color="white", width=1))))
save_plot_as_html(fig9, "Update Graph 9.html", "The Scatter Plot shows a weak correlation between the last update and ratings, suggesting that more frequent updates dont always result in better ratings")


# In[33]:


#Figure 10

fig10=px.box(

    apps,
    x='Type',
    y='Rating',
    color='Type',
    title='Rating for Paid vs Free Apps',
    color_discrete_sequence=px.colors.qualitative.Pastel,
    width=1200,
    height=570
)

fig10.update_layout (

    plot_bgcolor='black',
    paper_bgcolor="black",
    font_color="white",
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10, r=10,t=30,b=10)

)

#fig10.update_traces (marker-dict(pattern-dict(line-dict(color-'white', width-1))))
save_plot_as_html (fig10, "Paid Free Graph 10.html", "Paid apps generally have higher ratings compared to free apps, suggesting that users expect higher quality from apps they pay for")


# In[34]:


#Container splitting 
plot_containers_split=plot_containers.split('</div>')

if len(plot_containers_split) > 1:

    final_plot=plot_containers_split[-2]+'</div>'

else:

    final_plot=plot_containers


# ### **Step 8: Generating the Dashboard**

# In[35]:


dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Google Play Store Review Analytics</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background-color: #333;
            color: #fff;
            margin: 0;
            padding: 0;
        }}
        .header {{
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            background-color: #444;
        }}
        .header img {{
            margin: 0 10px;
            height: 50px;
        }}
        .container {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            padding: 20px;
        }}
        .plot-container {{
            border: 2px solid #555;
            margin: 10px;
            padding: 10px;
            width: {plot_width}px;
            height: {plot_height}px;
            overflow: hidden;
            position: relative;
            cursor: pointer;
        }}
        .insights {{
            display: none;
            position: absolute;
            right: 10px;
            top: 10px;
            background-color: rgba(0,0,0,0.7);
            padding: 5px;
            border-radius: 5px;
            color: #fff;
        }}
        .plot-container:hover .insights {{
            display: block;
        }}
    </style>
    <script>
        function openPlot(filename) {{
            window.open(filename, '_blank');
        }}
    </script>
</head>
<body>
    <div class="header">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Logo_2013_Google.png/800px-Logo_2013_Google.png" alt="Google Logo">
        <h1>Google Play Store Reviews Analytics</h1>
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/78/Google_Play_Store_badge_EN.svg/1024px-Google_Play_Store_badge_EN.svg.png">
    </div>
    <div class="container">
        {plots}
    </div>
</body>
</html>
"""


# In[36]:


# Formatting and saving the dashboard as an HTML file
final_html=dashboard_html.format(plots=plot_containers, plot_width=plot_width, plot_height=plot_height)

dashboard_path=os.path.join(html_files_path, "web page.html")

with open(dashboard_path, "w", encoding="utf-8") as f:
    f.write(final_html)

# Opening the dashboard in a browser
webbrowser.open('file://'+os.path.realpath(dashboard_path))


# # -----------------------------------------------------------------------------------------------------------
# 
# ### NULLCLASS INTERNSHIP TASKS (Jan, 2025-June, 2025)

# #### Task 1: Visualisation of Sentiment Distribution

# Goal: Using a stacked bar chart to show the sentiment distribution (positive, neutral, and negative) of user reviews by rating group (e.g., 1-2 stars, 3-4 stars, and 4-5 stars).
# 
# 1. Only apps with over 1,000 reviews should be included.
# 2. Sort the data into the top five groups.

# In[37]:


import plotly.graph_objects as go

# Step 1: Filtering the apps with more than 1,000 reviews
apps_filtered = apps[apps['Reviews'].astype(int) > 1000]
apps_filtered


# In[38]:


# Step 2: Selecting the top 5 categories based on the number of apps
top_cat = apps_filtered['Category'].value_counts().head(5).index
top_cat


# In[39]:


apps_top_cat = apps_filtered[apps_filtered['Category'].isin(top_cat)]


# In[40]:


# Step 3: Merging the filtered app data and original review data
merged_1 = pd.merge(apps_top_cat, reviews, on="App")
merged_1


# In[41]:


# Step 4: Segmenting ratings into groups
def classify_rating_group(rating):
    if 1 <= rating <= 2:
        return "1-2 stars"
    elif 3 <= rating <= 4:
        return "3-4 stars"
    elif 4 < rating <= 5:
        return "4-5 stars"

merged_1['Rating Group'] = merged_1['Rating'].apply(classify_rating_group)


# In[42]:


# Counting the number of values in each rating group
rating_group_counts = merged_1['Rating Group'].value_counts()
print(rating_group_counts)


# In[43]:


# Step 5: generating the data for stacked bar chart
sentiment = (
    merged_1.groupby(['Category', 'Rating Group', 'Sentiment'])['Sentiment'].count().unstack(fill_value=0).reset_index() )


# In[44]:


# Step 6: Creating the stacked bar chart
fig_1 = go.Figure()

for s in ['Positive', 'Neutral', 'Negative']:
    fig_1.add_trace(go.Bar(
        x=sentiment['Rating Group'],
        y=sentiment[s],
        name=s,
        marker=dict(color={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'}[s]) ))


# In[45]:


# Step 7: Customizing the chart
fig_1.update_layout(
    title="Sentiment Distribution by Rating Group (Top 5 Categories)",
    xaxis=dict(title="Rating Group"),
    yaxis=dict(title="Number of Reviews"),
    barmode='stack',
    legend_title="Sentiments",
    plot_bgcolor='black',
    paper_bgcolor="black",
    font_color="white",
    
)

# Step 8: Saving the chart as an HTML file
fig_1.write_html("sentiment_distribution_dashboard_T1.html")

# Step 10: To Open the dashboard in the browser
import webbrowser
webbrowser.open("sentiment_distribution_dashboard_T1.html")


# #### Task 2: Creating a Word Cloud for 5-Star Reviews 
# 
# Goal: To create an visually appealing word cloud of the most common keywords in 5-star reviews, omitting:
# 
# 1. Frequently used stopwords, such as "and," "the", etc
# 2. app names to guarantee applicability.
# 3. Furthermore, to restrict the reviews to only those from applications that fall under the "Health & Fitness" theme.

# In[46]:


# Step 1: Filtering the reviews for the "HEALTH_AND_FITNESS" category and 5 star ratings

health_rev = combined_data[
    (combined_data['Category'] == 'HEALTH_AND_FITNESS') &  
    (combined_data['Rating'] >= 5) &                      
    (combined_data['Translated_Review'].notnull())   
]

health_rev


# #### Above result shows that there are no records for exact 5 star rating in user reviews dataset.

# In[47]:


pip install wordcloud


# In[48]:


# Step 2: Combining all reviews into a single text block for word cloud generation
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Defining additional stopwords (common stopwords and app names to be excluded)
custom_stopwords = set(STOPWORDS)
custom_stopwords.update(["app", "apps", "use", "using", "good", "great", "fitness", "health", "work", "excellent", "well"])

# Combining all reviews into a single text string
all_reviews = " ".join(health_rev['Translated_Review'].dropna())


# In[49]:


# Step 3: Generating the word cloud
wc = WordCloud(
    width=800, 
    height=400, 
    background_color='white', 
    colormap='viridis', 
    max_words=200
).generate(all_reviews)


# #### Above error result shows that there are no exact 5 star ratings present for reviews. So, the WordCloud generation is not possible

# In[51]:


print(combined_data[combined_data['Category'] == 'HEALTH_AND_FITNESS']['Rating'].value_counts())


# ### For Task-2, exact 5 star rating is not available in the dataset. So, WordCloud is not generated.

# #### Task 3: Revenue versus. Installs Scatter Plot Visualisation for Paid Apps
# 
# Goal: To colour-code the points according to app types and add a trendline to show the association between revenue and installs for paid applications only.
# 
# Steps:
# 1. First filter the apps data for paid apps.
# 2. then, ensure numerical data types for proper visualization.
# 3. Creating an interactive scatter plot with Plotly.
# 4. Adding a trendline to show correlation.
# 5. Saving the plot as an HTML file for web visualization as did earlier.

# In[52]:


# Step 1: Filtering the apps data for paid apps only
paid_apps = apps[apps['Type'] == "Paid"]
paid_apps


# In[53]:


#Step 2: Ensuring revenue and installs are numeric for proper visualization
paid_apps['Revenue'] = pd.to_numeric(paid_apps['Revenue'], errors='coerce')
paid_apps['Installs'] = pd.to_numeric(paid_apps['Installs'], errors='coerce')


# In[54]:


# Step 3: Adding a trendline (using numpy for computation)
x = paid_apps['Installs']
y = paid_apps['Revenue']
coeff = np.polyfit(x, y, 1)  
tl = coeff[0] * x + coeff[1]


# In[55]:


# Step 4: Creating an interactive scatter plot based on App Categories with Plotly
fig_11 = px.scatter(
    paid_apps,
    x='Installs',
    y='Revenue',
    color='Category',
    title="Relationship Between Revenue and Installs for Paid Apps",
    labels={"Installs": "Number of Installs", "Revenue": "Revenue (in $)"},
    hover_data=['App'],
    opacity=0.8,  
    template='plotly_dark' 
)


# In[56]:


# Step 5: Adding the trendline to the plot
fig_11.add_scatter(
    x=paid_apps['Installs'], 
    y=tl, 
    mode='lines', 
    name='Trendline',
    line=dict(color='yellow', width=2)
)

# Step 6: Customing the layout
fig_11.update_layout(
    title="Relationship Between Revenue and Installs for Paid Apps",
    xaxis=dict(title="Number of Installs", gridcolor='gray'),
    yaxis=dict(title="Revenue (in $)", gridcolor='gray'),
    legend_title="App Categories",
    plot_bgcolor='black',
    paper_bgcolor="black",
    font_color="white",
    title_font=dict(size=18, color="white"),
    xaxis_title_font=dict(size=14),
    yaxis_title_font=dict(size=14),
    legend_title_font=dict(size=14),
    legend=dict(font=dict(size=12)) 
)

# Step 7: Saving the interactive plot to an HTML file
fig_11.write_html("scatter_revenue_vs_installs_paid_apps_T3.html")

# Step 8: To Open the dashboard in the browser
import webbrowser
webbrowser.open("scatter_revenue_vs_installs_paid_apps_T3.html")


# #### Task 4: Global Installs by Category Interactive Choropleth Map
# Goal: To produce an interactive choropleth map that shows worldwide installs by category under the following circumstances:
# 
# 1. Display only the statistics for the top five app categories according to the number of installs.
# 2. Remove app categories where there are fewer than one million installs and the initial characters of app categories are A, C, G, or S.
# 3. Verify that the graph is only visible from 6 to 8 p.m. IST.
# 

# In[57]:


from datetime import datetime, time

# Filtering top 5 categories based on total installs 
top_cat = apps.groupby('Category')['Installs'].sum().nlargest(5).index

# Taking out those top data
filtered_data_1 = apps[apps['Category'].isin(top_cat)]
filtered_data_1


# In[58]:


# Filtering further where there are fewer than one million installs and the initial characters of app categories are A, C, G, or S.
filtered_data = filtered_data_1[
    (filtered_data_1['Installs'] > 1_000_000) & (~filtered_data_1['Category'].str.startswith(('A', 'C', 'G', 'S')))]  # '~' denotes no to include those particular data

filtered_data


# ### Now, for the px.choropleth function to work, it is necessary to have a Country column (or equivalent geographical data) in order to make it run and work.
# 1. The apps dataset does not contain any column named Country. To create the choropleth map, we need to introduce a column that specifies countries corresponding to the data.
# 2. To solve which, we are adding/generating some random country data based on real world Google Play Store usage in top countries.

# In[141]:


#To guarantee that the graph is displayed only between 6 and 8 PM IST, logic is being added. 
#To get the current time and conditionally display the map, we are useing Python's datetime function.

start_time = time(18, 0) 
end_time = time(20, 0)


# In[142]:


# Getting the current time in IST
current_time_utc = datetime.utcnow()
current_time_ist = (current_time_utc + pd.Timedelta(hours=5, minutes=30)).time()


# In[143]:


# Listing the top countries where Google Play Store is used these days
filtered_data['Country'] = np.random.choice(
    ["United States", "India", "Germany", "Brazil", "Australia", "Indonesia", "Russia",
    "China", "Canada", "United Kingdom", "France", "Japan", "South Africa", "Mexico",
    "Italy", "South Korea", "Spain", "Turkey", "Argentina", "Saudi Arabia", "Netherlands",
    "Sweden", "Norway", "Poland", "Thailand", "Philippines", "Vietnam", "Pakistan",
    "New Zealand", "Egypt", "Malaysia", "Bangladesh", "Nigeria", "Kenya", "Chile",
    "Colombia", "Venezuela", "Peru", "Portugal", "Ireland", "Greece", "Switzerland",
    "Denmark", "Finland", "Belgium", "Austria", "Czech Republic", "Hungary", "Romania",
    "Ukraine", "Slovakia", "Bulgaria", "Croatia", "Slovenia", "Serbia", "Kazakhstan",
    "Uzbekistan", "Morocco", "Algeria", "Tunisia", "Angola", "Ethiopia", "Tanzania",
    "Uganda", "Zambia", "Zimbabwe", "Mozambique", "Botswana", "Ghana", "Ivory Coast",
    "Cameroon", "Senegal", "Cuba", "Jamaica", "Haiti", "Dominican Republic", "Panama",
    "Ecuador", "Bolivia", "Paraguay", "Uruguay", "Qatar", "United Arab Emirates",
    "Oman", "Kuwait", "Jordan", "Lebanon", "Sri Lanka", "Nepal", "Maldives", "Iraq",
    "Iran", "Afghanistan"], size=len(filtered_data)
)
filtered_data


# In[144]:


# Only proceeding with generating the graph if the time is within the specified range
if start_time <= current_time_ist <= end_time:
    # Creating the choropleth map
    fig_12 = px.choropleth(
        filtered_data,
        locations='Country',  # Column containing country names
        locationmode='country names',  # To match country names with Plotly's geographical embeddings.
        color='Installs',  # Coloring by the number of installs
        hover_name='Category',  # Showing app category on hover
        title="Global Installs by App Categories (Filtered)",
        color_continuous_scale='Viridis'
    )
    
    # Saving the map as an HTML file
    fig_12.write_html("interactive_choropleth_map_T4.html")

    # Opening the HTML file in the browser
    webbrowser.open("interactive_choropleth_map_T4.html")
else:
    print("Graph is not available outside the time range (6 PM - 8 PM IST).")


# #### Task 5: Time-Limited Dual-Axis Chart for Average Installs vs Revenue (Free vs Paid Apps)
# 
# Goal: To filter data according to-
# 
# 1. Installations: Over 10,000.
# 2. Revenue: Over $10,000.
# 3. Version of Android: Over 4.0.
# 4. Dimensions: Over 15 meters.
# 5. The content should be rated as "Everyone."
# 6. The length of the app name, including spaces and special characters, should not be more than 30 characters.
# 
# Within the top three app categories, we have to compare the average income and installs of free versus premium apps.
# 
# To make a dual-axis chart with a line for installs and a bar for revenue.
# 
# Making sure the chart only functions from 1:00 PM to 2:00 PM IST. The chart shouldn't show up on the dashboard outside of this time frame.

# In[63]:


import plotly.graph_objects as go

# -------------------- Data Filtering --------------------
# Filter 1: Apps with installs > 10,000
filtered_apps = apps[apps['Installs'] > 10_000]
filtered_apps


# In[64]:


filtered_apps.info()


# In[65]:


# Filter 2: Revenue > $10,000
filtered_apps = filtered_apps[filtered_apps['Revenue'] > 10_000]
filtered_apps


# #### As we can see from the dataset, the current existing version column contains the float values as well as string values. So before applying filter on it, we need to convert the string value into pure float values.

# In[66]:


# Converting the 'Android Ver' column to numeric by extracting the numeric part
filtered_apps['Android Ver'] = filtered_apps['Android Ver'].str.extract(r'(\d+(\.\d+)?)')[0]


# In[67]:


# Converting to numeric type
filtered_apps['Android Ver'] = pd.to_numeric(filtered_apps['Android Ver'], errors='coerce')


# In[68]:


# Filter 3: Android version > 4.0
filtered_apps = filtered_apps[filtered_apps['Android Ver'] > 4.0]
filtered_apps


# In[69]:


# Filter 4: Size > 15M
filtered_apps = filtered_apps[filtered_apps['Size'] > 15.0]

# Filter 5: Content Rating == "Everyone"
filtered_apps = filtered_apps[filtered_apps['Content Rating'] == 'Everyone']
filtered_apps


# In[70]:


# Filter 6: App name length <= 30 characters (including spaces and special characters)
filtered_apps = filtered_apps[filtered_apps['App'].str.len() <=30]
filtered_apps


# In[71]:


# -------------------- Selecting Top 3 Categories --------------------
top_categories = (
    filtered_apps.groupby('Category')['Installs']
    .sum()
    .nlargest(3)
    .index
)
filtered_apps = filtered_apps[filtered_apps['Category'].isin(top_categories)]


# In[72]:


# -------------------- Computing Averages --------------------
# Calculating average installs and revenue for free and paid apps
comparison_data = filtered_apps.groupby(['Category', 'Type'])[['Installs', 'Revenue']].mean().reset_index()


# In[121]:


# -------------------- Creating the Dual-Axis Chart --------------------
# Ensuring that the chart is displayed only between 1 PM and 2 PM IST
start_time = time(13, 0)  
end_time = time(14, 0)

# Getting current time in IST
current_time_utc = datetime.utcnow()
current_time_ist = (current_time_utc + pd.Timedelta(hours=5, minutes=30)).time()


# In[122]:


if start_time <= current_time_ist <= end_time:
    fig_13= go.Figure()

    # Adding bar traces for average revenue
    for c in top_categories:
        fig_13.add_trace(
            go.Bar(
                x=comparison_data[comparison_data['Category'] == c]['Type'],
                y=comparison_data[comparison_data['Category'] == c]['Revenue'],
                name=f"{c} Revenue",
                yaxis="y1",  # Mapping to the first y-axis
                text=comparison_data[comparison_data['Category'] == c]['Revenue'],
                textposition='auto'
            )
        )
    
    # Adding line traces for average installs
    for c1 in top_categories:
        fig_13.add_trace(
            go.Scatter(
                x=comparison_data[comparison_data['Category'] ==c1]['Type'],
                y=comparison_data[comparison_data['Category'] == c1]['Installs'],
                name=f"{c1} Installs",
                yaxis="y2", # Mapping to the second y-axis
                mode="lines+markers"
            )
        )

    # Layout configuration
    fig_13.update_layout(
        title="Average Installs vs Revenue for Free vs Paid Apps (Top 3 Categories)",
        xaxis=dict(title="App Type (Free/Paid)"),
        yaxis=dict(
            title="Average Revenue (in USD)",
            titlefont=dict(color="blue"),
            tickfont=dict(color="blue"),
        ),
        yaxis2=dict(
            title="Average Installs (in Millions)",
            titlefont=dict(color="green"),
            tickfont=dict(color="green"),
            overlaying="y",
            side="right",
        ),
        legend=dict(x=0.1, y=1.1),
        barmode="group"
    )
    # Saving chart as HTML file
    fig_13.write_html("dual_axis_chart_T5.html")

# Opening the chart in the browser
    webbrowser.open("dual_axis_chart_T5.html")
else:
    print("Chart is not available outside the time range (1 PM - 2 PM IST).")


# #### Task-6: Grouped bar chart to compare the average rating and total review count for the top 10 app categories
# By filtering apps based on the following conditions:
# 1. Average rating >= 4.0
# 2. Size >= 10 MB
# 3. Last update should be in January

# ### The Last Updated column in the dataset does not contain string values. Instead, it may contain datetime objects or other data types.
# ### For the 3rd condition to be applicable, the data-type of the 'Last-Updated' column should be changed from datetime to string type.  
# 

# In[75]:


print(apps['Last Updated'].dtype)


# In[76]:


# Adding a extra "Month" column extracting from "Last Updated" column for filtering according to the task
apps['Month'] = apps['Last Updated'].dt.month


# In[77]:


# Applying filters
filtered_apps_T6 = apps[
    (apps['Rating'] >= 4.0) & (apps['Size'] >= 10.0) & (apps['Month'] == 1)
]
filtered_apps_T6


# In[78]:


# Getting the top 10 categories by number of installs
top_cat_T6= (
    filtered_apps_T6.groupby('Category')['Installs']
    .sum()
    .nlargest(10)
    .index
)
top_cat_T6


# In[79]:


# Now that we have top 10 categories based on the no. of Installs done, moving with filtering the apps based on it.
filtered_apps_T6 = filtered_apps_T6[filtered_apps_T6['Category'].isin(top_cat_T6)]
filtered_apps_T6


# In[80]:


# Calculating average rating and total review counts for each category
comp_T6 = (
    filtered_apps_T6.groupby('Category')[['Rating', 'Reviews']]
    .agg({'Rating': 'mean', 'Reviews': 'sum'})
    .reset_index()
)
comp_T6


# In[125]:


# -------------------- Now creating the Grouped Bar Chart --------------------
# Ensuring that the chart is displayed only between 3 PM and 5 PM IST
start_T6 = time(15, 0)  
end_T6 = time(17, 0)

current_time_T6 = datetime.utcnow()
current_time_T6 = (current_time_T6+ pd.Timedelta(hours=5, minutes=30)).time()


# In[126]:


if start_T6  <= current_time_T6  <= end_T6:
    
    fig_14 = go.Figure()

    # Adding bar trace for average rating
    fig_14.add_trace(
        go.Bar(
            x=comp_T6['Category'],
            y=comp_T6['Rating'],
            name="Average Rating",
            text=comp_T6['Rating'],
            textposition='auto',
            marker_color='purple',
            opacity=0.7
        )
    )

    # Adding bar trace for total review counts
    fig_14.add_trace(
        go.Bar(
            x=comp_T6['Category'],
            y=comp_T6['Reviews'],
            name="Total Review Count",
            text=comp_T6['Reviews'],
            textposition='auto',
            marker_color='orange',
            opacity=0.7
        )
    )

    fig_14.update_layout(
        title="Comparison of Average Rating and Total Review Counts (Top 10 Categories by Installs)",
        xaxis=dict(title="App Category", tickangle=-45),
        yaxis=dict(title="Average Rating"),
        yaxis2=dict(
            title="Total Review Count",
            overlaying='y',
            side='right'
        ),
        legend=dict(x=0.1, y=1.1),
        barmode="group"
    )

    # Saving chart as HTML file
    fig_14.write_html("grouped_bar_chart_T6.html")

    # Opening the chart in the browser
    webbrowser.open("grouped_bar_chart_T6.html")
else:
    print("Chart is not available outside the time range (3 PM - 5 PM IST).")


# #### Task 7: Correlation with Time Restrictions Key App Metrics Heatmap
# 
# Step 1: To filter the data by:
# 
# 1. Including only apps and updating within the last year.
# 2. Having at least 100,000 installs.
# 3. Having more than 1,000 reviews.
# 4. Belonging to genres not starting with the characters A, F, E, G, I, or K.
# 
# Step 2: Using the above filtered data to compute the correlation matrix for installs, ratings, and review counts.
# Step 3: Generating the Heatmap by:
# 
# 1. Visualizing the correlation matrix
# 2. Set time conditions to ensure the heatmap is displayed only between 2 PM IST and 4 PM IST.

# ### To include the apps which are only updated within the last year, we need to find the "Latest Date" in the 'Last Updated' column of apps dataset because -
# 
# 1. If we use - "one_year_ago = current_date - pd.DateOffset(year=1) ; latest_apps = df[df['Last Updated'] >= one_year_ago]" , then it shows empty rows, which typically means there are no app records in the last year according to the current ongoing year.
# 2. We need to ensure the filter logic for "updated within the last year" considers the dataset's latest date as the reference instead of the current system date.

# In[83]:


# Checking the age of apps data
print("Earliest date in 'Last Updated':", apps['Last Updated'].min())
print("Latest date in 'Last Updated':", apps['Last Updated'].max())


# In[84]:


# Now defining the initial reference date (latest date as given in apps dataset)
latest_date = pd.to_datetime('2018-08-08')
one_year_ago = latest_date - pd.DateOffset(years=1)


# In[85]:


# Filtering the data
filtered_data_T7 = apps[
    (apps['Last Updated'] >= one_year_ago) &  # Updated within the last year
    (apps['Installs'] >= 100000) &           # At least 100,000 installs
    (apps['Reviews'] > 1000) &         # Reviews count > 1k
    (~apps['Genres'].str.startswith(tuple("AFEGIK"), na=False))  # Genres filter
]
filtered_data_T7


# In[86]:


# Computing the correlation matrix
corr_matrix= filtered_data_T7[['Installs', 'Rating', 'Reviews']].corr()
corr_matrix


# In[123]:


# Generating the heatmap only during 2 PM to 4 PM IST
from datetime import datetime, timedelta, timezone

# Getting the current time in IST
curr_time = datetime.now(timezone(timedelta(hours=5, minutes=30)))
start = curr_time.replace(hour=14, minute=0, second=0, microsecond=0)
end = curr_time.replace(hour=16, minute=0, second=0, microsecond=0)


# In[124]:


# Checking if within the time range
if start<= curr_time <= end:
    # Generating the heatmap
    fig_15 = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='Viridis',
            colorbar=dict(title="Correlation"),
        )
    )

    # Update layout
    fig_15.update_layout(
        title="Correlation Matrix (Installs, Ratings, and Reviews Count)-Task-7",
        xaxis=dict(title="Metrics"),
        yaxis=dict(title="Metrics"),
        width=800,
        height=600,
    )

    # Saving the heatmap as an HTML file
    heatmap_filename = "correlation_heatmap_T7.html"
    fig_15.write_html(heatmap_filename)

    # Opening the HTML file in the default web browser
    webbrowser.open(heatmap_filename)
else:
    print("Heatmap is not available outside the time range (2 PM - 4 PM IST).")


# #### Task 8: Violin Plot Visualisation
# 
# Goal:
# To obtain insights into category-specific rating patterns for apps that satisfy predetermined criteria, use a violin plot to visualise the distribution of ratings across several app categories, concentrating on particular filtered data.
# 
# Objectives:
# 
# 1. To analyze the distribution of ratings for app categories with significant representation (i.e. more than 50 apps).
# 2. To filter on apps with ratings below 4.0 to identify patterns in underperforming apps.
# 3. To exclude apps with fewer than 10 reviews by considering only apps with sufficient user feedback.
# 4. To restrict the data to contain apps whose names contain the letter "C",
# 5. To enable the graph only between 4 PM IST to 6 PM IST.
# 

# In[127]:


# Ensuring 'Reviews Count' and 'Rating' columns are numeric only.
apps['Reviews'] = pd.to_numeric(apps['Reviews'], errors='coerce')
apps['Rating'] = pd.to_numeric(apps['Rating'], errors='coerce')
apps.info()


# In[128]:


# Filter 1: Categories with more than 50 apps
cat_cnt = apps['Category'].value_counts()
cat_cnt


# In[129]:


cat_50 = cat_cnt[cat_cnt > 50].index
cat_50.value_counts()


# In[130]:


filtered_data_T8= apps[apps['Category'].isin(cat_50)]
filtered_data_T8


# In[131]:


# Filter 2: App names containing the letter "C/c"
filtered_data_T8 = filtered_data_T8[filtered_data_T8['App'].str.contains('C', case=False, na=False)] # case=False denotes the case insensitivity of letter
filtered_data_T8


# In[132]:


# Filter 3: Excluding the apps with < 10 reviews
filtered_data_T8 = filtered_data_T8[filtered_data_T8['Reviews'] >= 10]
filtered_data_T8


# In[133]:


# Filter 4: Including the apps with a rating < 4.0 only
filtered_data_T8 = filtered_data_T8[filtered_data_T8['Rating'] < 4.0]
filtered_data_T8


# In[134]:


# Getting the current time in IST
current_time = datetime.now(timezone(timedelta(hours=5, minutes=30)))
start_time = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
end_time = current_time.replace(hour=18, minute=0, second=0, microsecond=0)


# In[136]:


# Generating and displaying the violin plot only if within the specified time range
if start_time <= current_time <= end_time:
    # Creating the violin plot
    fig_16= px.violin(
        filtered_data_T8,
        x='Category',
        y='Rating',
        box=True,  # Adding a box plot within the violin plot
        points="all",  
        title="Distribution of Ratings by App Category (Filtered Data)- Task-8",
        labels={'Rating': 'App Ratings', 'Category': 'App Category'},
        color_discrete_sequence=["#636EFA"],  
    )

    # Customizing the layout
    fig_16.update_layout(
        xaxis_title="App Category",
        yaxis_title="App Ratings",
        xaxis=dict(tickangle=-45),
        height=600,
        width=900,
    )

    # Save the plot as an HTML file
    plot_filename = "violin_plot_T8.html"
    fig_16.write_html(plot_filename)

    # Opening the plot in the default web browser
    webbrowser.open(plot_filename)
else:
    print("Violin plot is not available outside the time range (4 PM - 6 PM IST).")


# #### Task 9: Bubble Chart Creation
# 
# Goal: To show a bubble chart and finding out the connection between average ratings and app size.
# 
# To add the apps only if they fulfil these below requirements:
# 1. Rating greater than a 3.5.
# 2. the apps should be of the "Games" category.
# 3. have over 50,000 installations.
# 4. to restrict this chart's display to the hours of 5 PM IST to 7 PM IST. The chart won't show up on the dashboard if it is viewed outside of this window.
# 

# In[98]:


apps.info()


# In[99]:


# Filter 1: Rating > 3.5
filtered_data_T9 = apps[apps['Rating'] > 3.5]
filtered_data_T9


# In[100]:


#Listing all the castegories present along with their counts before applying filter-2

# Count of apps in each category (original apps dataset)
category_counts_original = apps['Category'].value_counts()
print("App counts per category in the original dataset:")
print(category_counts_original)

# Count of apps in each category (after applying filter-1)
category_counts_filtered = filtered_data_T9['Category'].value_counts()
print("\nApp counts per category in the filtered dataset after Filter 1:")
print(category_counts_filtered)


# In[101]:


# Filter 2: Category is "GAME"
filtered_data_T9 = filtered_data_T9[filtered_data_T9['Category'] == 'GAME']
filtered_data_T9 


# In[102]:


# Filter 3: Installs > 50,000
filtered_data_T9 = filtered_data_T9 [filtered_data_T9 ['Installs'] > 50000]
filtered_data_T9


# In[137]:


# Time-based Display
start_time = time(17, 0) 
end_time = time(19, 0)   
current_time = datetime.now().time()


# In[139]:


# Displaying Bubble Chart only within that above time range
if start_time <= current_time <= end_time:
    #Creating the Bubble Chart
    fig_17 = px.scatter(
        filtered_data_T9,
        x='Size',
        y='Rating',
        size='Installs',
        color='Installs',
        hover_name='App',
        title="Bubble Chart: Relationship between App Size and Ratings (Games Category)",
        labels={'Size_MB': 'App Size (MB)', 'Rating': 'Average Rating', 'Installs': 'Number of Installs'},
    )

    # Updating the layout for better visualization
    fig_17.update_layout(
        xaxis_title="App Size (in MB)",
        yaxis_title="Average Rating",
        coloraxis_colorbar=dict(title="Installs"),
        template="plotly_white",
    )
    
    # Saving the chart as an HTML file
    html_file = "bubble_chart_T9.html"
    fig_17.write_html(html_file)
    
    # Opening the chart in a web browser
    webbrowser.open(html_file)
else:
    print("Bubble chart is not available outside the time range (5 PM - 7 PM IST).")


# #### Task 10: Growth Highlighting Time Series Line Chart
# 
# Goal:
# To visualize the trend of total installs over time, segmented by app category, and highlight periods of significant growth (install increases >20% month-over-month).
# 
# 1. Creating a time series chart showing the trend of total installs for each app category.
# 2. Defining "significant growth" as a month-over-month increase exceeding 20%.
# 3. To include only apps with Content Rating = "Teen", app name starting with the letter 'E', Installs > 10,000.
# 4. The graph should only be visible between 6 PM IST and 9 PM IST.

# In[105]:


filtered_data_T10 = apps[(apps['Content Rating'] == 'Teen')]
filtered_data_T10


# In[106]:


filtered_data_T10 =filtered_data_T10[(filtered_data_T10['App'].str.startswith('E'))]
filtered_data_T10


# In[107]:


filtered_data_T10 = filtered_data_T10[(filtered_data_T10['Installs'] > 10000)]
filtered_data_T10


# In[108]:


# Extract year and month
filtered_data_T10['YearMonth'] = filtered_data_T10['Last Updated'].dt.to_period('M')
filtered_data_T10


# In[109]:


# Step 3: Aggregating the installs by category and month
installs = (
    filtered_data_T10.groupby(['YearMonth', 'Category'])['Installs']
    .sum()
    .reset_index()
    .sort_values('YearMonth')
)


# In[110]:


# Step 4: Calculating the Month-over-Month Growth
installs['Pct_Change'] = (
    installs.groupby('Category')['Installs']
    .pct_change() * 100
)


# In[111]:


# Step 5: Adding a 'Significant Growth' flag
installs['Significant Growth'] = installs['Pct_Change'] > 20
installs


# #### Base on the above output we can see that -
# 1. The 'YearMonth' column offers a clear chronological sequence for monitoring app install trends and is well-structured.
# 2. The monthly percentage change in installs is appropriately shown in the Pct_Change column. Changes, both positive and bad, are accurately documented.
# 3. The Significant Growth column correctly flags periods where the percentage increase in installs exceeds 20%.
# 4. Instances (such as first entries for categories) where the Pct_Change cannot be computed because of missing prior data are suitably denoted as NaN.
# 5. The output accurately depicts key growth periods, including:
# 
# a. Growth rate for 2017-04 (GAME): 1900%.
# b. FAMILY 2018-05: 2000% growth.
# c. Family: 9923.08% increase in 2018–07.

# In[112]:


installs.info()


# #### The 'YearMonth' column in your DataFrame is of type Period, which Plotly cannot serialize directly for use in a chart.
# #### So, we need to convert the Period objects in the YearMonth column to a string or datetime format that Plotly can handle.

# #### Converting the Data Type of 'YearMonth' column from period to datetime format.

# In[113]:


#Before creating the chart, converting the YearMonth column to a string format first
installs['YearMonth'] = installs['YearMonth'].astype(str)
installs.info()


# In[114]:


#There are values in the 'YearMonth' column that are not in the correct YYYY-MM format. In particular the string may have extra text at the end.
#Checking the actual values in the YearMonth column to understand the format.
print(installs['YearMonth'].unique())


# In[115]:


# If YearMonth contains any unwanted text, we can clean it by extracting only the relevant part of the string-
# installs['YearMonth'] = installs['YearMonth'].str[:7]


# In[116]:


# After cleaning, converting the 'YearMonth' to the datetime format finally.

installs['YearMonth'] = pd.to_datetime(installs['YearMonth'], format='%Y-%m', errors='coerce')
installs.info()


# In[117]:


# Step 5: Plot the Time Series Chart
current_time = datetime.now().time()
start_time = datetime.strptime("18:00", "%H:%M").time()
end_time = datetime.strptime("21:00", "%H:%M").time()


# In[118]:


if start_time <= current_time <= end_time:
    # Creating the line chart with Plotly
    fig_18 = px.line(
        installs,
        x='YearMonth',
        y='Installs',
        color='Category',
        line_group='Category',
        title="Trend of Total Installs Over Time (Teen, Apps Starting with 'E')",
        labels={'YearMonth': 'Month-Year', 'Installs': 'Total Installs', 'Category': 'App Category'},
    )

    # Highlighting significant growth areas by adding filled areas
    for category in installs['Category'].unique():
        category_data = installs[(installs['Category'] == category) & (installs['Significant Growth'])]
        fig_18.add_scatter(
            x=category_data['YearMonth'],
            y=category_data['Installs'],
            fill='tozeroy',
            mode='lines',
            name=f"Significant Growth: {category}",
            opacity=0.3
        )

    # Updating the layout for better visualization
    fig_18.update_layout(
        xaxis_title="Month-Year",
        yaxis_title="Total Installs",
        template="plotly_white",
        legend_title="App Categories",
    )

    # Saving the chart as an HTML file
    html_file = "time_series_T10.html"
    fig_18.write_html(html_file)

    # Opening the chart in a web browser
    webbrowser.open(html_file)
else:
    print("Time Series Chart is not available outside the time range (6 PM - 9 PM IST).")


# In[119]:


import webbrowser

# Saving the dashboard HTML content
dashboard_filename = "Final-Dashboard.html"

dashboard_html = """ 
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Google Play Store Analytics Dashboard</title>
    <style>
        /* General Styles */
        body {
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 0;
            background-color: #121212;
            color: white;
            transition: background-color 0.5s, color 0.5s;
            overflow-x: hidden;
        }

        /* Light Mode */
        body.light-mode {
            background-color: white;
            color: black;
        }

        /* Header */
        .header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(90deg, #34A853, #0F9D58);
            color: white;
            font-size: 24px;
            font-weight: bold;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 15px;
            position: relative;
        }

        .header img {
            height: 50px;
            cursor: pointer;
            transition: transform 0.3s ease-in-out;
        }

        .light-mode .header {
            background: linear-gradient(90deg, #ffcc00, #ff9900);
        }

        /* Toggle Mode Button */
        .toggle-container {
            position: absolute;
            right: 20px;
            top: 20px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .toggle-icon {
            width: 40px;
            height: 40px;
            transition: transform 0.3s ease-in-out;
        }

        .light-mode .toggle-icon {
            transform: rotate(180deg);
        }

        /* Container */
        .container {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
            width: 100%;
        }

        /* Plot Cards */
        .plot-card {
            width: 90%;
            height: 600px;
            background: #1E1E1E;
            border-radius: 10px;
            overflow: hidden;
            position: relative;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            cursor: pointer;
            margin-bottom: 30px;
            box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.3);
        }

        .light-mode .plot-card {
            background: #f9f9f9;
            color: black;
        }

        .plot-card:hover {
            transform: scale(1.02);
            box-shadow: 0px 10px 20px rgba(0, 0, 0, 0.4);
        }
        
        /* Disabled Graphs */
        .disabled {
            background: #333 !important;
            cursor: not-allowed;
            color: #bbb;
            text-align: center;
            font-size: 20px;
            padding: 50px;
        }

        .light-mode .disabled {
            background: #e0e0e0;
            color: #666;
        }

        /* Plot Titles */
        .plot-title {
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            padding: 10px;
        }

        /* Embed Graphs */
        .plot-card embed {
            width: 100%;
            height: 100%;
            border: none;
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .plot-card {
                width: 100%;
                height: 500px;
            }
        }
    </style>
    <script>
        function openPlot(filename) {
            window.open(filename, '_blank');
        }

        // Toggle Light/Dark Mode
        function toggleMode() {
            document.body.classList.toggle("light-mode");
            let modeIcon = document.getElementById("modeIcon");

            if (document.body.classList.contains("light-mode")) {
                localStorage.setItem("theme", "light");
                modeIcon.src = "https://cdn-icons-png.flaticon.com/512/1164/1164954.png"; // Light mode icon
            } else {
                localStorage.setItem("theme", "dark");
                modeIcon.src = "https://cdn-icons-png.flaticon.com/512/747/747374.png"; // Dark mode icon
            }
        }

        // Load the theme from localStorage
        window.onload = function () {
            if (localStorage.getItem("theme") === "light") {
                document.body.classList.add("light-mode");
                document.getElementById("modeIcon").src = "https://cdn-icons-png.flaticon.com/512/1164/1164954.png";
            }
        };
    </script>
</head>
<body>
    <div class="header">
        <img src="https://upload.wikimedia.org/wikipedia/commons/7/78/Google_Play_Store_badge_EN.svg" alt="Google Play Store Logo">
        Google Play Store Review Analytics
        <div class="toggle-container" onclick="toggleMode()">
            <img id="modeIcon" class="toggle-icon" src="https://cdn-icons-png.flaticon.com/512/747/747374.png" alt="Toggle Theme">
        </div>
    </div>
    <div class="container">
        <!-- Available Graphs -->
        <div class="plot-card" onclick="openPlot('sentiment_distribution_dashboard_T1.html')">
            <embed src="sentiment_distribution_dashboard_T1.html">
            <p class="plot-title">Sentiment Distribution Analysis</p>
        </div>
        
        <div class="plot-card" onclick="openPlot('scatter_revenue_vs_installs_paid_apps_T3.html')">
            <embed src="scatter_revenue_vs_installs_paid_apps_T3.html">
            <p class="plot-title">Revenue vs Installs (Paid Apps)</p>
        </div>

        <!-- Time-Restricted Graphs -->
        <div class="plot-card disabled">
            <p class="plot-title">Choropleth Map (Available 6 PM - 8 PM)</p>
        </div>
        <div class="plot-card disabled">
            <p class="plot-title">Dual Axis Chart (Available 1 PM - 2 PM)</p>
        </div>
        <div class="plot-card disabled">
            <p class="plot-title">Grouped Bar Chart (Available 3 PM - 5 PM)</p>
        </div>
        <div class="plot-card disabled">
            <p class="plot-title">Correlation Heatmap (Available 2 PM - 4 PM)</p>
        </div>
        <div class="plot-card disabled">
            <p class="plot-title">Violin Plot (Available 4 PM - 6 PM)</p>
        </div>
        <div class="plot-card disabled">
            <p class="plot-title">Bubble Chart (Available 5 PM - 7 PM)</p>
        </div>
        <div class="plot-card disabled">
            <p class="plot-title">Time Series Chart (Available 6 PM - 9 PM)</p>
        </div>
    </div>
</body>
</html>
"""

# Saving the dashboard as an HTML file
with open(dashboard_filename, "w", encoding="utf-8") as file:
    file.write(dashboard_html)

# Opening the dashboard in the web browser
webbrowser.open(dashboard_filename)

print("Dashboard has been successfully opened in the browser.")

import webbrowser
import http.server
import socketserver
import threading

# Defining the port
PORT = 8000

# Function to start the server in a separate thread
def start_server():
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"Serving at http://localhost:{PORT}")
        httpd.serve_forever()

# Starting the server in the background
threading.Thread(target=start_server, daemon=True).start()

# Opening the dashboard in the browser
webbrowser.open(f"http://localhost:{PORT}/Final-Dashboard.html")
print("Dashboard is now accessible at http://localhost:8000/Final-Dashboard.html")


# In[ ]:





# In[ ]:




