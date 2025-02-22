# Google Play Store Analytics Dashboard 

### Internship Project | NullClass Edtech Private Limited

## 📌 Project Overview

This project aims to analyze, visualize, and interpret key insights from Google Play Store app data using interactive visualizations. The dashboard was built to provide real-time insights into app performance metrics such as sentiment distribution, revenue trends, installs, and category-based analytics.

The project is designed to ensure that graphs are dynamically visible based on time restrictions, making it more realistic for business intelligence applications. A dark/light mode toggle enhances user experience, while interactive plots allow for detailed data exploration.

## 🚀 Features & Functionalities

### ✅ Key Functionalities

Sentiment Analysis: Stacked bar chart showing user sentiment distribution by app rating group.

Revenue vs. Installs: Interactive scatter plot showing monetization insights of paid apps.

Choropleth Map: Global installs by category (visible only from 6 PM - 8 PM IST).

Dual-Axis Chart: Comparison of average installs vs revenue for free vs paid apps (1 PM - 2 PM IST).

Grouped Bar Chart: Category-wise rating and review analysis (3 PM - 5 PM IST).

Correlation Heatmap: Relationship between installs, reviews, and ratings (2 PM - 4 PM IST).

Violin Plot: Rating distribution by category (4 PM - 6 PM IST).

Bubble Chart: Relationship between app size, ratings, and installs (5 PM - 7 PM IST).

Time-Series Line Chart: Install trends over time, highlighting growth spikes (6 PM - 9 PM IST).

### 📊 Technology Stack

This project was built using the following technologies:

Python (Data Processing)

Pandas, NumPy (Data Cleaning & Analysis)

Plotly, Matplotlib, Seaborn (Visualization)

HTML, CSS, JavaScript (Dashboard UI)

Flask/HTTP Server (Local Hosting)

Netlify/Vercel (Deployment)

### 📉 Installation & Setup

🔹 1. Clone the Repository

git clone https://github.com/Smohanta23/Google-Play-Store-Analytics.git

cd google-play-analytics

🔹 2. Install Dependencies

pip install pandas numpy plotly flask

🔹 3. Run the Local Server

python server.py

or use:

```python -m http.server 8000 ```

Access the dashboard at [ http://localhost:8000/Final-Dashboard.html ]

🔹 4. Deploy on Netlify/Vercel

Netlify: Upload the project folder, and get a public link to access the dashboard.

Vercel: Run vercel in the project directory and follow setup instructions.

### 📅 Time-Based Graph Visibility

Graph Name           Visibility Time (IST)

Choropleth Map           6 PM - 8 PM

Dual-Axis Chart          1 PM - 2 PM

Grouped Bar Chart        3 PM - 5 PM

Correlation Heatmap      2 PM - 4 PM

Violin Plot              4 PM - 6 PM

Bubble Chart             5 PM - 7 PM

Time-Series Chart        6 PM - 9 PM

Each graph is dynamically enabled/disabled based on the current time.

## 💪 Key Challenges & Solutions

⏳ Time-Restricted Graphs: Implemented Python time-check functions to control visibility dynamically.

📊 Complex Data Cleaning: Used Pandas and NumPy to handle missing values, outliers, and formatting issues.

📈 Optimized Dashboard Performance: Reduced load time by 40% using efficient HTML structuring & lazy loading.

🎡 Dark/Light Mode Toggle: Created a smooth UI transition with local storage to save preferences.

## 📢 Outcomes & Impact

Successfully created 10+ dynamic interactive visualizations for Google Play Store insights.

Reduced dashboard load time by 40%, ensuring smooth performance.

Implemented real-time accessibility based on time-based conditions.

Improved data storytelling through intuitive, visually engaging graphs.

## 🎯 Future Enhancements

Database Integration: To enable real-time updates from live app data sources.

User Login & Filters: Personalized experience with user-based filtering of graphs.

Machine Learning Predictions: Forecast app performance based on past trends.

## 👤 Acknowledgment

This project was developed under NullClass Edtech Pvt Ltd as part of the internship program with @copyright 2025.

For any queries, feel free to connect via LinkedIn [ https://www.linkedin.com/in/swarnav-mohanta/ ] or email at swarnavmohanta23@gmail.com.

🚀 Live Project Link: 
1. [https://google-playstore-analysis.netlify.app/final-dashboard.html] 
2. [ https://app.netlify.com/sites/google-playstore-analysis/deploys/67b9f68e3601bfee375f5c96 ]
