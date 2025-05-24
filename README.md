# 🚗 Uber
## 📌 Project objective
This project looks at Uber ride data from New York City between April and September 2014 to understand when and where people are most likely to request rides and other deeper insights. By developing and evaluating models like XGBoost, Random Forest, and LSTM to predict future demand, the goal is to help Uber prepare ahead—putting drivers in the right places, reducing wait times, planning promotions, and making better decisions to improve both service and efficiency.
## 🛠 Technologies used
Python (Pandas, NumPy, Matplotlib, Seaborn, XGBoost, Random Forest, LSTM, MAPE) by using GoogleColab
## 🏢 About the company
Uber Technologies, Inc., founded in 2009 and headquartered in San Francisco, California, is a leading technology company that revolutionized urban transportation with its ride-sharing platform. The company’s mission is to make transportation as reliable as running water, everywhere, for everyone. Uber’s innovative app connects millions of riders with drivers in over 70 countries, offering a convenient alternative to traditional taxis. 
## 📂 Dataset
- Dataset: [Uber Pickups in New York City](https://www.kaggle.com/datasets/fivethirtyeight/uber-pickups-in-new-york-city/code)
This folder contains data on over 4.5 million Uber pickups in New York City from April to September 2014, collected by the website FiveThirtyEight from the New York Taxi & Limousine Commission (TLC) through a Freedom of Information Act request on July 20, 2015.
The collected data is mainly concentrated in the Bronx and several locations in Manhattan, including:
- Fordham, Bronx (BX)
- Morris Heights, Bronx
- Jerome Ave, Bronx
- High Bridge, Bronx
- Kingsbridge, Bronx
- Williamsbridge, Bronx
- Manhattan (NYC), main routes such as Broadway, 10th Ave NYC, and W 204 St NYC.

## I. Data cleaning
Before processing data, the first step is to import the necessary libraries. These libraries include:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.cm as cm
import plotly.express as px
```
### 1. Loading data:
The data showing the number of Uber trips from April to September 2014 is spread across several CSV files. To keep things organized and consistent, each file is loaded into its own DataFrame using the pd.read_csv() function.
```python
uber_apr14=pd.read_csv('/content/drive/MyDrive/link data uber/uber-raw-data-apr14.csv')
uber_may14=pd.read_csv('/content/drive/MyDrive/link data uber/uber-raw-data-may14.csv')
uber_jun14=pd.read_csv('/content/drive/MyDrive/link data uber/uber-raw-data-jun14.csv')
uber_jul14=pd.read_csv('/content/drive/MyDrive/link data uber/uber-raw-data-jul14.csv')
uber_aug14=pd.read_csv('/content/drive/MyDrive/link data uber/uber-raw-data-aug14.csv')
uber_sep14=pd.read_csv('/content/drive/MyDrive/link data uber/uber-raw-data-sep14.csv')
```
After that, these DataFrames are combined into a single dataset using pd.concat(). The parameter ignore_index=True is used to reset the row indices, making sure the data stays properly ordered and consistent.
```python
uber = pd.concat([df_apr14, df_may14, df_jun14, df_jul14, df_aug14, df_sep14], ignore_index=True)
```
```python
uber.info()
```
<center>
      <img src="png/info.png"/>
  </center>
  
### 2. Checking null values:
```python
uber.isnull().sum()
```
<center>
      <img src="png/isnull.png"/>
  </center>
  
In this case, a check of the dataset shows that there are no missing values. As a result, there's no need to perform any additional steps like replacing or removing data.
### 3. Checking duplicated values:
```python
uber.duplicated().sum()
```
<center>
      <img src="png/dup.png"/>
  </center>
  
The results show that there are 82,581 duplicate rows in the dataset. Since these duplicates don't add any value to the analysis, the drop_duplicates(inplace=True) method is used to remove them directly from the DataFrame.
```python
uber.drop_duplicates(inplace = True)
```
### 4. Time data processing
We can see that the time data is currently in object format, so it's important to convert it to a proper datetime format. After that, we can add new columns to represent the date, hour, and minute to make the analysis easier and more flexible.
```python
uber.rename(columns={'Date/Time':'Date_time'},inplace=True)

uber['Date_time'] = pd.to_datetime(uber['Date_time'])

uber['month'] = uber['Date_time'].dt.month
uber['weekday'] = uber['Date_time'].dt.day_name()
uber['day'] = uber['Date_time'].dt.day
uber['hour'] = uber['Date_time'].dt.hour
uber['minute'] = uber['Date_time'].dt.minute
```
### 5. Data overview:
```python
uber.info()
```
<center>
      <img src="png/info2.png"/>
  </center>

```python
print(uber['Date_time'].min())
print(uber['Date_time'].max())
```
<center>
      <img src="png/timeminmax.png"/>
  </center>
  
The collected data covers trips from April 1, 2014, to September 30, 2014, totaling 4,451,746 records. Descriptive statistics have been calculated for the time-related column (Date_time), location coordinates (Lat, Lon), and time components such as day (Day), hour (Hour), and minute (Minute).

## II. EDA
### 1. Total Trips
-**By hours**
```python
# Analyze trips by hour across different months
plt.figure(figsize=(30, 20))
colors = ['skyblue', 'salmon', 'lightgreen', 'gold', 'orchid', 'lightcoral']

for i, month in enumerate(uber['month'].unique()):
    plt.subplot(3, 2, i + 1)
    sns.histplot(data=uber, x="hour", bins=24, kde=True, color=colors[i % len(colors)])
    plt.xlabel('Hour of Month {}'.format(month))
    plt.ylabel('Total Trips')

plt.tight_layout()
plt.show()
```
<center>
      <img src="png/trips.hours.png"/>
  </center>
  
- **Peak Hours**: Trips increase from early morning, peaking between **3–6 PM**, likely due to post-work or school travel.
- **Secondary Peaks**: Smaller peaks occur around **7–9 AM** and **12–1 PM**, reflecting morning commutes and lunch breaks.
- **Late Night Drop**: After **11 PM**, trip volume drops sharply.
- **Consistent Pattern**: These trends are consistent across all months, indicating stable daily demand unaffected by season.
- **Business Insight**: Targeted pricing or promotions during peak hours could be effective.

-**By day of the week**
```python
# Plot the total trips by day of the week
plt.figure(figsize=(8,8))
sns.countplot(data=uber, x="weekday", hue="weekday", palette="rainbow", legend=False)

# Add title and axis labels
plt.title("Total Number of Trips by Day of the Week", fontsize=16)
plt.xlabel("Day of the Week", fontsize=14)
plt.ylabel("Total Number of Trips", fontsize=14)

# Show the plot
plt.show()
```
<center>
      <img src="png/trips.weekday.png"/>
  </center>
  
- **Midweek (Thursday, Friday)** sees the **highest** number of trips, exceeding 700,000 per day, likely due to increased travel for work, meetings, and evening activities.
- **Weekends** (Saturday, Sunday) show a **decline**, with **Sunday** having the **lowest** trips (~480,000), possibly because more people stay home or use personal vehicles.

-**By months**
```python
# Analyze total trips by day across different months
plt.figure(figsize=(30, 20))
colors = ['skyblue', 'salmon', 'lightgreen', 'gold', 'orchid', 'lightcoral']

for i, month in enumerate(uber['month'].unique()):
    plt.subplot(3, 2, i + 1)
    sns.histplot(data=uber, x=uber[uber['month'] == month]['day'], bins=31, kde=True, color=colors[i % len(colors)])
    plt.xlabel('Day of Month {}'.format(month))
    plt.ylabel('Total Trips')

plt.tight_layout()
plt.show()
```
<center>
      <img src="png/trips.daymonth.png"/>
  </center>

<center>
      <img src="png/trips.daymonth1.png"/>
  </center>
  
- **Demand spikes every 6–7 days**, likely aligning with **Fridays through Sundays**.
- **August and September** are the **busiest months**, possibly due to **seasonal effects** (e.g., end of summer, return to school, or favorable weather).
- **One day is missing in April, June, and September**, likely due to **data entry errors or missing records**.

### 2. Total Trip Hours
-**By day of the week**
```python
# Box plot analyzing Uber trip times throughout the week
sns.boxplot(x='weekday', y='hour', data=uber, hue='weekday', palette='coolwarm', legend=False)
plt.title("Uber Trip Times Throughout the Week")
plt.xlabel("Day of the Week")
plt.ylabel("Hour of the Day")
plt.show()
```
<center>
      <img src="png/hour.weekday.png"/>
  </center>
  
- The **busiest period is from 10 AM to 7 PM**, capturing the bulk of daily demand.
- The **median trip time stays consistent (~3–4 PM**) across all days, indicating that **Uber usage peaks in the afternoon**. Trips are distributed across the full 24 hours, showing **nearly continuous servic**e.
- **Weekends** (Saturday & Sunday) show **wider variability**, likely due to more **late-night trips** related to social events, while **Weekdays** (Monday to Friday) have **similar, stable patterns** with less variability, reflecting **commuting routines**.
- Overall, the **distribution is quite uniform** across the week, with **few outliers**, suggesting **no major behavioral shift between weekdays and weekends**.

-**By day of the month**
```python
#Total driving hours per day in the month
uber.groupby('day')['hour'].count()
```
<center>
      <img src="png/hour.daymonth.png"/>
  </center>
  
```python
# Analyze total driving hours by day of the month
day_hours = uber.groupby('day')['hour'].count().reset_index(name='Total Driving Hours')

# Use a new colormap
cmap = plt.colormaps['rainbow']

# Create a list of colors from the colormap
colors = [cmap(i / len(day_hours)) for i in range(len(day_hours))]

# Plot a bar chart with vibrant colors
plt.figure(figsize=(20, 10))
bars = sns.barplot(x='day', y='Total Driving Hours', hue='day', data=day_hours, palette=colors, legend=False)

# Add title and axis labels
plt.title("Total Driving Hours by Day of the Month", fontsize=16)
plt.xlabel("Day of the Month", fontsize=14)
plt.ylabel("Total Driving Hours", fontsize=14)

# Show the plot
plt.show()
```
<center>
      <img src="png/hour.daymonth1.png"/>
  </center>
  
- Driving hours are **relatively stable** throughout the month, with **low activity in the first few days**, especially on the 1st.
- The **30th** often shows a **noticeable spike** in total driving hours, possibly due to end-of-month activities like salary payments or shopping.
- The **31st** sees a **sharp drop** in total driving hours, likely because not all months have 31 days.

-**By months**
```python
#Analyzes which month has the highest total number of trip hours
uber.groupby('month')['hour'].count()
```
<center>
      <img src="png/hour.month.png"/>
  </center>
  
```python
monthly_trips = uber.groupby('month')['hour'].count().reset_index(name='Total Trip Hours')
colors = ['skyblue', 'salmon', 'lightgreen', 'gold', 'orchid', 'lightcoral']

plt.figure(figsize=(10, 10))
sns.barplot(data=monthly_trips, x='month', y='Total Trip Hours', hue='month', palette=colors, legend=False)
plt.xticks(ticks=[0,1,2,3,4,5], labels=['April', 'May', 'June', 'July', 'August', 'September'])
plt.xlabel('Month')
plt.title("Total Trip Hours by Month", fontsize=16)
plt.show()
```
<center>
      <img src="pnghour.month1.png"/>
  </center>
  
- Total driving hours **steadily rise** from **April to September**, **peaking** in **September** with over 1 million hours. The most significant growth occurs between **June and September**, likely due to the **summer tourism peak** in New York, attracting many visitors and boosting Uber demand.
- Warm weather and outdoor activities encourage more travel. **August and September** also see **increased** demand as students and workers **return from summer break**s, especially in **early September**.

### 3. Geolocation
-**Passenger latitude**
```python
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Plot the graph
plt.figure(figsize=(15, 10))
sns.pointplot(data=uber, x='hour', y='Lat', hue='weekday', palette="rainbow")

# Set the title (translated to English)
plt.title("Peak Hour Analysis by Passenger Latitude", fontsize=16)

# Automatically get handles and labels to preserve colors
handles, labels = plt.gca().get_legend_handles_labels()

# Add legend with weekday labels and keep colors consistent
plt.legend(handles=handles, labels=days, title="Day of the Week", fontsize=12)
plt.xlabel("Hour", fontsize=14)
plt.ylabel("Latitude", fontsize=14)

# Show the plot
plt.show()
```
<center>
      <img src="png/lat.png"/>
  </center>
  
- **Weekday vs Weekend Patterns**: **Monday to Friday** show **higher average latitude**s, indicating that pickups tend to cluster more **northward** (likely in Manhattan). **Saturday and Sunday** have **lower average latitude**s, suggesting more activity **further south** or outside central business areas.
- **Morning Spike (7–9 AM)**: Clear uptick in latitude on **weekdays**, indicating work-related commuting patterns into more northern (business) areas.
- **Evening Decline (after 6 PM)**: Gradual decrease in latitude, implying dispersal from business centers.
- **Late Night Drop (after 10 PM)**: All days show a sharp decline in trip latitude, reflecting lower demand and more scattered locations.

-**Base**
📌 A **base** is a licensed For-Hire Vehicle (FHV) base registered with the New York City Taxi and Limousine Commission (TLC). Each base (e.g., B02598) connects drivers with passengers and reports trip data to the city. Uber operates through multiple such bases, which may represent different regions, partners, or business units within the platform.
```python
uber['Base'].nunique()
```
<center>
      <img src="png/unique.png"/>
  </center>
  
```python
#analyzes which locations (bases) have the highest total number of trips by hour
base = uber.groupby(['Base','hour'])['Date_time'].count().reset_index()
base
```
<center>
      <img src="png/base.png"/>
  </center>
  
```python
fig4=px.scatter(base,
            x='hour',
            y='Date_time',
            color='Base',
            template='plotly_dark',
            color_discrete_sequence=['#50F9F1','#6AE5A8','#96C96A','#B7AA47','#C78845'])
fig4.update_layout(title_text='Total Trips by Location and Time of Day', title_x=0.5)


fig4.update_xaxes(title_text="Hour")
fig4.update_yaxes(title_text="Total Trips")

fig4.show()
```
<center>
      <img src="png/base1.png"/>
  </center>
  
- Total trips are **lowest** between **2 AM and 5 AM**, reflecting minimal overnight demand.
- Trips begin to **rise steadily after 6 AM**, aligning with the morning commute.
- The **highest** trip volumes occur between **3 PM and 7 PM**, likely due to evening commute and after-work activities.
- There is a **noticeable spike** at **midnight** (12 AM), possibly linked to late-night travel or nightlife.
- Bases **B02598** and **B02617** consistently record **higher trip counts**, while **B02512** and **B02764** have **lower overall activity**.

-**City**
```python
Demand=(uber.groupby(['Lat','Lon']).count()['Date_time']).reset_index()
Demand.columns=['Latitude','Longitude','Number of Trips']
import folium as fo
from folium.plugins import HeatMap
base1=fo.Map()
HeatMap(Demand,zoom=20,radius=15).add_to(base1)
base1
```
<center>
      <img src="png/geo.png"/>
  </center>
  
- **Manhattan** is the main **hotspot** for taxi demand due to its **high population density, business centers, tourism, and entertainment**.
- Major routes to **Long Island** and **New Jersey** show **high trip density**, reflecting strong **commuter traffic** between the city and suburbs.
- Other notable hotspots include areas around **Newark Airport**, and suburban zones like **Stamford** and **New Brunswick**, indicating **significant regional travel**.

### 4. Heatmaps
```python
#Heatmap
def count_rows(row):
    return len(row)


def heatmap(col1,col2):
    by_cross = uber.groupby([col1,col2]).size().unstack()
    plt.figure(figsize=(12,8))
    sns.heatmap(by_cross, cmap="coolwarm")
```
```python
#By hour and day of the week
heatmap('weekday', 'hour')
plt.xlabel('Hour')
plt.ylabel('Weekday')
plt.title('Heatmap of Uber Trips by Hour and Day of Week')
plt.show()
```
<center>
      <img src="png/hd.png"/>
  </center>
  
```python
#By Hour and Day of the month
heatmap('day', 'hour')
plt.xlabel('Hour of Day')
plt.ylabel('Day of Month')
plt.title('Heatmap of Uber Trips by Day and Hour')
plt.show()
```
<center>
      <img src="png/dh.png"/>
  </center>
  
```python
#by month and day of the month
heatmap('month', 'day')
plt.xlabel('Day of Month')
plt.ylabel('Month')
plt.title('Heatmap of Uber Trips by Month and Day of Month')
plt.show()
```
<center>
      <img src="png/mm.png"/>
  </center>
  
```python
#by month and weekday 
heatmap('month', 'weekday')
plt.xlabel('Day of Week')
plt.ylabel('Month')
plt.title('Heatmap of Uber Trips by Month and Day of Week')
plt.show()
```
<center>
      <img src="png/md.png"/>
  </center>
  
📝 Key observations
- The peak demand hour 17:00.
- The main customer category are workers.
- An indicator of Uber's improvement from April to September.
- People tend to use Uber to go to work around 7:00 and 8:00 on working days.
- People tend to use Uber late at night (around midnight) during weekends.
- We should investigate why people don't use uber on Mondays as much as they do on other working days.

## III. Machine learning







