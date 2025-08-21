#Importing all the required Python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

#Loading the dataset
df = pd.read_csv("FloridaBikeRentals.csv", encoding="ISO-8859-1")

#Drop rows with missing values
df.dropna(inplace=True)

#Remove duplicates
df.drop_duplicates(inplace=True)

#Optimize data types
df['Hour'] = df['Hour'].astype('int8')
df['Temperature(°C)'] = df['Temperature(°C)'].astype('float32')
df['Humidity(%)'] = df['Humidity(%)'].astype('float32')
df['Wind speed (m/s)'] = df['Wind speed (m/s)'].astype('float32')

#Export cleaned data to JSON
df.to_json("bike_rental_cleaned.json", orient="records", lines=True)

#Data Processing and Statistical Analysis

#Scale temperature
df['Temperature(°C)'] *= 10

#Normalize Visibility
scaler = MinMaxScaler()
df['Visibility_Scaled'] = scaler.fit_transform(df[['Visibility (10m)']])

#Basic stats
print(df[['Temperature(°C)', 'Humidity(%)', 'Rented Bike Count']].describe())

#Save processed data
df.to_csv("bike_rental_processed.csv", index=False)

#Data Analysis with Pandas

#Categorical vs Numerical
cat_vars = df.select_dtypes(include='object').columns.tolist()
num_vars = df.select_dtypes(include=['int', 'float']).columns.tolist()

#Pivot: Avg rented bike count by season
season_avg = df.groupby('Seasons')['Rented Bike Count'].mean()

#Holiday & Functioning Day impact
holiday_analysis = df.groupby(['Holiday', 'Functioning Day'])['Rented Bike Count'].mean()

#Hourly distribution
hourly_dist = df.groupby('Hour')['Rented Bike Count'].mean()

#Encode categoricals
df_encoded = pd.get_dummies(df, columns=['Seasons', 'Holiday', 'Functioning Day'])
df_encoded.to_csv("Rental_Bike_Data_Dummy.csv", index=False)

#Data Visualization

#Bar Plot: Rentals by Season
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='Seasons', y='Rented Bike Count', estimator=np.mean)
plt.title("Average Rentals by Season")
plt.savefig("season_rentals.png")

#Line Plot: Hourly Rentals
plt.figure(figsize=(10, 5))
sns.lineplot(data=hourly_dist)
plt.title("Hourly Bike Rentals")
plt.xlabel("Hour")
plt.ylabel("Average Rentals")
plt.savefig("hourly_rentals.png")

#Heatmap: Correlation
plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Between Features")
plt.savefig("correlation_heatmap.png")

#Boxplots: Temperature and Rented Bike Count
plt.figure(figsize=(6, 4))
sns.boxplot(x=df['Temperature(°C)'])
plt.title("Temperature Outliers")
plt.savefig("temperature_boxplot.png")

plt.figure(figsize=(6, 4))
sns.boxplot(x=df['Rented Bike Count'])
plt.title("Rented Bike Count Outliers")
plt.savefig("rental_count_boxplot.png")