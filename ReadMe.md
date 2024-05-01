# **Comprehensive Cardiovascular Disease Data Analysis and Application**
![image](https://github.com/moseskigo/Capstone-Project/assets/128637199/b0d0b4aa-9783-49ab-ac00-740d83b97613)

## Team Composition
* Moses Kigo
* Josephine Wanjiru
* Eunita Nyengo
* Erick Lekishon
* Chepkemoi Chepkemoi

## Project Overview
This project aims to leverage machine learning techniques to predict cardiovascular disease based on clinical and demographic data. By analyzing historical patient data, we intend to uncover risk factors, develop predictive models, and implement these models in a real-world health application.

## Background
Cardiovascular diseases (CVDs) are a top cause of death globally, taking an estimated 17.9 million lives each year, according to the World Health Organization. Utilizing data science to predict cardiovascular disease can help in early diagnosis and potentially save lives through timely intervention.

## Data Description
The dataset for this project comes from a public health database. It includes records for 70,000 patients with features such as age, gender, blood pressure, cholesterol level, and other medical history indicators. This comprehensive dataset provides a basis for developing robust predictive models.

## Objectives
1. To identify the Key Risk Factors: Use feature importance and exploratory data analysis to highlight significant predictors of CVD.
2. To  develop a Predictive Model: Construct an accurate model to assess the risk of CVD among individuals.
3. Patient Segmentation: Develop a patient segmentation strategy to efficiently identify individuals with high-risk profiles by analyzing pertinent risk factors.

## Methodology 
The project followed these steps:
1. **Data Cleaning and Preparation**: Filter out unrealistic entries and handle missing values.
2. **Feature Engineering**: Develop new features that may help improve model accuracy.
3. **Exploratory Data Analysis**: Visualize the data to understand patterns and relationships.
4. **Model Development**: Train and evaluate models like Logistic Regression, Random Forest, and Gradient Boosting.
5. **Clustering Analysis**: Use K-means to segment patients based on their risk profiles.
6. **Deployment**: Outline the process for integrating models into a web application.

## Data Acquisition Cleaning & Preparation

```python
# Loading Dataset
path = 'cardio_train.csv'
df = pd.read_csv(path,delimiter=';')
#Display the first few rows
df.head()
```

  <div id="df-ee276431-f7a8-4231-8d60-4e483537725f" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>age</th>
      <th>gender</th>
      <th>height</th>
      <th>weight</th>
      <th>ap_hi</th>
      <th>ap_lo</th>
      <th>cholesterol</th>
      <th>gluc</th>
      <th>smoke</th>
      <th>alco</th>
      <th>active</th>
      <th>cardio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>18393</td>
      <td>2</td>
      <td>168</td>
      <td>62.0</td>
      <td>110</td>
      <td>80</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>20228</td>
      <td>1</td>
      <td>156</td>
      <td>85.0</td>
      <td>140</td>
      <td>90</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>18857</td>
      <td>1</td>
      <td>165</td>
      <td>64.0</td>
      <td>130</td>
      <td>70</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>17623</td>
      <td>2</td>
      <td>169</td>
      <td>82.0</td>
      <td>150</td>
      <td>100</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>17474</td>
      <td>1</td>
      <td>156</td>
      <td>56.0</td>
      <td>100</td>
      <td>60</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-ee276431-f7a8-4231-8d60-4e483537725f')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>


## **Data description**

There are 3 types of input features:

1. **Objective** : factual information;
2. **Examination** : results of medical examination;
3. **Subjective** : information given by the patient.

Features:

| Number | Column Name | Description | Feature Type | Data Type |
|--------|-------------|-------------|--------------|-----------|
| 1 | age | Age | Objective Feature | int (days) |
| 2 | height | Height | Objective Feature | int (cm) |
| 3 | weight | Weight | Objective Feature | float (kg) |
| 4 | gender | Gender | Objective Feature | categorical code; 1:Female, 2:Male |
| 5 | ap_hi | Systolic blood pressure | Examination Feature | int |
| 6 | ap_lo | Diastolic blood pressure | Examination Feature | int |
| 7 | cholesterol | Cholesterol | Examination Feature | 1: normal, 2: above normal, 3: well above normal |
| 8 | gluc | Glucose | Examination Feature | 1: normal, 2: above normal, 3: well above normal |
| 9 | smoke | Smoking | Subjective Feature | binary |
| 10 | alco | Alcohol intake | Subjective Feature | binary |
| 11 | active | Physical activity | Subjective Feature | binary |
| 12 | cardio | Presence or absence of cardiovascular disease | Target Variable | binary |

### **Data cleaning**

There are no missing values or duplicates in this dataset

There are no duplicated rows in this dataset

```python
#concise summary of the dataset
df.info()
```
### **Summary Statistics**
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 70000 entries, 0 to 69999
    Data columns (total 13 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   id           70000 non-null  int64  
     1   age          70000 non-null  int64  
     2   gender       70000 non-null  int64  
     3   height       70000 non-null  int64  
     4   weight       70000 non-null  float64
     5   ap_hi        70000 non-null  int64  
     6   ap_lo        70000 non-null  int64  
     7   cholesterol  70000 non-null  int64  
     8   gluc         70000 non-null  int64  
     9   smoke        70000 non-null  int64  
     10  alco         70000 non-null  int64  
     11  active       70000 non-null  int64  
     12  cardio       70000 non-null  int64  
    dtypes: float64(1), int64(12)
    memory usage: 6.9 MB
    
    Number of unique values for each categorical column:
    gender         2
    cholesterol    3
    gluc           3
    smoke          2
    alco           2
    active         2
    cardio         2
    dtype: int64

### **Data Visualisation**

# The distribution of numerical variables using histograms
    
![png](Capstone_Project_21042024_files/Capstone_Project_21042024_21_0.png)
    
# Box plot plotting to check on outliers
    
![png](Capstone_Project_21042024_files/Capstone_Project_21042024_22_0.png)

# Plotting scatter plots for each pair of numerical columns
    
![png](Capstone_Project_21042024_files/Capstone_Project_21042024_23_0.png)
    
# The distribution of categorical features
    
![png](Capstone_Project_21042024_files/Capstone_Project_21042024_24_0.png)
    
# The relationship between categorical and numerical variables using  violin plots.
    
![png](Capstone_Project_21042024_files/Capstone_Project_21042024_25_0.png)
    
**Correlation Analysis**

# Correlation Matrix Heatmap

![png](Capstone_Project_21042024_files/Capstone_Project_21042024_28_0.png)
    

**Exploring the Target Variable**

# The distribution of the target variable to understand class imbalance.
    
![png](Capstone_Project_21042024_files/Capstone_Project_21042024_30_0.png)
    

The target variable 'cardio' appears to be balanced, with approximately equal representation of both classes (0 and 1).

# Visualizing the relationship between the target variable and other features
    
![png](Capstone_Project_21042024_files/Capstone_Project_21042024_32_0.png)
    

### Feature Engineering

## **Summary of new feature categories**

- Age Categories
1. Under 40
2. Age 40-44
3. Age 45-49
4. Age 50-54
5. Age 55-59
6. Age 60-64
7. Over 65

- BMI Categories
1. Underweight
2. Normal Weight
3. Overweight
4. Obese

- Blood Pressure Categories
0. Uncategorized
1. Normal
2. Elevated
3. Hypertension Stage 1
4. Hypertension Stage 2
5. Hypertensive Crisis

- Pulse Pressure Categories
1. Normal
2. Elevated
3. High

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>age</th>
      <th>gender</th>
      <th>height</th>
      <th>weight</th>
      <th>ap_hi</th>
      <th>ap_lo</th>
      <th>cholesterol</th>
      <th>gluc</th>
      <th>smoke</th>
      <th>alco</th>
      <th>active</th>
      <th>cardio</th>
      <th>age_years</th>
      <th>age_category</th>
      <th>BMI</th>
      <th>BMI_Category</th>
      <th>BP_Category</th>
      <th>pulse_pressure</th>
      <th>pulse_pressure_category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>18393</td>
      <td>2</td>
      <td>168</td>
      <td>62.0</td>
      <td>110</td>
      <td>80</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>50</td>
      <td>4</td>
      <td>22.0</td>
      <td>2</td>
      <td>3</td>
      <td>30</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>20228</td>
      <td>1</td>
      <td>156</td>
      <td>85.0</td>
      <td>140</td>
      <td>90</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>55</td>
      <td>5</td>
      <td>35.0</td>
      <td>4</td>
      <td>4</td>
      <td>50</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>18857</td>
      <td>1</td>
      <td>165</td>
      <td>64.0</td>
      <td>130</td>
      <td>70</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>52</td>
      <td>4</td>
      <td>24.0</td>
      <td>2</td>
      <td>3</td>
      <td>60</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>17623</td>
      <td>2</td>
      <td>169</td>
      <td>82.0</td>
      <td>150</td>
      <td>100</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>48</td>
      <td>3</td>
      <td>29.0</td>
      <td>3</td>
      <td>4</td>
      <td>50</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>17474</td>
      <td>1</td>
      <td>156</td>
      <td>56.0</td>
      <td>100</td>
      <td>60</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>48</td>
      <td>3</td>
      <td>23.0</td>
      <td>2</td>
      <td>1</td>
      <td>40</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-8d9604a7-b463-4c5e-ac3f-200147ef3bfe')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

# checking correlation using a heatmap
    
![png](Capstone_Project_21042024_files/Capstone_Project_21042024_46_0.png)
    
#relationship between age and BMI
    
![png](Capstone_Project_21042024_files/Capstone_Project_21042024_47_0.png)
    
# Define age intervals of 5 years
  
![png](Capstone_Project_21042024_files/Capstone_Project_21042024_48_0.png)
    
This bar graph  may not be a true representation of the average BMI per age group due to non uniformity in the number of respondents in each age group.

  <div id="df-9b9d0b56-787e-4c78-a2cc-119bfec193ad" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>age</th>
      <th>gender</th>
      <th>height</th>
      <th>weight</th>
      <th>ap_hi</th>
      <th>ap_lo</th>
      <th>cholesterol</th>
      <th>gluc</th>
      <th>smoke</th>
      <th>alco</th>
      <th>active</th>
      <th>cardio</th>
      <th>age_years</th>
      <th>age_category</th>
      <th>BMI</th>
      <th>BMI_Category</th>
      <th>BP_Category</th>
      <th>pulse_pressure</th>
      <th>pulse_pressure_category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>18393</td>
      <td>2</td>
      <td>168</td>
      <td>62.0</td>
      <td>110</td>
      <td>80</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>50</td>
      <td>4</td>
      <td>22.0</td>
      <td>2</td>
      <td>3</td>
      <td>30</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>20228</td>
      <td>1</td>
      <td>156</td>
      <td>85.0</td>
      <td>140</td>
      <td>90</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>55</td>
      <td>5</td>
      <td>35.0</td>
      <td>4</td>
      <td>4</td>
      <td>50</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>18857</td>
      <td>1</td>
      <td>165</td>
      <td>64.0</td>
      <td>130</td>
      <td>70</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>52</td>
      <td>4</td>
      <td>24.0</td>
      <td>2</td>
      <td>3</td>
      <td>60</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>17623</td>
      <td>2</td>
      <td>169</td>
      <td>82.0</td>
      <td>150</td>
      <td>100</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>48</td>
      <td>3</td>
      <td>29.0</td>
      <td>3</td>
      <td>4</td>
      <td>50</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>17474</td>
      <td>1</td>
      <td>156</td>
      <td>56.0</td>
      <td>100</td>
      <td>60</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>48</td>
      <td>3</td>
      <td>23.0</td>
      <td>2</td>
      <td>1</td>
      <td>40</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

### Feature Distribution in Cluster
    
![png](Capstone_Project_21042024_files/Capstone_Project_21042024_51_0.png)
  
# **Removing outliers**

Blood Pressure: We chose to limit systolic (ap_hi) and diastolic (ap_lo) blood pressure to reasonable adult ranges, such as systolic from 90 to 250 mmHg, and diastolic from 60 to 150 mmHg.\
Height and Weight: Remove heights below 50 cm and above 250 cm.
In the same way, filter out weights below 30 kg or above 200 kg as they are less likely to be accurate.
  
![png](Capstone_Project_21042024_files/Capstone_Project_21042024_55_0.png)
    
### Clustering & Segmentation

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>age</th>
      <th>gender</th>
      <th>height</th>
      <th>weight</th>
      <th>ap_hi</th>
      <th>ap_lo</th>
      <th>cholesterol</th>
      <th>gluc</th>
      <th>smoke</th>
      <th>alco</th>
      <th>active</th>
      <th>cardio</th>
      <th>age_years</th>
      <th>age_category</th>
      <th>BMI</th>
      <th>BMI_Category</th>
      <th>BP_Category</th>
      <th>pulse_pressure</th>
      <th>pulse_pressure_category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>18393</td>
      <td>2</td>
      <td>168</td>
      <td>62.0</td>
      <td>110</td>
      <td>80</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>50</td>
      <td>4</td>
      <td>22.0</td>
      <td>2</td>
      <td>3</td>
      <td>30</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>20228</td>
      <td>1</td>
      <td>156</td>
      <td>85.0</td>
      <td>140</td>
      <td>90</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>55</td>
      <td>5</td>
      <td>35.0</td>
      <td>4</td>
      <td>4</td>
      <td>50</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>18857</td>
      <td>1</td>
      <td>165</td>
      <td>64.0</td>
      <td>130</td>
      <td>70</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>52</td>
      <td>4</td>
      <td>24.0</td>
      <td>2</td>
      <td>3</td>
      <td>60</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>17623</td>
      <td>2</td>
      <td>169</td>
      <td>82.0</td>
      <td>150</td>
      <td>100</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>48</td>
      <td>3</td>
      <td>29.0</td>
      <td>3</td>
      <td>4</td>
      <td>50</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>17474</td>
      <td>1</td>
      <td>156</td>
      <td>56.0</td>
      <td>100</td>
      <td>60</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>48</td>
      <td>3</td>
      <td>23.0</td>
      <td>2</td>
      <td>1</td>
      <td>40</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

 ## Correlation Heatmap
    
![png](Capstone_Project_21042024_files/Capstone_Project_21042024_60_0.png)
    
## Silhouette Score for Optimal Number of Clusters

![WhatsApp Image 2024-04-21 at 21 22 57_dc02fc86](https://github.com/moseskigo/Capstone-Project/assets/128637199/87eaf461-53b8-4fb8-91f8-964051cae342)

From the graph above the optimal number of clusters is 5.

## Title
    
![png](Capstone_Project_21042024_files/Capstone_Project_21042024_64_0.png)
    
# Distribution of Data Points in Clusters'
    
![png](Capstone_Project_21042024_files/Capstone_Project_21042024_65_0.png)
    
**Interpretation**

- **Cluster 0:** Highest prevalence of cardiovascular disease (75% of individuals have it).
- **Cluster 1:**Lowest prevalence of cardiovascular disease (22%  have it).
- **Cluster 2:** Second highest prevalence of cardiovascular disease (70%  have it).
- **Cluster 3:** Second lowest prevalence of cardiovascular disease (34% have it).
- **Cluster 4:** Mixed prevalence of cardiovascular disease (59% do not have it).

# Characteristics of Clusters
    
![png](Capstone_Project_21042024_files/Capstone_Project_21042024_67_0.png)

# Feature Distribution in Cluster
    
![png](Capstone_Project_21042024_files/Capstone_Project_21042024_68_0.png)
    
Cluster Summaries:

- **Cluster 0 - The High-Risk Elderly:** This cluster is marked by the oldest average age and elevated mean values for both blood pressure and cholesterol. This group's health metrics suggest a higher prevalence of age-related cardiovascular risks, warranting close medical attention and potential intervention.

- **Cluster 1 - The Younger, Healthier Subset:** Patients in this cluster present the healthiest profile, with the lowest average cholesterol and glucose levels among all clusters. The youthful demographic of this cluster aligns with the favorable health indicators, suggesting a current low cardiovascular risk.

- **Cluster 2 - The Metabolic Challenge Group:** Exhibiting the highest average blood pressure and substantial mean cholesterol and glucose levels, this cluster is indicative of a significant metabolic syndrome risk. These individuals are prime candidates for aggressive lifestyle and medicinal strategies to manage hypertension and glucose metabolism.

- **Cluster 3 - Young, Yet at Risk:** Despite having the lowest mean age, this cluster shows high mean BMI and BP categories. The juxtaposition of youth with emerging risk factors implies a critical window for effective lifestyle modifications to prevent long-term health complications.

- **Cluster 4 - Well-managed Mature Group:** Patients in this cluster, while similar in age to Cluster 0, display better-controlled health parameters, with the lowest average values for BP, cholesterol, and glucose. It represents an older group that appears to manage their health effectively, possibly through proactive healthcare engagement and treatment adherence.

**Conclusions:**\
The cluster analysis highlights diverse health profiles within the patient population, identifying both high-risk groups and those with better-managed health parameters. The insights provided by the clustering enable targeted healthcare strategies to improve patient outcomes. For high-risk clusters, especially Cluster 0 and Cluster 2, heightened medical supervision is advised. In contrast, Clusters 1 and 4 can continue preventive care and regular health monitoring. Cluster 3's juxtaposition of youth and elevated BMI/BP suggests prioritizing preventive care to mitigate the risk of future health issues.

## relationship between cardiovascular disease and BMI category,BP category,pulse pressure, age in years
   
![png](Capstone_Project_21042024_files/Capstone_Project_21042024_70_0.png)
    
# Histograms for BMI and blood pressure by cluster
    
![png](Capstone_Project_21042024_files/Capstone_Project_21042024_71_0.png)
    

## Modelling
We used six models: 
1. Linear Regression - Baseline model
2. Logistic Regression
3. KNN Classifier
4. Random Forest Classifier
5. Decision Tree Classifier
6. XG Boost Classifier 

    Linear Regression Model Performance:
    Accuracy: 0.7133406034169393
    ROC AUC Score: 0.7763864270613108
    Recall: 0.6443636363636364
    Precision: 0.7473009446693657
    Root Mean Squared Error (RMSE): 0.44255496716948906
       
![png](Capstone_Project_21042024_files/Capstone_Project_21042024_76_1.png)
    
# Classifier Model selection, evaluation and tuning

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Model</th>
      <th>Metric</th>
      <th>Before Tuning</th>
      <th>After Tuning</th>
      <th>Difference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Logistic Regression</td>
      <td>Train Accuracy</td>
      <td>70.81</td>
      <td>70.81</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td>Logistic Regression</td>
      <td>Validation Accuracy</td>
      <td>70.38</td>
      <td>70.38</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td>Logistic Regression</td>
      <td>Validation ROC AUC</td>
      <td>76.87</td>
      <td>76.87</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td>Logistic Regression</td>
      <td>Validation Recall</td>
      <td>63.75</td>
      <td>63.75</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td>Logistic Regression</td>
      <td>Validation Precision</td>
      <td>73.11</td>
      <td>73.11</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td>Logistic Regression</td>
      <td>Validation RMSE</td>
      <td>0.44</td>
      <td>0.44</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td>Random Forest</td>
      <td>Train Accuracy</td>
      <td>72.34</td>
      <td>72.05</td>
      <td>-0.29</td>
    </tr>
    <tr>
      <td>Random Forest</td>
      <td>Validation Accuracy</td>
      <td>70.43</td>
      <td>70.73</td>
      <td>0.30</td>
    </tr>
    <tr>
      <td>Random Forest</td>
      <td>Validation ROC AUC</td>
      <td>76.88</td>
      <td>77.57</td>
      <td>0.69</td>
    </tr>
    <tr>
      <td>Random Forest</td>
      <td>Validation Recall</td>
      <td>64.66</td>
      <td>64.66</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td>Random Forest</td>
      <td>Validation Precision</td>
      <td>72.71</td>
      <td>73.22</td>
      <td>0.51</td>
    </tr>
    <tr>
      <td>Random Forest</td>
      <td>Validation RMSE</td>
      <td>0.44</td>
      <td>0.44</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td>KNN</td>
      <td>Train Accuracy</td>
      <td>69.54</td>
      <td>70.59</td>
      <td>1.05</td>
    </tr>
    <tr>
      <td>KNN</td>
      <td>Validation Accuracy</td>
      <td>67.54</td>
      <td>69.07</td>
      <td>1.53</td>
    </tr>
    <tr>
      <td>KNN</td>
      <td>Validation ROC AUC</td>
      <td>72.07</td>
      <td>74.47</td>
      <td>2.40</td>
    </tr>
    <tr>
      <td>KNN</td>
      <td>Validation Recall</td>
      <td>59.05</td>
      <td>61.45</td>
      <td>2.40</td>
    </tr>
    <tr>
      <td>KNN</td>
      <td>Validation Precision</td>
      <td>70.70</td>
      <td>72.10</td>
      <td>1.40</td>
    </tr>
    <tr>
      <td>KNN</td>
      <td>Validation RMSE</td>
      <td>0.47</td>
      <td>0.46</td>
      <td>-0.01</td>
    </tr>
    <tr>
      <td>Decision Tree</td>
      <td>Train Accuracy</td>
      <td>72.34</td>
      <td>72.14</td>
      <td>-0.20</td>
    </tr>
    <tr>
      <td>Decision Tree</td>
      <td>Validation Accuracy</td>
      <td>70.34</td>
      <td>70.51</td>
      <td>0.17</td>
    </tr>
    <tr>
      <td>Decision Tree</td>
      <td>Validation ROC AUC</td>
      <td>76.64</td>
      <td>76.97</td>
      <td>0.33</td>
    </tr>
    <tr>
      <td>Decision Tree</td>
      <td>Validation Recall</td>
      <td>64.16</td>
      <td>64.62</td>
      <td>0.46</td>
    </tr>
    <tr>
      <td>Decision Tree</td>
      <td>Validation Precision</td>
      <td>72.83</td>
      <td>72.87</td>
      <td>0.04</td>
    </tr>
    <tr>
      <td>Decision Tree</td>
      <td>Validation RMSE</td>
      <td>0.44</td>
      <td>0.44</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td>XGBoost</td>
      <td>Train Accuracy</td>
      <td>72.10</td>
      <td>71.73</td>
      <td>-0.37</td>
    </tr>
    <tr>
      <td>XGBoost</td>
      <td>Validation Accuracy</td>
      <td>70.64</td>
      <td>71.09</td>
      <td>0.45</td>
    </tr>
    <tr>
      <td>XGBoost</td>
      <td>Validation ROC AUC</td>
      <td>77.49</td>
      <td>77.83</td>
      <td>0.34</td>
    </tr>
    <tr>
      <td>XGBoost</td>
      <td>Validation Recall</td>
      <td>64.50</td>
      <td>65.79</td>
      <td>1.29</td>
    </tr>
    <tr>
      <td>XGBoost</td>
      <td>Validation Precision</td>
      <td>73.16</td>
      <td>73.22</td>
      <td>0.06</td>
    </tr>
    <tr>
      <td>XGBoost</td>
      <td>Validation RMSE</td>
      <td>0.44</td>
      <td>0.44</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>

## Visualising the results 

<table id="T_a712e" class="dataframe">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_a712e_level0_col0" class="col_heading level0 col0" colspan="6">Before Tuning</th>
      <th id="T_a712e_level0_col6" class="col_heading level0 col6" colspan="6">After Tuning</th>
    </tr>
    <tr>
      <th class="index_name level1" >Metric</th>
      <th id="T_a712e_level1_col0" class="col_heading level1 col0" >Train Accuracy</th>
      <th id="T_a712e_level1_col1" class="col_heading level1 col1" >Validation Accuracy</th>
      <th id="T_a712e_level1_col2" class="col_heading level1 col2" >Validation Precision</th>
      <th id="T_a712e_level1_col3" class="col_heading level1 col3" >Validation RMSE</th>
      <th id="T_a712e_level1_col4" class="col_heading level1 col4" >Validation ROC AUC</th>
      <th id="T_a712e_level1_col5" class="col_heading level1 col5" >Validation Recall</th>
      <th id="T_a712e_level1_col6" class="col_heading level1 col6" >Train Accuracy</th>
      <th id="T_a712e_level1_col7" class="col_heading level1 col7" >Validation Accuracy</th>
      <th id="T_a712e_level1_col8" class="col_heading level1 col8" >Validation Precision</th>
      <th id="T_a712e_level1_col9" class="col_heading level1 col9" >Validation RMSE</th>
      <th id="T_a712e_level1_col10" class="col_heading level1 col10" >Validation ROC AUC</th>
      <th id="T_a712e_level1_col11" class="col_heading level1 col11" >Validation Recall</th>
    </tr>
    <tr>
      <th class="index_name level0" >Model</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
      <th class="blank col5" >&nbsp;</th>
      <th class="blank col6" >&nbsp;</th>
      <th class="blank col7" >&nbsp;</th>
      <th class="blank col8" >&nbsp;</th>
      <th class="blank col9" >&nbsp;</th>
      <th class="blank col10" >&nbsp;</th>
      <th class="blank col11" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_a712e_level0_row0" class="row_heading level0 row0" >Decision Tree</th>
      <td id="T_a712e_row0_col0" class="data row0 col0" >72.34</td>
      <td id="T_a712e_row0_col1" class="data row0 col1" >70.34</td>
      <td id="T_a712e_row0_col2" class="data row0 col2" >72.83</td>
      <td id="T_a712e_row0_col3" class="data row0 col3" >0.44</td>
      <td id="T_a712e_row0_col4" class="data row0 col4" >76.64</td>
      <td id="T_a712e_row0_col5" class="data row0 col5" >64.16</td>
      <td id="T_a712e_row0_col6" class="data row0 col6" >72.14</td>
      <td id="T_a712e_row0_col7" class="data row0 col7" >70.51</td>
      <td id="T_a712e_row0_col8" class="data row0 col8" >72.87</td>
      <td id="T_a712e_row0_col9" class="data row0 col9" >0.44</td>
      <td id="T_a712e_row0_col10" class="data row0 col10" >76.97</td>
      <td id="T_a712e_row0_col11" class="data row0 col11" >64.62</td>
    </tr>
    <tr>
      <th id="T_a712e_level0_row1" class="row_heading level0 row1" >KNN</th>
      <td id="T_a712e_row1_col0" class="data row1 col0" >69.54</td>
      <td id="T_a712e_row1_col1" class="data row1 col1" >67.54</td>
      <td id="T_a712e_row1_col2" class="data row1 col2" >70.70</td>
      <td id="T_a712e_row1_col3" class="data row1 col3" >0.47</td>
      <td id="T_a712e_row1_col4" class="data row1 col4" >72.07</td>
      <td id="T_a712e_row1_col5" class="data row1 col5" >59.05</td>
      <td id="T_a712e_row1_col6" class="data row1 col6" >70.59</td>
      <td id="T_a712e_row1_col7" class="data row1 col7" >69.07</td>
      <td id="T_a712e_row1_col8" class="data row1 col8" >72.10</td>
      <td id="T_a712e_row1_col9" class="data row1 col9" >0.46</td>
      <td id="T_a712e_row1_col10" class="data row1 col10" >74.47</td>
      <td id="T_a712e_row1_col11" class="data row1 col11" >61.45</td>
    </tr>
    <tr>
      <th id="T_a712e_level0_row2" class="row_heading level0 row2" >Logistic Regression</th>
      <td id="T_a712e_row2_col0" class="data row2 col0" >70.81</td>
      <td id="T_a712e_row2_col1" class="data row2 col1" >70.38</td>
      <td id="T_a712e_row2_col2" class="data row2 col2" >73.11</td>
      <td id="T_a712e_row2_col3" class="data row2 col3" >0.44</td>
      <td id="T_a712e_row2_col4" class="data row2 col4" >76.87</td>
      <td id="T_a712e_row2_col5" class="data row2 col5" >63.75</td>
      <td id="T_a712e_row2_col6" class="data row2 col6" >70.81</td>
      <td id="T_a712e_row2_col7" class="data row2 col7" >70.38</td>
      <td id="T_a712e_row2_col8" class="data row2 col8" >73.11</td>
      <td id="T_a712e_row2_col9" class="data row2 col9" >0.44</td>
      <td id="T_a712e_row2_col10" class="data row2 col10" >76.87</td>
      <td id="T_a712e_row2_col11" class="data row2 col11" >63.75</td>
    </tr>
    <tr>
      <th id="T_a712e_level0_row3" class="row_heading level0 row3" >Random Forest</th>
      <td id="T_a712e_row3_col0" class="data row3 col0" >72.34</td>
      <td id="T_a712e_row3_col1" class="data row3 col1" >70.43</td>
      <td id="T_a712e_row3_col2" class="data row3 col2" >72.71</td>
      <td id="T_a712e_row3_col3" class="data row3 col3" >0.44</td>
      <td id="T_a712e_row3_col4" class="data row3 col4" >76.88</td>
      <td id="T_a712e_row3_col5" class="data row3 col5" >64.66</td>
      <td id="T_a712e_row3_col6" class="data row3 col6" >72.05</td>
      <td id="T_a712e_row3_col7" class="data row3 col7" >70.73</td>
      <td id="T_a712e_row3_col8" class="data row3 col8" >73.22</td>
      <td id="T_a712e_row3_col9" class="data row3 col9" >0.44</td>
      <td id="T_a712e_row3_col10" class="data row3 col10" >77.57</td>
      <td id="T_a712e_row3_col11" class="data row3 col11" >64.66</td>
    </tr>
    <tr>
      <th id="T_a712e_level0_row4" class="row_heading level0 row4" >XGBoost</th>
      <td id="T_a712e_row4_col0" class="data row4 col0" >72.10</td>
      <td id="T_a712e_row4_col1" class="data row4 col1" >70.64</td>
      <td id="T_a712e_row4_col2" class="data row4 col2" >73.16</td>
      <td id="T_a712e_row4_col3" class="data row4 col3" >0.44</td>
      <td id="T_a712e_row4_col4" class="data row4 col4" >77.49</td>
      <td id="T_a712e_row4_col5" class="data row4 col5" >64.50</td>
      <td id="T_a712e_row4_col6" class="data row4 col6" >71.73</td>
      <td id="T_a712e_row4_col7" class="data row4 col7" >71.09</td>
      <td id="T_a712e_row4_col8" class="data row4 col8" >73.22</td>
      <td id="T_a712e_row4_col9" class="data row4 col9" >0.44</td>
      <td id="T_a712e_row4_col10" class="data row4 col10" >77.83</td>
      <td id="T_a712e_row4_col11" class="data row4 col11" >65.79</td>
    </tr>
  </tbody>
</table>

# Visualisation for Rresult Comparisons
## Accuracy Comparison

![png](Capstone_Project_21042024_files/Capstone_Project_21042024_79_0.png)

## Confusion Matrix
    
![png](Capstone_Project_21042024_files/Capstone_Project_21042024_80_0.png)
    
## ROC Comparisons
    
![png](Capstone_Project_21042024_files/Capstone_Project_21042024_81_0.png)
    
## Precision Recall  Comparison
    
![png](Capstone_Project_21042024_files/Capstone_Project_21042024_82_0.png)

## Feature Importance Analysis
   
![png](Capstone_Project_21042024_files/Capstone_Project_21042024_83_0.png)
    
We then stacked the models to achieve a comprehensive predictive model taking advantage of the benefit of all the classifier models

## Deployment of clustering and predictive models using Joblib
# Conclusions
# Recommendations
# Limitations
# Future Works

