# airbnb dataset
## Motivation
Nowadays, the usage of accommodation reservation platforms such as Airbnb is increasing. With the growth of the accommodation market, users have more options and users are exploring more options for accommodation. This has made transparency and prediction about the price of the accommodation more important. Reflecting these market trends, the project seeks to provide users and accommodation companies with information about the predicted prices and the factors that influence them. The project aims to meet these users' needs and help hosts efficiently utilize resources and set fair prices.

## Objective	
The main objective of this project is to forecast the price of an Airbnb accommodation through the Airbnb dataset, and to find features that additionally affect the price. It helps market analysis by allowing both customers and hosts to understand the factors that influence the pricing of the accommodation and the forecast prices. It helps customers choose accommodations and helps hosts efficiently operate the accommodations and set fair prices. Information can be provided to customers so that they can determine the average price of similar accommodations based on their accommodation selection conditions and prepare an appropriate budget. The host can provide the information necessary to determine the accommodation to register with Airbnb, and can decide whether to invest or not by calculating the return on investment based on the predicted price. It also helps market by using factors that affect prices.

## Data file download
1. Go to the kaggle page below and download it
https://www.kaggle.com/datasets/paramvir705/airbnb-data/code

## Data file setting
This project uses the Airbnb_Data.csv file.  
In order to import data correctly, the data file must be located in the following path:

### How to set the location of a data file
1. Download the Airbnb_Data.csv file.
2. Save the file to the c:/work/ directory. 
   - For Windows users: Create a work folder on drive C and put the Airbnb_Data.csv file in it.
   - For Mac or Linux users: You can adapt the project directory structure to your Windows environment, or modify the code to adapt the path of the data file to your local environment.

### Change the data path from the code
If you want to save the data in a different location, modify the next part of the file to change the path of the data file

- MultipleLinearRegression.py
  
df_origin = pd.read_csv('c:/work/Airbnb_Data.csv', encoding='utf-8')

- KNNClassification.py
  
data = pd.read_csv('c:/work/Airbnb_Data.csv')
## + Include data preprocessing in each python code in algorithm folder
