import streamlit as st 
import pandas as pd 
import base64
import matplotlib.pyplot as plt 
import numpy as np 
import yfinance as yf 
from selenium import webdriver
import time


def get_driver():
  options = webdriver.ChromeOptions()
  options.add_argument('disable-infobars')
  options.add_argument('start-maximized')
  options.add_argument('disable-dev-shm-usage')
  options.add_argument('no-sandbox')
  options.add_experimental_option('excludeSwitches', ['enable-automation'])
  options.add_argument('disable-blink-features=AutomationControlled')
  driver = webdriver.Chrome(options=options)
  driver.get('https://automated.pythonanywhere.com/')
  return driver

def clean_text(text):
  """Extract onlt the temperature from the text"""
  output = float(text.split(': ')[1])
  return str(output)

def main_one():
  driver = get_driver()
  time.sleep(2)
  element = driver.find_element(by='xpath', value='/html/body/div[7]/div/div[10]/div[2]/div/div[2]/div[2]/div/div/div/div/div/div[2]/div[2]/div[2]/div/div/div/div/div/div/div/g-card-section/div/g-card-section/div[2]/div[1]/span[1]/span/span')
  return clean_text(element.text)

def main_two():
  driver = get_driver()
  time.sleep(2)
  element = driver.find_element(by='xpath', value='/html/body/div[7]/div/div[10]/div[2]/div/div[2]/div[2]/div/div/div/div/div/div[2]/div[2]/div[2]/div/div/div/div/div/div/div/g-card-section/div/g-card-section/div[2]/div[1]/span[2]/span[1]')
  return clean_text(element.text)

pts = print(main_one())

delta = print(main_two())

col1 = st.columns(1)

col1.metric("S&P 500 Pts.", pts, delta)









st.title('S&P 500 App')

st.markdown("""
This app retrieves the list of the **S&P 500** (from Wikipedia) and its corresponding **stock closing price** (year-to-date)!
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib and yfinance
* **Data source:** [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies).
""")

st.sidebar.header('User Input Features')

# Web scraping of S&P 500 Data

@st.cache
def load_table_one():
  url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
  html = pd.read_html(url, header = 0)
  df = html[0]
  return df

df = load_table_one()
sector = df.groupby('GICS Sector')

# Sidebar - Sector selection
sorted_sector_unique = sorted( df['GICS Sector'].unique() )
selected_sector = st.sidebar.multiselect('Sector', sorted_sector_unique, sorted_sector_unique)

# Filtering Data
df_selected_sector = df[ (df['GICS Sector'].isin(selected_sector)) ]

st.header('Display Companies in Selected Sector')
st.write('Data Dimension: ' + str(df_selected_sector.shape[0]) + ' rows and ' + str(df_selected_sector.shape[1]) + ' columns.')
st.dataframe(df_selected_sector)


# Download S&P500 dataframe

def filedownload(df):
  csv = df.to_csv(index=False)
  b64 = base64.b64encode(csv.encode()).decode() # does the string/byte conversion
  href = f'<a href="data:file/csv;base64,{b64}" download="SP500.csv">Download CSV File</a>'
  return href

st.markdown(filedownload(df_selected_sector), unsafe_allow_html=True)


data = yf.download(
  tickers = list(df_selected_sector[:10].Symbol),
  period = "ytd",
  interval = "1d",
  group_by = 'ticker',
  auto_adjust = True,
  prepost = True,
  threads = True,
  proxy = None
  )

st.set_option('deprecation.showPyplotGlobalUse', False)

def price_plot(symbol):
  df = pd.DataFrame(data[symbol].Close)
  df['Date'] = df.index
  plt.fill_between(df.Date, df.Close, color='skyblue', alpha=0.3)
  plt.plot(df.Date, df.Close, color='skyblue', alpha=0.8)
  plt.xticks(rotation=90)
  plt.title(symbol, fontweight='bold')
  plt.xlabel('Date', fontweight='bold')
  plt.ylabel('Closing Price', fontweight='bold')
  return st.pyplot()



num_company = st.slider('Number of Companies', 1, 10)

if st.button('Show Plots'):
  st.header('Stock Closing Price')
  for i in list(df_selected_sector.Symbol)[:num_company]:
    price_plot(i)