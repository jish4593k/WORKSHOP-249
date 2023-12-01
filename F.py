import os
import time
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

def initialize_browser():
    chromedriver = "./chromedriver"
    os.environ["webdriver.chrome.driver"] = chromedriver
    return webdriver.Chrome(chromedriver)

def get_website_data(driver, search_query="rtx 2080"):
    thing_url = "https://craftmybox.com/placas-video"
    driver.get(thing_url)

    # Search
    searchbox = driver.find_element_by_css_selector("input[placeholder='Procurar placas de vídeo...']")
    searchbox.click()
    searchbox.send_keys(search_query)

    # Wait for the page to load
    time.sleep(5)

    # Get source code (HTML)
    html = driver.execute_script("return document.getElementsByTagName('html')[0].innerHTML")
    return BeautifulSoup(html, "html.parser")

def extract_product_information(soup):
    names = []
    memories = []
    prices = []

    
    for line in soup.findAll('div', attrs={'class': 'is-flex-widescreen is-vcentered'}):
        name_text = " ".join(line.text.split())
        names.append(name_text)

  
    for line in soup.findAll('td', attrs={'data-label': 'Memória'}):
        memory_text = " ".join(line.text.split())
        memories.append(memory_text)

  
    for line in soup.findAll('td', attrs={'data-label': 'Preço boleto'}):
        price_text = line.text.replace('R$', '').replace('.', '').replace(',', '.')
        price_text = " ".join(price_text.split())
        prices.append(price_text)

    return names, memories, prices

def save_to_csv(data, filename='craftmybox-TESTEFINAL.csv'):
    data.to_csv(filename, encoding='ISO-8859-1', index=False, header=False, mode='a')

def train_neural_network(data):
  
    X = tf.random.normal((len(data), 10))  # Assume 10 features for each product
    y = tf.random.uniform((len(data), 1), 0, 2, dtype=tf.int32)  # Binary labels (0 or 1)

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(10,)),
        layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    print("Neural network training completed.")

def perform_craftmybox_scraping_and_analysis(search_query="rtx 2080"):
    driver = initialize_browser()

    try:
        soup = get_website_data(driver, search_query)
        names, memories, prices = extract_product_information(soup)

        
        current_time = time.strftime("%d/%m/%Y %H:%M:%S")

        df = pd.DataFrame({
            'Data': [current_time] * len(names),
            'Produto': names,
            'Memória': memories,
            'Preço': prices
        })

        # Save to CSV
        save_to_csv(df)

    
        train_neural_network(df)

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        driver.quit()

if __name__ == "__main__":
    perform_craftmybox_scraping_and_analysis()

