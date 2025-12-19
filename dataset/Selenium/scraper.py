from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import ElementNotInteractableException

from selenium.webdriver.support import expected_conditions as EC
import time

from selenium.webdriver.chrome.options import Options

timeout = 10

# Setup Chrome options for automatic file downloading
chrome_options = Options()
download_folder = "/path/for/download/folder"

prefs = {
    "download.default_directory": download_folder,
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True
}

chrome_options.add_experimental_option("prefs", prefs)

# Initialize the Chrome driver with options
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# Replace the URL below with the URL where you need to log in
login_url = 'https://www.dartfish.tv/Login?LRedirect=https%253a%252f%252fwww.dartfish.tv%252fVideos%253fCR%253dp120133'
driver.get(login_url)

# Wait for the page to load
time.sleep(2)

# Replace these with the actual ID or name of the login form's input fields
username_field_id = 'ctl00_MC_Usn'
password_field_id = 'ctl00_MC_PwdReader_Password'

# Replace these with your actual login credentials
your_username = 'username'
your_password = 'password'

# Find the login inputs and submit button, then input your credentials and submit
driver.find_element(By.ID, username_field_id).send_keys(your_username)
driver.find_element(By.ID, password_field_id).send_keys(your_password)
driver.find_element(By.ID, password_field_id).send_keys(Keys.RETURN)


time.sleep(30)  # Adjust based on your internet speed and page response time

def click_download_button():
    try:
        # Wait for the download button to be clickable
        download_button = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="ctl00_MCL_HQVidFileList_DFileRepeater_ctl00_FileLink"]'))
        )
        # Click the button
        download_button.click()
        return True
    except TimeoutException:
        print("Download button not found within the time limit.")
        return False
    
def click_download_button_csv():
    try:
        # Wait for the download button to be clickable
        element = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.ID, "ctl00_MCL_ExportFileList_ExportMatch"))
        )
        driver.execute_script("arguments[0].scrollIntoView(true);", element)
        element.click()
        
        return True
    except TimeoutException:
        print("Download button not found within the time limit.")
        return False
    
def safe_click(selector, retries=3):
    attempts = 0
    while attempts < retries:
        try:
            element = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.ID, selector))
            )
            driver.execute_script("arguments[0].scrollIntoView(true);", element)
            element.click()
            return True
        except ElementNotInteractableException:
            print(f"Element not interactable, retrying... Attempt {attempts + 1}")
            time.sleep(2)  # Adjust delay as needed
            attempts += 1
    return False



# Find all video titles by class name and loop through them
video_titles = driver.find_elements(By.CSS_SELECTOR, ".minivideo__title a[title]")

for index, video_title in enumerate(video_titles):
    if index < 128:
        continue
    # Since clicking will navigate away, find elements again each time to avoid stale element reference error
    video_titles = driver.find_elements(By.CSS_SELECTOR, ".minivideo__title a[title]")
    video_title = video_titles[index]
    
    # Print the video title for debug purposes
    print(f"Accessing: {video_title.get_attribute('title')}")
    driver.execute_script("arguments[0].scrollIntoView(true);", video_title)
    # Click on the video title to go to the video's page
    video_title.click()
    
    
    time.sleep(2)  
    download_selector = 'ctl00_MCL_DownBt'
    if not safe_click(download_selector):
        print("Failed to click download_selector retries. Handling error...")
    
    # print("==================1111==================")
    # WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, download_selector))).click()
    # print("==================2222==================")
    # driver.find_element(By.ID, download_selector).click()

    # Usage
    if not safe_click("ctl00_MCL_HQVidFileList_DFileRepeater_ctl00_FileLink"):
        print("Failed to click after retries. Handling error...")
    # if not click_download_button():
    #     # Handle failure (e.g., retry, log, or correct)
    #     print(f"Failed to click download for {video_title.text}. Moving to the next video.")

    time.sleep(2)
    csv_selector = 'ctl00_MCL_CsvPaneLink'
    if not safe_click(csv_selector):
        print("Failed to click csv_selector retries. Handling error...")
    # print("==================3333==================")
    # WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, csv_selector))).click()
    # print("==================4444==================")
    # driver.find_element(By.ID, csv_selector).click()
    time.sleep(1)
    if not safe_click("ctl00_MCL_ExportFileList_ExportMatch"):
        print("Failed to click after retries. Handling error...")
    # if not click_download_button_csv():
    #     # Handle failure (e.g., retry, log, or correct)
    #     print(f"Failed to click download for {video_title.text}. Moving to the next ccsv.")
    
    time.sleep(2)
    
    driver.get("https://www.dartfish.tv/Videos?CR=p120133")
    
    time.sleep(5)
    
driver.quit()


