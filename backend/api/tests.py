from django.contrib.staticfiles.testing import StaticLiveServerTestCase
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from django.test import TestCase
from api.models import FashionItem




class ModelTest(TestCase):
    #in case the api data is not available, we can use the fixtures to test the model
    fixtures = ['asos_data.json']
    def test_loaded_items(self):
        self.assertTrue(FashionItem.objects.exists())

    def test_fields(self):
        #test to check if the fields of the FashionItem model are loaded correctly and in the correct format
        item = FashionItem.objects.first()
        self.assertIsNotNone(item, "No FashionItem objects found.")
        if item:
            self.assertIsNotNone(item.product_id)
            self.assertIsNotNone(item.name)
            self.assertIsNotNone(item.price)
            self.assertIsInstance(item.is_trendy, bool)

class MainTest(StaticLiveServerTestCase):
    fixtures = ['asos_data.json']
    #this tests is for the main page and functionality of the app
    def setUp(self):
        self.driver = webdriver.Chrome()
        self.driver.implicitly_wait(5)

    def tearDown(self):
        self.driver.quit()


    def test_functionality(self):
        driver = self.driver
        #open the main page on localhost
        driver.get("http://localhost:3000")
        

        #first search: search for trainers which should return as a trendy item
        time.sleep(1)
        search_box = driver.find_element(By.TAG_NAME, "input")
        search_box.send_keys("trainers")
        time.sleep(2)
        search_box.send_keys(Keys.RETURN)
        time.sleep(2)

        WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.CLASS_NAME, "verdict-main"))
        )
        time.sleep(2)
        print("First search completed.")

        #click on the back button to go back to the main page and continue with the next search

        driver.find_element(By.CLASS_NAME, "back-btn").click()
        time.sleep(2)

        #second search: search for chelsea boots which should return as a not currently trendy item
        search_box = driver.find_element(By.TAG_NAME, "input")
        search_box.clear()
        search_box.send_keys("chelsea boots")
        time.sleep(2)
        search_box.send_keys(Keys.RETURN) 
        time.sleep(2)

        WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.CLASS_NAME, "verdict-main"))
        )
        print("Second search completed.")
        #click on the back button to go back to the main page and continue with the next search
        driver.find_element(By.CLASS_NAME, "back-btn").click()
        time.sleep(2)
        #fourth search: search for dresses which should return as a not a searchable item as it is not a shoe
        #and therefore would not have a verdict based on the data
        search_box = driver.find_element(By.TAG_NAME, "input")
        search_box.clear()
        search_box.send_keys("dresses")
        time.sleep(2)
        search_box.send_keys(Keys.RETURN)
        time.sleep(2)
        print("Third search completed.")
