from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager


if __name__ == "__main__":

    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)

    # driver.get("https://www.google.com/")
    # print(driver.title)

    # search_box = driver.find_element_by_name("q")
    # search_box.send_keys("ChromeDriver")
    # search_box.submit()
    # print(driver.title)

    # driver.get("https://www.google.com/")
    driver.get("https://www.digest.elyza.ai/")
    print(driver.title)

    text_area = driver.find_element_by_xpath("//div/textarea")
    print(text_area)

    driver.save_screenshot("result/img/search_results.png")
    driver.quit()
