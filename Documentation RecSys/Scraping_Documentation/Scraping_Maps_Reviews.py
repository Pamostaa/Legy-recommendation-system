import os
import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException
import time
import random
from fuzzywuzzy import fuzz


def read_scraped_restaurants(file_name='dakar_reviews.csv'):
    """ Read the already scraped restaurants from the CSV file """
    if os.path.exists(file_name):
        with open(file_name, 'r', newline='', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            return {row[0] for row in reader}  # Store the restaurant names in a set for fast lookup
    return set()


def scroll_reviews(driver, max_scrolls=1000):
    try:
        time.sleep(2)
        review_panel = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="QA0Szd"]/div/div/div[1]/div[3]/div/div[1]/div/div/div[3]'))
        )
        print("✅ Correct review container located.")
        previous_count = 0
        retries = 0

        for i in range(max_scrolls):
            try:
                driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight;", review_panel)
                time.sleep(random.uniform(2.5, 4))
                review_blocks = driver.find_elements(By.XPATH, '//div[contains(@class, "jftiEf")]')
                current_count = len(review_blocks)
                print(f"📦 Scroll {i+1}/{max_scrolls} — Loaded reviews: {current_count}")
                if current_count == previous_count:
                    retries += 1
                    if retries > 4:
                        print("🚧 No more new reviews found.")
                        break
                else:
                    retries = 0
                    previous_count = current_count
            except StaleElementReferenceException:
                print(f"♻️ Stale element at scroll {i+1} — recovering.")
                review_panel = driver.find_element(By.XPATH, '//*[@id="QA0Szd"]/div/div/div[1]/div[3]/div/div[1]/div/div/div[3]')
                continue
    except Exception as e:
        print(f"⚠️ Error while scrolling reviews: {e}")


def scroll_sidebar_until_all_loaded(driver, max_scrolls=50):
    sidebar = driver.find_element(By.XPATH, '//div[@role="feed"]')
    previous_count = 0
    retries = 0
    for _ in range(max_scrolls):
        driver.execute_script("arguments[0].scrollBy(0, 1000);", sidebar)
        time.sleep(random.uniform(1.5, 3))
        restaurants = driver.find_elements(By.XPATH, '//div[contains(@class, "Nv2PK") and contains(@class, "THOPZb")]')
        current_count = len(restaurants)
        if current_count == previous_count:
            retries += 1
            if retries > 3:
                break
        else:
            retries = 0
            previous_count = current_count
    print(f"✅ Total restaurants loaded: {current_count}")
    return [r.text.split("\n")[0].strip() for r in restaurants if r.text.strip()]


def find_and_click_restaurant_by_name(driver, target_name, max_attempts=30):
    try:
        sidebar = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, '//div[@role="feed"]')))
        
        for attempt in range(max_attempts):
            cards = driver.find_elements(By.XPATH, '//div[contains(@class, "Nv2PK") and contains(@class, "THOPZb")]')
            for card in cards:
                try:
                    title_elem = card.find_element(By.CLASS_NAME, "qBF1Pd")
                    title = title_elem.text.strip().lower()
                    if fuzz.ratio(target_name.strip().lower(), title) > 80:  # Fuzzy matching
                        driver.execute_script("arguments[0].scrollIntoView();", card)
                        WebDriverWait(driver, 10).until(EC.visibility_of(card))  # Ensure the element is visible
                        time.sleep(random.uniform(1, 2))
                        card.click()
                        return True
                except NoSuchElementException as e:
                    print(f"⚠️ Could not locate restaurant card: {e}")
                    continue
            # Refetch the list of restaurant cards before the next scroll
            driver.execute_script("arguments[0].scrollBy(0, 500);", sidebar)
            time.sleep(random.uniform(1.5, 2))
        print(f"❌ Could not find: {target_name}")
        return False
    except Exception as e:
        print(f"❌ Error clicking on {target_name}: {e}")
        return False


def scrape_dakar_reviews():
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-extensions")
    options.add_argument("--remote-debugging-port=9222")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    profile_path = os.path.join(os.getcwd(), "chrome_profile")
    options.add_argument(f"--user-data-dir={profile_path}")

    driver = webdriver.Chrome(
        service=ChromeService(executable_path="C:/chromedriver-win64/chromedriver.exe"),
        options=options
    )

    url = "https://www.google.tn/maps/search/restaurants+dakar/@14.710898,-17.4725377,13.58z?hl=fr"
    driver.get(url)
    time.sleep(10)

    with open('dakar_reviews.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Restaurant', 'Rating', 'User', 'User Rating', 'Review'])

    # Read already scraped restaurants from CSV
    already_scraped = read_scraped_restaurants()

    restaurant_names = scroll_sidebar_until_all_loaded(driver)
    print(f"✅ Cached {len(restaurant_names)} restaurant names")

    for index, name in enumerate(restaurant_names):
        # Skip restaurants already scraped
        if name in already_scraped:
            print(f"🚫 Skipping already scraped restaurant: {name}")
            continue

        print(f"\n👉 Visiting restaurant {index + 1} of {len(restaurant_names)}: {name}")
        
        # Click directly on the next restaurant in the sidebar
        if not find_and_click_restaurant_by_name(driver, name):
            continue

        time.sleep(4)
        try:
            rating = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.XPATH, '//span[@class="MW4etd"]/span[@aria-hidden="true"]'))
            ).text
        except:
            try:
                rating = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.XPATH, '//div[@class="F7nice "]/span/span'))
                ).text
            except:
                rating = "No rating"
        print(f"⭐ Rating: {rating}")

        try:
            review_tab = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, '//button[contains(@aria-label, "Avis")]'))
            )
            review_tab.click()
            print("🟢 Clicked 'Avis' button.")
            time.sleep(2)

            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="QA0Szd"]/div/div/div[1]/div[3]/div/div[1]/div/div/div[3]'))
            )
            scroll_reviews(driver)

        except Exception as e:
            print(f"⚠️ No review tab or reviews for {name}: {e}")
            driver.back()
            time.sleep(3)
            continue

        review_blocks = driver.find_elements(By.XPATH, '//div[contains(@class, "jftiEf")]')
        print(f"📝 Found {len(review_blocks)} reviews for {name}")

        for review in review_blocks:
            try:
                user = review.find_element(By.CLASS_NAME, 'd4r55').text
                user_rating = review.find_element(By.CLASS_NAME, 'kvMYJc').get_attribute('aria-label')
                try:
                    comment = review.find_element(By.CLASS_NAME, 'wiI7pd').text
                except:
                    comment = ""
                with open('dakar_reviews.csv', 'a', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow([name, rating, user, user_rating, comment])
                print(f"✅ Saved review from: {user}")
            except Exception as e:
                print(f"⚠️ Review scrape error: {e}")
                continue

    driver.quit()
    print("\n✅ All reviews scraped successfully.")


# Start it
scrape_dakar_reviews()
