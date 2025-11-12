"""Scrape EPL match data using Selenium for JavaScript-rendered content."""

import logging
import pandas as pd
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

from src.config import RAW_DATA_DIR, SCRAPED_SELENIUM_FILE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def scrape_epl_results_selenium() -> pd.DataFrame:
    """
    Scrape EPL match results using Selenium for JS-rendered content.
    
    Returns:
        DataFrame with columns: match_date, home_team, away_team, home_goals, away_goals, result
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    
    matches = []
    driver = None
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        url = "https://www.premierleague.com/results"
        driver.get(url)
        
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "matchFixtureContainer")))
        
        match_elements = driver.find_elements(By.CLASS_NAME, "matchFixtureContainer")
        
        for match_elem in match_elements[:20]:
            try:
                date_elem = match_elem.find_element(By.TAG_NAME, "time")
                date_str = date_elem.get_attribute("datetime")
                
                if not date_str:
                    continue
                
                match_date = pd.to_datetime(date_str).date()
                
                teams = match_elem.find_elements(By.CLASS_NAME, "teamName")
                if len(teams) < 2:
                    continue
                
                home_team = teams[0].text.strip()
                away_team = teams[1].text.strip()
                
                scores = match_elem.find_elements(By.CLASS_NAME, "score")
                if len(scores) >= 2:
                    home_goals = int(scores[0].text.strip())
                    away_goals = int(scores[1].text.strip())
                    
                    if home_goals > away_goals:
                        result = "H"
                    elif away_goals > home_goals:
                        result = "A"
                    else:
                        result = "D"
                else:
                    home_goals = None
                    away_goals = None
                    result = None
                
                matches.append({
                    "match_date": match_date,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "result": result
                })
            except Exception as e:
                logger.debug(f"Skipping match element: {e}")
                continue
        
    except Exception as e:
        logger.warning(f"Failed to scrape with Selenium: {e}")
        logger.info("Ensure ChromeDriver is installed and in PATH")
    
    finally:
        if driver:
            driver.quit()
    
    df = pd.DataFrame(matches)
    
    if not df.empty:
        df.to_csv(SCRAPED_SELENIUM_FILE, index=False)
        logger.info(f"Saved {len(df)} scraped matches to {SCRAPED_SELENIUM_FILE}")
    else:
        logger.warning("No matches scraped. Creating sample structure.")
        df = pd.DataFrame(columns=["match_date", "home_team", "away_team", "home_goals", "away_goals", "result"])
    
    return df


if __name__ == "__main__":
    df = scrape_epl_results_selenium()
    logger.info(f"Scraped {len(df)} matches")

