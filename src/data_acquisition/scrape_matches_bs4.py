"""Scrape EPL match data using BeautifulSoup."""

import logging
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime

from src.config import RAW_DATA_DIR, SCRAPED_BS4_FILE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def scrape_epl_results_bs4() -> pd.DataFrame:
    """
    Scrape EPL match results using BeautifulSoup.
    
    Returns:
        DataFrame with columns: match_date, home_team, away_team, home_goals, away_goals, result
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    url = "https://www.premierleague.com/results"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    matches = []
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        match_elements = soup.find_all("div", class_="matchFixtureContainer")
        
        for match_elem in match_elements[:20]:
            try:
                date_elem = match_elem.find("time")
                if not date_elem:
                    continue
                
                date_str = date_elem.get("datetime", "")
                if date_str:
                    match_date = pd.to_datetime(date_str).date()
                else:
                    continue
                
                teams = match_elem.find_all("span", class_="teamName")
                if len(teams) < 2:
                    continue
                
                home_team = teams[0].get_text(strip=True)
                away_team = teams[1].get_text(strip=True)
                
                scores = match_elem.find_all("span", class_="score")
                if len(scores) >= 2:
                    home_goals = int(scores[0].get_text(strip=True))
                    away_goals = int(scores[1].get_text(strip=True))
                    
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
            except (ValueError, AttributeError) as e:
                logger.debug(f"Skipping match element due to error: {e}")
                continue
        
        time.sleep(1)
        
    except requests.RequestException as e:
        logger.warning(f"Failed to scrape data: {e}")
        logger.info("Creating empty DataFrame as fallback")
    
    df = pd.DataFrame(matches)
    
    if not df.empty:
        df.to_csv(SCRAPED_BS4_FILE, index=False)
        logger.info(f"Saved {len(df)} scraped matches to {SCRAPED_BS4_FILE}")
    else:
        logger.warning("No matches scraped. Creating sample structure.")
        df = pd.DataFrame(columns=["match_date", "home_team", "away_team", "home_goals", "away_goals", "result"])
    
    return df


if __name__ == "__main__":
    df = scrape_epl_results_bs4()
    logger.info(f"Scraped {len(df)} matches")

