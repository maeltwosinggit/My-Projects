import requests
from bs4 import BeautifulSoup
import re

# URL of the webpage
url = "https://www.google.com/search?q=Ir.+ALBERT+GOH+CHEE+KONG"

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find and extract the first search result link
    first_link = None
    search_results = soup.find_all('a')
    for link in search_results:
        href = link.get('href')
        if href and re.match(r'/url\?q=https://[a-zA-Z0-9-]+\.linkedin\.com', href):
            first_link = href.replace('/url?q=', '').split('&')[0]
            print(f'Linkedin Profile = {first_link}')
            
    
    # Find and extract the title of the first link
    title = soup.title.string if soup.title else None
