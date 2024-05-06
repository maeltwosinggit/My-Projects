import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import time

file_path = "C:/Users/uadrian/OneDrive - Averroes Data Science/Documents/My-Projects/Project 4 - Webscrape/bem_mechanical_PE_Practicing_Certificate_List.csv"

try:
	with open(file_path, 'r') as file:
		content = file.read()
	print(content)
except FileNotFoundError as e:
	print(f"FileNotFoundError: {e}")


# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)


#test looping
df1 = df.head(1)
df_result = []

for index, url in df1.iterrows():           
    print(f'No. {index +1}.{url['Google Link Redirect']}')
    response = requests.get(url['Google Link Redirect'])
    # Check if the request was successful (status code 200)
    if response.status_code == 200:

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
    #     # Find and extract the first search result link
    #     first_link = None
    #     search_results = soup.find_all('a')
    #     for link in search_results:
    #         href = link.get('href')
    #         if href and re.match(r'/url\?q=https://[a-zA-Z0-9-]+\.linkedin\.com', href):
    #             linkedin_profile = href.replace('/url?q=', '').split('&')[0]
    #             print(f'Linkedin Profile = {linkedin_profile}')
    #             df_result.append([{'URL':url['Name_name'],
    #                                           'LinkedIn Profile':linkedin_profile}])
                
                
        
    #     # # Find and extract the title of the first link
    #     # title = soup.title.string if soup.title else None

    # time.sleep(5)
    

# %%
df_result1 = pd.concat([pd.DataFrame(result) for result in df_result], ignore_index=True)
print(df_result1)

