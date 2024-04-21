import requests
from bs4 import BeautifulSoup

# URL of the webpage
url = "http://bem.org.my/web/guest/professional-engineer-with-practising-competency?p_p_id=engineerresult_WAR_engineerdirectoryportlet&p_p_lifecycle=0&p_p_state=normal&p_p_mode=view&p_p_col_id=column-3&p_p_col_pos=1&p_p_col_count=3&_engineerresult_WAR_engineerdirectoryportlet_jspPage=%2Fjsps%2Fengineer-result%2Fview.jsp&_engineerresult_WAR_engineerdirectoryportlet_key=WEBPEPC&_engineerresult_WAR_engineerdirectoryportlet_name=&_engineerresult_WAR_engineerdirectoryportlet_dcipline=MECHANICAL&_engineerresult_WAR_engineerdirectoryportlet_regNumber=&_engineerresult_WAR_engineerdirectoryportlet_delta=200&_engineerresult_WAR_engineerdirectoryportlet_keywords=&_engineerresult_WAR_engineerdirectoryportlet_advancedSearch=false&_engineerresult_WAR_engineerdirectoryportlet_andOperator=true&_engineerresult_WAR_engineerdirectoryportlet_resetCur=false&_engineerresult_WAR_engineerdirectoryportlet_cur=1"

# Send a GET request to the URL
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
print(soup.body.text)
print(soup.find_all("td",class_= "table-cell"))

# # Check if the request was successful (status code 200)
# if response.status_code == 200:
#     # Parse the HTML content
#     soup = BeautifulSoup(response.text, 'html.parser')
    
#     # Now you can use BeautifulSoup to extract specific information from the webpage
#     # For example, to find all links on the page:
#     links = soup.find_all('a')
    
#     # Iterate over the links and print their href attributes
#     for link in links:
#         print(link.get('href'))
# else:
#     print("Failed to retrieve webpage. Status code:", response.status_code)
