import requests
from bs4 import BeautifulSoup


def get_stock_price(ticker):
    r = requests.get('https://finance.yahoo.com/quote/'+ticker+'?p='+ticker+'&.tsrc=fin-srch', timeout=5)
    soup = BeautifulSoup(r.text, features="html.parser")
    price = soup.find_all('div', {'class': 'My(6px) Pos(r) smartphone_Mt(6px)'})[0].find('span').text
    return price


while True:
    print('Current stock value : '+get_stock_price("UBER"))
