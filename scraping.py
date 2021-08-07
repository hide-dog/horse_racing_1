# horse_racing
#スクレイピングに必要なモジュール
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

# レースIDの作成
import itertools
YEAR = ['2020']
CODE = [str(num+1).zfill(2) for num in range(10)]
RACE_COUNT = ['01']
DAYS = ['01']
RACE_NUM = ['01']
race_ids = list(itertools.product(YEAR,CODE,RACE_COUNT,DAYS,RACE_NUM))
# サイトURLの作成
SITE_URL = ["https://race.netkeiba.com/race/result.html?race_id={}".format(''.join(race_id)) for race_id in race_ids]
import time #sleep用
import sys  #エラー検知用
import re   #正規表現
import numpy    #csv操作
import pandas as pd
result_df = pd.DataFrame()
#サイトURLをループしてデータを取得する
for sitename,race_id in zip(SITE_URL,race_ids):
    # 時間をあけてアクセスするように、sleepを設定する
    time.sleep(3)
    
    try:
        # スクレイピング対象の URL にリクエストを送り HTML を取得する
        res = requests.get(sitename)
        res.raise_for_status()  #URLが正しくない場合，例外を発生させる
        # レスポンスの HTML から BeautifulSoup オブジェクトを作る
        soup = BeautifulSoup(res.content, 'html.parser')
        # title タグの文字列を取得する
        title_text = soup.find('title').get_text()
        print(title_text)
        #順位のリスト作成
        Ranks = soup.find_all('div', class_='Rank')
        Ranks_list = []
        for Rank in Ranks:
            Rank = Rank.get_text()
            #リスト作成
            Ranks_list.append(Rank)
        #馬名取得
        Horse_Names = soup.find_all('span', class_='Horse_Name')
        Horse_Names_list = []
        for Horse_Name in Horse_Names:
            #馬名のみ取得(lstrip()先頭の空白削除，rstrip()改行削除)
            Horse_Name = Horse_Name.get_text().lstrip().rstrip('\n')
            #リスト作成
            Horse_Names_list.append(Horse_Name)
        #人気取得
        Ninkis = soup.find_all('span', class_='OddsPeople')
        Ninkis_list = []
        for Ninki in Ninkis:
            Ninki = Ninki.get_text()
            #リスト作成
            Ninkis_list.append(Ninki)
        #枠取得
        Wakus = soup.find_all('td', class_=re.compile("Num Waku"))
        Wakus_list = []
        for Waku in Wakus:
            Waku = Waku.get_text().replace('\n','')
            #リスト作成
            Wakus_list.append(Waku)
        #コース,距離取得
        Distance_Course = soup.find_all('span')
        Distance_Course = re.search(r'.[0-9]+m', str(Distance_Course))
        Course = Distance_Course.group()[0]
        Distance = re.sub("\\D", "", Distance_Course.group())
        df = pd.DataFrame({
            'レースID':''.join(race_id),
            '順位':Ranks_list,
            '枠':Wakus_list,
            '馬名':Horse_Names_list,
            'コース':Course,
            '距離':Distance,
            '人気':Ninkis_list,
        })
        
        result_df = pd.concat([result_df,df],axis=0)

    except:
        print(sys.exc_info())
        print("サイト取得エラー")

result_df.to_csv("input", encoding="utf-8")