import os
from selenium import webdriver
from dotenv import load_dotenv
import csv
import time

import urllib.error
import urllib.request
import urllib.parse

import pandas as pd
import shutil


# results.csvの形式を少し変更
# 画像ファイル名のカラムを追加
def pre_processing_csv():
    df = pd.read_csv('results.csv')

    df_new = df.assign(image_file_name=df.image_url.str.split('/').str[-1])
    print(df_new)
    df_new = df_new.sort_values(by=['image_file_name'], ascending=True)
    print(df_new)

    df_new.to_csv('results2.csv', index=False)


# 画像をジャンル毎にフォルダ分け
def move_images():
    df = pd.read_csv('results2.csv')

    genres = df['genre'].unique()

    for genre in genres:
        dir_path = 'data/' + genre + '/'
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        extracted_df = df.query('genre == "%s"' % genre)

        for index, row in extracted_df.iterrows():
            image_path = './images/' + row['image_file_name']
            if os.path.exists(image_path):
                shutil.move(image_path, dir_path + row['image_file_name'])


def download_image(url, dst_path):
    try:
        data = urllib.request.urlopen(url).read()
        with open(dst_path, mode="wb") as f:
            f.write(data)
    except urllib.error.URLError as e:
        print(e)


def set_driver():
    load_dotenv('.env')
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')

    driver = webdriver.Chrome(executable_path=os.environ['PATH_OF_CHROME_DRIVER'], options=options)
    home_url = 'https://www.talent-databank.co.jp'

    driver.get(home_url)
    driver.set_window_size(2000, 2000)

    return driver


def data_extract(driver):
    image_elements = driver.find_elements_by_xpath('//*[@id="search-results"]/tbody/tr/td[1]/a/img')
    name_elements = driver.find_elements_by_xpath('//*[@id="search-results"]/tbody/tr/td[2]/a')
    furigana_elements = driver.find_elements_by_xpath('//*[@id="search-results"]/tbody/tr/td[3]')
    genre_elements = driver.find_elements_by_xpath('//*[@id="search-results"]/tbody/tr/td[4]')
    age_elements = driver.find_elements_by_xpath('//*[@id="search-results"]/tbody/tr/td[5]')
    for i, image_element in enumerate(image_elements):
        row = []
        image_url = image_element.get_attribute('src')
        name = name_elements[i].text
        furigana = furigana_elements[i].text
        genre = genre_elements[i].text.split()[0]
        age = age_elements[i].text

        dst_path = './images/' + image_url.split('/')[-1]
        download_image(image_url, dst_path)

        row.append(image_url)
        row.append(name)
        row.append(furigana)
        row.append(genre)
        row.append(age)
        writer.writerow(row)

        # print(i)
        # print(image_url)
        # print(name)
        # print(furigana)
        # print(genre)
        # print(age)
        # print(row)
    global count
    count += 1
    print(count)
    time.sleep(1)


if __name__ == '__main__':
    driver = set_driver()
    count = 0
    # 検索
    driver.find_element_by_xpath('//*[@id="talentsearch02"]/div[16]/input').click()
    with open('results.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')  # 改行コード（\n）を指定しておく
        writer.writerow(['image_url', 'name', 'furigana', 'genre', 'age'])

        data_extract(driver)
        # 次のページ（一回目のみ）
        driver.find_element_by_xpath('//*[@id="contentsBg"]/div/div[1]/span/a[5]').click()

        while driver.find_elements_by_xpath('//*[@id="contentsBg"]/div/div[1]/span/a[6]'):
            data_extract(driver)
            driver.find_element_by_xpath('//*[@id="contentsBg"]/div/div[1]/span/a[6]').click()

    driver.quit()
