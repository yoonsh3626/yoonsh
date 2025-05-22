# yoonsh
import matplotlib.pyplot as plt
import pandas as pd

url = 'https://docs.google.com/spreadsheets/d/1V5d_wDlnsv6Cu1Fgx5l2RyOsGL0maZOI/export?format=csv&gid=501925475'
df = pd.read_csv(url)

print(df.head())  # 데이터 미리보기

def preprocess_data(df):
    # 필요한 컬럼 추출 및 이름 정리
    df = df[['날짜', '측정소명', '미세먼지', '초미세먼지']].copy()
    df.columns = ['date', 'district', 'pm10', 'pm25']

    # 결측치 확인
    print(df.isna().sum())

    # 결측치 처리
    df = df.dropna()  # 또는 df.fillna(df.median())

    # 이상치 제거 (도메인 기반 기준, 예: pm10 > 500 제거)
    df = df[df['pm10'] < 500]
    df = df[df['pm25'] < 250]

    # 자료형 변환
    df['date'] = pd.to_datetime(df['date'])
    df[['pm10', 'pm25']] = df[['pm10', 'pm25']].astype(float)

    return df


def create_features(df):
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

    def get_season(month):
        if month in [3,4,5]: return 'spring'
        elif month in [6,7,8]: return 'summer'
        elif month in [9,10,11]: return 'autumn'
        else: return 'winter'
    
    df['season'] = df['month'].apply(get_season)
    return df


def classify_pm10(df):
    def get_grade(val):
        if val <= 30: return 'good'
        elif val <= 80: return 'normal'
        elif val <= 150: return 'bad'
        else: return 'worse'
    
    df['pm_grade'] = df['pm10'].apply(get_grade)
    return df



def analyze(df):
    print(f"\n[4-1] 전체 PM10 평균: {df['pm10'].mean():.2f}")

    max_row = df[df['pm10'] == df['pm10'].max()]
    print(f"\n[5-1] PM10 최댓값 발생일 및 구:\n{max_row[['date','district','pm10']]}")

    top5_districts = df.groupby('district')['pm10'].mean().sort_values(ascending=False).head(5).reset_index()
    top5_districts.columns = ['district', 'avg_pm10']
    print(f"\n[6-2] 구별 평균 PM10 상위 5개:\n{top5_districts}")

    seasonal_avg = df.groupby('season')[['pm10', 'pm25']].mean().sort_values(by='pm10').reset_index()
    seasonal_avg.columns = ['season', 'avg_pm10', 'avg_pm25']
    print(f"\n[7-2] 계절별 평균 PM10/PM25:\n{seasonal_avg}")

    grade_freq = df['pm_grade'].value_counts().reset_index()
    grade_freq.columns = ['pm_grade', 'n']
    grade_freq['pct'] = (grade_freq['n'] / len(df) * 100).round(2)
    print(f"\n[8-2] PM10 등급 분포:\n{grade_freq}")

    good_pct_by_district = df[df['pm_grade'] == 'good'].groupby('district').size().reset_index(name='n')
    total_by_district = df.groupby('district').size().reset_index(name='total')
    good_rate = pd.merge(good_pct_by_district, total_by_district, on='district')
    good_rate['pct'] = (good_rate['n'] / good_rate['total'] * 100).round(2)
    good_top5 = good_rate.sort_values(by='pct', ascending=False).head(5)
    print(f"\n[9-2] good 등급 비율 상위 5개 구:\n{good_top5[['district', 'n', 'pct']]}")


def plot_pm10_trend(df):
    plt.figure(figsize=(12,6))
    sns.lineplot(data=df, x='date', y='pm10')
    plt.title('Daily Trend of PM10 in Seoul, 2019')
    plt.ylabel('PM10')
    plt.xlabel('Date')
    plt.tight_layout()
    plt.show()

def plot_seasonal_grade_distribution(df):
    temp = df.groupby(['season', 'pm_grade']).size().reset_index(name='n')
    total = df.groupby('season').size().reset_index(name='total')
    temp = pd.merge(temp, total, on='season')
    temp['pct'] = (temp['n'] / temp['total']) * 100

    plt.figure(figsize=(10,6))
    sns.barplot(data=temp, x='season', y='pct', hue='pm_grade',
                order=['spring', 'summer', 'autumn', 'winter'],
                hue_order=['good', 'normal', 'bad', 'worse'])
    plt.title('Seasonal Distribution of PM10 Grades in Seoul, 2019')
    plt.ylabel('Percentage (%)')
    plt.tight_layout()
    plt.show()
