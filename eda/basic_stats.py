from utils import *
import matplotlib.pyplot as plt
import seaborn as sbn

df = read_forum_json('../data/json/levergunscommunity.com.json')

id_count = df['author_id'].value_counts()
username_count = df['author_username'].value_counts()
location_count = df['author_location'].value_counts()
id_location_count = df.groupby(['author_id','author_location'])['author_id'].count()

df['time'] = pd.to_datetime(df.created_at, unit='ms')
df['time_hour'] = df['time'].apply(lambda x: x.hour)
df['time_weekday'] = df['time'].apply(lambda x: x.weekday())
df['time_year'] = df['time'].apply(lambda x: x.year)
df['time_month'] = df['time'].apply(lambda x: x.month)

hour_count = df.time_hour.value_counts(sort=False)
weekday_count = df.time_weekday.value_counts(sort=False)
year_count = df.time_year.value_counts(sort=False)
month_count = df.time_month.value_counts(sort=False)

fig, axes = plt.subplots(2, 2, figsize=(10, 10));

#plt.subplots_adjust(wspace=0.1, hspace=0.1);
hour_count.plot(subplots=True, ax=axes[0][0], legend=False, sharex=False, sharey=False,kind='bar');
weekday_count.plot(subplots=True, ax=axes[0][1], legend=False, sharex=False, sharey=False,kind='bar');
year_count.plot(subplots=True, ax=axes[1][0], legend=False, sharex=False, sharey=False,kind='bar');
month_count.plot(subplots=True, ax=axes[1][1], legend=False, sharex=False, sharey=False,kind='bar');


