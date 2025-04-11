import re
import pandas as pd

def preprocess(data):
    pattern = r"\[(\d{2}/\d{2}/\d{4}, \d{2}:\d{2}:\d{2})\] ([\w\s]+): (.*)"

    matches = re.findall(pattern, data)

    dates = [match[0] for match in matches]  # Extract all timestamps
    users = [match[1] for match in matches]  # Extract all users
    messages = [match[2] for match in matches]  # Extract all name + message

    df = pd.DataFrame({'message_date': dates, 'user': users, 'user_message': messages})
    df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%Y, %H:%M:%S')
    df.rename(columns={'message_date': 'date'}, inplace=True)

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month_name()
    df['month_num']= df['date'].dt.month
    df['only_date']=df['date'].dt.date
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['second'] = df['date'].dt.second

    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))
    df['period'] = period

    return df

