import re
import pandas as pd

def preprocess(data):
    # Handle both formats:
    # 1. [dd/mm/yyyy, hh:mm:ss] Name: message (iPhone style)
    # 2. dd/mm/yyyy, hh:mm - Name: message (Android style)
    pattern_android = r'(\d{2}/\d{2}/\d{4}, \d{2}:\d{2}) - (.*?): (.*)'
    pattern_iphone = r'\[(\d{2}/\d{2}/\d{4}, \d{2}:\d{2}:\d{2})\] (.*?): (.*)'

    matches = re.findall(pattern_android, data)
    if not matches:
        matches = re.findall(pattern_iphone, data)
        time_format = '%d/%m/%Y, %H:%M:%S'
    else:
        time_format = '%d/%m/%Y, %H:%M'

    # Unpack matches
    dates = [match[0] for match in matches]
    users = [match[1] for match in matches]
    messages = [match[2] for match in matches]

    # Create DataFrame
    df = pd.DataFrame({'message_date': dates, 'user': users, 'user_message': messages})
    df['message_date'] = pd.to_datetime(df['message_date'], format=time_format, errors='coerce')
    df.rename(columns={'message_date': 'date'}, inplace=True)

    # Drop any rows with invalid dates
    df.dropna(subset=['date'], inplace=True)

    # Add time-based features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month_name()
    df['month_num'] = df['date'].dt.month
    df['only_date'] = df['date'].dt.date
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    # Add period
    period = []
    for hour in df['hour']:
        if hour == 23:
            period.append(f"{hour}-00")
        elif hour == 0:
            period.append("00-1")
        else:
            period.append(f"{hour}-{hour + 1}")
    df['period'] = period

    return df
