import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import zipfile
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Download NLTK data
nltk.download('stopwords')

# Function to process the uploaded file
def process_chat(file):
    chat_text = ""
    
    with zipfile.ZipFile(file, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.endswith('.txt'):
                with zip_ref.open(file_name) as f:
                    chat_text = f.read().decode('utf-8')
    
    dates = []
    times = []
    senders = []
    messages = []
    
    pattern = r'(\d{1,2}/\d{1,2}/\d{4}), (\d{1,2}:\d{2})\s?[apAP][mM] - (.*?): (.*)'
    matches = re.findall(pattern, chat_text)
    
    for match in matches:
        dates.append(match[0])
        times.append(match[1])
        senders.append(match[2])
        messages.append(match[3])
    
    df = pd.DataFrame({
        'Date': dates,
        'Time': times,
        'Sender': senders,
        'Message': messages
    })
    
    df = df[~df['Message'].isin(['Missed voice call', '<Media omitted>'])]
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df['Hour'] = df['Time'].dt.hour
    df['Message_Length'] = df['Message'].apply(len)
    
    return df

# Streamlit app layout
st.title('WhatsApp Chat Analysis')

uploaded_file = st.file_uploader("Upload your WhatsApp chat ZIP file", type="zip")

if uploaded_file is not None:
    df = process_chat(uploaded_file)

    st.header('Questions Answered by the Visualizations')
    
    st.subheader('1. What time of day are messages sent the most?')
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Hour', data=df, color='purple')
    plt.title('Messages Sent by Hour of the Day')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Messages')
    st.pyplot(plt)

    st.subheader('2. How does the number of messages vary over time?')
    daily_messages = df.groupby('Date').size()
    plt.figure(figsize=(10, 6))
    daily_messages.plot(kind='line', color='blue')
    plt.title('Message Frequency Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Messages')
    st.pyplot(plt)

    st.subheader('3. On which days of the week are the most messages sent?')
    df['DayOfWeek'] = df['Date'].dt.day_name()
    plt.figure(figsize=(10, 6))
    sns.countplot(x='DayOfWeek', data=df, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], palette='viridis')
    plt.title('Messages Sent by Day of the Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('Number of Messages')
    st.pyplot(plt)

    st.subheader('4. What are the most common words used in the chat?')
    text = ' '.join(message for message in df['Message'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

    st.subheader('5. Who are the most active participants in the chat?')
    sender_counts = df['Sender'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sender_counts.index, y=sender_counts.values, palette='coolwarm')
    plt.title('Number of Messages by Sender')
    plt.xlabel('Sender')
    plt.ylabel('Number of Messages')
    plt.xticks(rotation=90)
    st.pyplot(plt)

    st.subheader('6. What are the most common words used by a specific sender?')
    sender_name = st.selectbox('Select a sender', df['Sender'].unique())
    sender_messages = df[df['Sender'] == sender_name]['Message']
    all_text = " ".join(sender_messages)
    words = [word.lower() for word in all_text.split() if word.lower() not in stop_words and word.isalpha()]
    word_freq = Counter(words)
    plt.figure(figsize=(10, 6))
    plt.bar(*zip(*word_freq.most_common(10)), color='blue')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title(f'Top 10 Most Common Words Used by {sender_name}')
    plt.xticks(rotation=45)
    st.pyplot(plt)

    st.subheader('7. How do message lengths vary?')
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Message_Length'], bins=50, kde=True, color='purple')
    plt.xlabel('Message Length')
    plt.ylabel('Frequency')
    plt.title('Distribution of Message Lengths')
    st.pyplot(plt)

    st.subheader('8. What are the top 10 most active days?')
    most_active_days = daily_messages.nlargest(10)
    formatted_dates = most_active_days.index.strftime('%Y-%m-%d')
    plt.figure(figsize=(10, 6))
    sns.barplot(x=formatted_dates, y=most_active_days.values, palette="viridis")
    plt.xlabel('Date')
    plt.ylabel('Number of Messages')
    plt.title('Top 10 Most Active Days')
    plt.xticks(rotation=45)
    st.pyplot(plt)

# Running the Streamlit app
if __name__ == "__main__":
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.sidebar.info("Upload your WhatsApp chat zip file to generate visualizations")
    st.markdown("This app generates various visualizations from your WhatsApp chat data.")
