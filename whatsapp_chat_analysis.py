import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import zipfile
import re
from collections import Counter
from nltk.corpus import stopwords

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
st.title('WhatsApp Chat Analysis Developed by Imad ud din khattak')

# Instructions for downloading the WhatsApp chat ZIP file
st.markdown("""
### How to Download Your WhatsApp Chat as a ZIP File:

1. Open WhatsApp on your phone.
2. Go to the chat you want to export.
3. Tap the three dots in the upper right corner.
4. Select **More**.
5. Tap **Export Chat**.
6. Choose **Without Media** to get a smaller file size.
7. Save or email the ZIP file to yourself.
8. Upload the ZIP file here to analyze your chat data.
""")

uploaded_file = st.file_uploader("Upload your WhatsApp chat ZIP file", type="zip")

if uploaded_file is not None:
    df = process_chat(uploaded_file)

    st.header('Questions Answered by the Visualizations')
    
    st.subheader('1. What time of day are messages sent the most?')
    plt.figure(figsize=(10, 6))
    sns.set_palette("rocket")  # Sets a color palette
    sns.countplot(x='Hour', data=df, palette='rocket')
    plt.title('Messages Sent by Hour of the Day', fontsize=16, weight='bold')
    plt.xlabel('Hour of the Day', fontsize=12)
    plt.ylabel('Number of Messages', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(plt)

    st.subheader('2. How does the number of messages vary over time?')
    daily_messages = df.groupby('Date').size()
    plt.figure(figsize=(10, 6))
    daily_messages.plot(kind='line', marker='o', linestyle='-', color='blue')
    plt.title('Message Frequency Over Time', fontsize=16, weight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Number of Messages', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(plt)

    st.subheader('3. On which days of the week are the most messages sent?')
    df['DayOfWeek'] = df['Date'].dt.day_name()
    plt.figure(figsize=(10, 6))
    sns.set_palette("pastel")  # Sets a pastel color palette
    sns.countplot(x='DayOfWeek', data=df, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], palette='pastel')
    plt.title('Messages Sent by Day of the Week', fontsize=16, weight='bold')
    plt.xlabel('Day of the Week', fontsize=12)
    plt.ylabel('Number of Messages', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(plt)

    st.subheader('4. What are the most common words used in the chat?')
    text = ' '.join(message for message in df['Message'])
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Most Common Words', fontsize=16, weight='bold')
    st.pyplot(plt)

    st.subheader('5. Who are the most active participants in the chat?')
    sender_counts = df['Sender'].value_counts()
    plt.figure(figsize=(12, 8))
    sns.barplot(x=sender_counts.values, y=sender_counts.index, palette='coolwarm')
    plt.title('Number of Messages by Sender', fontsize=16, weight='bold')
    plt.xlabel('Number of Messages', fontsize=12)
    plt.ylabel('Sender', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(plt)

    stop_words = set(stopwords.words('english'))
    st.subheader('6. What are the most common words used by a specific sender?')
    sender_name = st.selectbox('Select a sender', df['Sender'].unique())
    sender_messages = df[df['Sender'] == sender_name]['Message']
    all_text = " ".join(sender_messages)
    words = [word.lower() for word in all_text.split() if word.lower() not in stop_words and word.isalpha()]
    word_freq = Counter(words)
    plt.figure(figsize=(10, 6))
    plt.bar(*zip(*word_freq.most_common(10)), color='skyblue')
    plt.xlabel('Words', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Top 10 Most Common Words Used by {sender_name}', fontsize=16, weight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(plt)

    st.subheader('7. How do message lengths vary?')
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Message_Length'], bins=50, kde=True, color='purple')
    plt.xlabel('Message Length', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Message Lengths', fontsize=16, weight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(plt)

    st.subheader('8. What are the top 10 most active days?')
    most_active_days = daily_messages.nlargest(10)
    formatted_dates = most_active_days.index.strftime('%Y-%m-%d')
    plt.figure(figsize=(12, 8))
    sns.barplot(x=formatted_dates, y=most_active_days.values, palette="viridis")
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Number of Messages', fontsize=12)
    plt.title('Top 10 Most Active Days', fontsize=16, weight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(plt)

# Running the Streamlit app
if __name__ == "__main__":
    st.sidebar.info("Upload your WhatsApp chat zip file to generate visualizations")
    st.markdown("This app generates various visualizations from your WhatsApp chat data.")
