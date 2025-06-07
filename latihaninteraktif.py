# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import openai
from datetime import datetime

# --- Setup Page ---
st.set_page_config(page_title="ZingPop Media Dashboard", layout="wide")
st.title("📊 ZingPop Media Intelligence Dashboard")

# --- Load Dataset ---
df = pd.read_csv("ZingPop.csv")
df['Date'] = pd.to_datetime(df['Date'])

# --- Sidebar Filters ---
st.sidebar.header("🔧 Filter Data")
start_date = st.sidebar.date_input("Start Date", df['Date'].min().date())
end_date = st.sidebar.date_input("End Date", df['Date'].max().date())

selected_platforms = st.sidebar.multiselect(
    "Select Platforms",
    options=df['Platform'].unique(),
    default=df['Platform'].unique()
)

filtered_df = df[
    (df['Date'].dt.date >= start_date) &
    (df['Date'].dt.date <= end_date) &
    (df['Platform'].isin(selected_platforms))
]

# --- Setup OpenAI Client via OpenRouter ---
api_key = st.secrets["api_keys"]["openrouter"]
client = openai.OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)

def get_insights(chart_title, df_context, question, model="mistralai/mistral-7b-instruct-v0.2"):
    context = df_context.to_csv(index=False)
    prompt = f"""
Anda adalah analis intelijen media profesional. Analisis dataset berikut untuk: {chart_title}.
DATA:
{context}
PERTANYAAN: {question}
Berikan 3 insight yang ringkas, berbobot, dan dapat ditindaklanjuti, semuanya dalam **bahasa Indonesia**.
"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content

# =======================
# SECTION 1: Sentiment
# =======================
with st.expander("📌 Sentiment Breakdown", expanded=True):
    sentiment_data = filtered_df['Sentiment'].value_counts().reset_index()
    sentiment_data.columns = ['Sentiment', 'Count']
    fig1 = px.bar(sentiment_data, x='Sentiment', y='Count', color='Sentiment', title="Sentiment Breakdown")
    st.plotly_chart(fig1, use_container_width=True)

with st.expander("🔍 AI Insight - Sentiment Breakdown"):
    st.write(get_insights(
        "Sentiment Breakdown",
        sentiment_data,
        "Apa insight paling menonjol dari distribusi sentimen terhadap ZingPop dan bagaimana hal ini dapat memengaruhi strategi media mereka?"
    ))

# =======================
# SECTION 2: Engagement Trend
# =======================
with st.expander("📈 Engagement Trend Over Time", expanded=True):
    engagement_trend = filtered_df.groupby('Date')['Engagements'].sum().reset_index()
    fig2 = px.line(engagement_trend, x='Date', y='Engagements', title="Engagement Trend Over Time")
    st.plotly_chart(fig2, use_container_width=True)

with st.expander("🔍 AI Insight - Engagement Trend"):
    st.write(get_insights(
        "Engagement Trend Over Time",
        engagement_trend,
        "Apa pola dan lonjakan keterlibatan audiens terhadap ZingPop dari waktu ke waktu?"
    ))

# =======================
# SECTION 3: Platform Engagement
# =======================
with st.expander("🧭 Platform Engagement", expanded=True):
    platform_data = filtered_df.groupby('Platform')['Engagements'].sum().reset_index().sort_values(by='Engagements', ascending=False)
    fig3 = px.bar(platform_data, x='Engagements', y='Platform', orientation='h', color='Platform', title="Platform Engagements")
    st.plotly_chart(fig3, use_container_width=True)

with st.expander("🔍 AI Insight - Platform Engagement"):
    st.write(get_insights(
        "Platform Engagements",
        platform_data,
        "Platform mana yang menunjukkan performa terbaik dalam kampanye ZingPop dan apa yang dapat disimpulkan dari hal ini?"
    ))

# =======================
# SECTION 4: Media Type Mix
# =======================
with st.expander("🧪 Media Type Mix", expanded=True):
    media_mix = filtered_df['Media_Type'].value_counts().reset_index()
    media_mix.columns = ['Media_Type', 'Count']
    fig4 = px.pie(media_mix, values='Count', names='Media_Type', title="Media Type Mix", hole=0.4)
    st.plotly_chart(fig4, use_container_width=True)

with st.expander("🔍 AI Insight - Media Type Mix"):
    st.write(get_insights(
        "Media Type Mix",
        media_mix,
        "Apa format media paling disukai dan bagaimana rekomendasi strategi konten untuk ZingPop?"
    ))

# =======================
# SECTION 5: Top Locations
# =======================
with st.expander("🌍 Top 5 Locations by Engagement", expanded=True):
    location_data = filtered_df.groupby('Location')['Engagements'].sum().reset_index().sort_values(by='Engagements', ascending=False).head(5)
    fig5 = px.bar(location_data, x='Engagements', y='Location', orientation='h', color='Location', title="Top 5 Locations by Engagement")
    st.plotly_chart(fig5, use_container_width=True)

with st.expander("🔍 AI Insight - Top Locations"):
    st.write(get_insights(
        "Top 5 Locations by Engagement",
        location_data,
        "Lokasi mana yang paling efektif untuk target audiens ZingPop dan apa yang menyebabkannya?"
    ))
