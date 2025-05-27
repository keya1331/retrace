# retrace 
an analyzer for exported whatsapp chats — structure, sentiment, and stats.

built during a data science course.  
my part: ideation, system behavior, and sentiment analysis pipeline.

---

## features  
- parses raw chat logs (android & iphone)  
- visualizes message timelines, emoji stats, word clouds  
- tf-idf scoring per user  
- sentiment analysis using textblob + vader  
- supports both group-level and user-specific insights

---

## visual insights 📊  
snapshots from the streamlit-based chat analyzer ↓

<p align="center">
  <img src="screens/a1.jpg" alt="activity timeline" height="200px"/>
  <img src="screens/a2.jpg" alt="emoji usage" height="200px"/>
  <img src="screens/a3.jpg" alt="word cloud" height="200px"/>
</p>

<p align="center">
  <img src="screens/a4.jpg" alt="tf-idf output" height="200px"/>
  <img src="screens/a5.jpg" alt="user stats" height="200px"/>
  <img src="screens/a6.jpg" alt="chat heatmap" height="200px"/>
</p>

<p align="center">
  <img src="screens/a7.jpg" alt="sentiment chart" height="200px"/>
  <img src="screens/a8.jpg" alt="overall breakdown" height="200px"/>
  <img src="screens/a9.jpg" alt="example output" height="200px"/>
</p>

---

## tech stack  
- python, streamlit  
- pandas, matplotlib, seaborn  
- nltk, textblob, vader, scikit-learn

---

## run it  
[› deployed app](https://chat-analysis-ds.streamlit.app/)  
[› project report](./chat-analyzer-report.pdf)

---

*some conversations end, others get analyzed.*
