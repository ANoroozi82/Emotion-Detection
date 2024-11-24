import sqlite3
import pytz
from datetime import datetime

conn = sqlite3.connect('./data/data.db', check_same_thread=False)
c = conn.cursor()

IRAN = pytz.timezone("Asia/Tehran")

def createEmotionclfTable():
    c.execute('CREATE TABLE IF NOT EXISTS emotionclfTable(rawtext TEXT, prediction TEXT, probability NUMBER, timeOfvisit TIMESTAMP)')

def addPredictionDetails(rawtext, prediction, probability, timeOfvisit=None):
    if timeOfvisit is None:
        timeOfvisit = datetime.now(IRAN).strftime("%Y-%m-%d %H:%M:%S")
    else:
        timeOfvisit = timeOfvisit.astimezone(IRAN).strftime("%Y-%m-%d %H:%M:%S")
    c.execute('INSERT INTO emotionclfTable(rawtext, prediction, probability, timeOfvisit) VALUES (?, ?, ?, ?)', (rawtext, prediction, probability, timeOfvisit))
    conn.commit()

def viewAllPredictionDetails():
    c.execute('SELECT * FROM emotionclfTable')
    data = c.fetchall()
    return data
