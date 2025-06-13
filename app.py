from flask import Flask, render_template, jsonify, request
import threading
import time
import datetime
import asyncio
import queue
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import telegram
import feedparser
from datetime import datetime, timedelta

# === CONFIGURATION ===
EMA_FAST = 10
EMA_SLOW = 30
ATR_PERIOD = 14
RSI_PERIOD = 16
ATR_FLOOR = 0.3
ATR_CAP = 2.5
COOLDOWN = 2  # minutes
SYMBOL = "XAUUSD"

# Initialize asyncio event loop
loop = asyncio.new_event_loop()


def start_event_loop():
    asyncio.set_event_loop(loop)
    loop.run_forever()

threading.Thread(target=start_event_loop, daemon=True).start()

# Global state for breakout/pullback logic
last_breakout_price = None
pullback_confirmed = False

# === MT5 Initialization ===
mt5.initialize()
print("✅ MT5 initialized")

# === Telegram Setup ===
TELEGRAM_TOKEN = "7068366818:AAEPcJ1u46fE0ALRS4IJvNWtWWV9Ly_uZj8"
CHAT_ID = "786821772"
bot = telegram.Bot(token=TELEGRAM_TOKEN)

# Message queue for rate-limited messaging
message_queue = queue.Queue()

async def send_message_async(msg):
    try:
        await bot.send_message(chat_id=CHAT_ID, text=msg)
    except Exception as e:
        print("Error sending message:", e)

async def message_worker():
    while True:
        msg = message_queue.get()
        await send_message_async(msg)
        await asyncio.sleep(1.5)

# Start the message worker
asyncio.run_coroutine_threadsafe(message_worker(), loop)

def send_message(msg):
    message_queue.put(msg)

# === Utility Functions ===

def get_data(symbol, timeframe, bars):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    return pd.DataFrame(rates)

def calculate_indicators(df):
    df['EMA_FAST'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
    df['EMA_SLOW'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()
    df['ATR'] = (df['high'] - df['low']).rolling(window=ATR_PERIOD).mean()

    # RSI calculation
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=RSI_PERIOD).mean()
    avg_loss = loss.rolling(window=RSI_PERIOD).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def is_liquidity_sweep(df):
    if len(df) < 2:
        return False
    price_movement = df['close'].iloc[-1] - df['close'].iloc[-2]
    return abs(price_movement) > 0.2

def is_choc_bos(df):
    if len(df) < 2:
        return None
    prev_high = df['high'].iloc[-2]
    prev_low = df['low'].iloc[-2]
    curr_high = df['high'].iloc[-1]
    curr_low = df['low'].iloc[-1]
    if curr_high > prev_high:
        return "BOS"
    elif curr_low < prev_low:
        return "CHoCH"
    return None

def is_order_block(df):
    if len(df) < 2:
        return False
    prev_high = df['high'].iloc[-2]
    prev_low = df['low'].iloc[-2]
    curr_price = df['close'].iloc[-1]
    zone = 0.02
    return (curr_price > prev_low - zone and curr_price < prev_high + zone)

# === Pullback Entry Logic (Flexible) ===
def entry_condition(df):
    global last_breakout_price, pullback_confirmed

    last = df.iloc[-1]
    prev = df.iloc[-2]
    body_ratio = abs(last['close'] - last['open']) / (last['high'] - last['low'] + 1e-6)
    sweep = is_liquidity_sweep(df)
    structure_break = is_choc_bos(df)
    in_ob = is_order_block(df)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🟡 Checking entry: body_ratio={body_ratio:.2f}, sweep={sweep}, break={structure_break}, in_OB={in_ob}")

    # === Detect breakout ===
    if last_breakout_price is None:
        if last['close'] > prev['high'] and body_ratio > 0.5 and structure_break == "BOS":
            last_breakout_price = prev['high']
            print(f"🔔 Potential BUY breakout at {last['close']} (zone: {last_breakout_price})")
        elif last['close'] < prev['low'] and body_ratio > 0.5 and structure_break == "CHoCH":
            last_breakout_price = prev['low']
            print(f"🔔 Potential SELL breakout at {last['close']} (zone: {last_breakout_price})")
        else:
            print("⏳ No entry or conditions not met yet.")
        return None

    # === Confirm pullback into zone ===
    if last_breakout_price:
        if (last['low'] <= last_breakout_price and df['EMA_FAST'].iloc[-1] > df['EMA_SLOW'].iloc[-1]):
            pullback_confirmed = True
            print(f"✅ BUY pullback confirmed at {last['close']}")
            return 'BUY'
        elif (last['high'] >= last_breakout_price and df['EMA_FAST'].iloc[-1] < df['EMA_SLOW'].iloc[-1]):
            pullback_confirmed = True
            print(f"✅ SELL pullback confirmed at {last['close']}")
            return 'SELL'

    print("⏳ Waiting for pullback confirmation...")
    return None

def calculate_weighted_score(df):
    last_row = df.iloc[-1]
    trend_score = 1 if last_row['EMA_FAST'] > last_row['EMA_SLOW'] and last_row['RSI'] > 55 else 0
    liquidity_score = 1 if is_liquidity_sweep(df) else 0
    order_block_score = 1 if is_order_block(df) else 0
    atr = last_row['ATR']
    if ATR_FLOOR <= atr <= ATR_CAP:
        volatility_score = 1
    elif ATR_FLOOR <= atr <= (ATR_CAP * 0.9):
        volatility_score = 0.5
    else:
        volatility_score = 0
    bos_choc_score = 1 if is_choc_bos(df) else 0

    total_score = (trend_score * 0.3 + liquidity_score * 0.2 +
                   order_block_score * 0.15 + volatility_score * 0.15 +
                   bos_choc_score * 0.2)

    print(f"📊 Score - Trend: {trend_score}, Liquidity: {liquidity_score}, OB: {order_block_score}, Vol: {volatility_score}, Break: {bos_choc_score} -> Total: {total_score}")
    return total_score

def check_trade_conditions(df):
    score = calculate_weighted_score(df)
    return score >= 0.7

def calculate_tp_sl(entry_price, atr, direction, symbol):
    info = mt5.symbol_info(symbol)
    point = info.point if info else 0.01
    if direction == 'BUY':
        sl = entry_price - (entry_price * 0.003)
        tp = entry_price + atr * 2
    else:
        sl = entry_price + (entry_price * 0.003)
        tp = entry_price - atr * 2
    return round(tp / point) * point, round(sl / point) * point

def is_high_impact_news_soon():
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
    feed = feedparser.parse(url)
    now = datetime.utcnow() + timedelta(hours=1)
    keywords = ["USD", "FOMC", "Fed", "Gold", "Powell", "Interest Rate", "Monetary Policy"]
    buffer = 30
    for entry in feed.entries:
        title = entry.title
        summary = entry.get("summary", "")
        full_text = f"{title} {summary}"
        if any(k in full_text for k in keywords):
            try:
                parts = title.split('|')
                dt_str = parts[-1].strip()
                event_time = datetime.strptime(dt_str, "%b %d %Y %I:%M%p")
                if now <= event_time <= now + timedelta(minutes=buffer):
                    print(f"🛑 News risk detected: {title}")
                    return True
            except:
                continue
    return False

def execute_trade(symbol, action, entry_price, tp, sl):
    order_type = mt5.ORDER_TYPE_BUY if action == 'BUY' else mt5.ORDER_TYPE_SELL
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"❌ Symbol not found: {symbol}")
        send_message(f"❌ Symbol not found: {symbol}")
        return
    if not symbol_info.visible:
        mt5.symbol_select(symbol, True)
    digits = symbol_info.digits
    entry_price = round(entry_price, digits)
    tp = round(tp, digits)
    sl = round(sl, digits)

    for filling_mode in [mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN]:
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": 0.05,
            "type": order_type,
            "price": entry_price,
            "sl": sl,
            "tp": tp,
            "deviation": 10,
            "magic": 234000,
            "comment": "Scalping bot 1",
            "type_filling": filling_mode,
            "type_time": mt5.ORDER_TIME_GTC
        }
        print(f"📤 Trying filling_mode: {filling_mode}")
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"✅ Trade executed: {action} at {entry_price} with filling_mode {filling_mode}")
            send_message(f"✅ Trade executed: {action} at {entry_price}\nTP: {tp}, SL: {sl}")
            return
        else:
            print(f"❌ Failed with filling_mode {filling_mode}: {result.comment}")
    print("❌ All filling modes failed.")
    send_message("❌ Trade failed on all filling modes.")

# === MAIN LOOP ===
# We will run the main trading loop in a thread, controlled by start/stop commands.
# For now, define the function

def trading_loop(stop_event):
    global last_breakout_price, pullback_confirmed
    last_trade_time = time.time() - COOLDOWN * 60
    while not stop_event.is_set():
        try:
            if time.time() - last_trade_time < COOLDOWN * 60:
                time.sleep(5)
                continue

            if is_high_impact_news_soon():
                print("⚠️ Skipping trade due to upcoming news.")
                time.sleep(60)
                continue

            df = get_data(SYMBOL, mt5.TIMEFRAME_M1, 100)
            if df.empty:
                print("No data received.")
                time.sleep(60)
                continue

            df = calculate_indicators(df)
            action = entry_condition(df)

            if action and pullback_confirmed and check_trade_conditions(df):
                entry_price = df['close'].iloc[-1]
                atr = df['ATR'].iloc[-1]
                tp, sl = calculate_tp_sl(entry_price, atr, action, SYMBOL)
                execute_trade(SYMBOL, action, entry_price, tp, sl)

                last_trade_time = time.time()
                last_breakout_price = None
                pullback_confirmed = False
            time.sleep(60)
        except Exception as e:
            print("Error in trading loop:", e)
            time.sleep(60)

# Thread control
trade_thread = None
stop_event = threading.Event()

# === Flask App ===
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_bot():
    global trade_thread, stop_event
    if trade_thread and trade_thread.is_alive():
        return jsonify({"status": "Bot already running"})
    stop_event.clear()
    trade_thread = threading.Thread(target=trading_loop, args=(stop_event,))
    trade_thread.start()
    return jsonify({"status": "Bot started"})

@app.route('/status')
def status():
    running = trade_thread is not None and trade_thread.is_alive()
    return jsonify({"running": running})

@app.route('/stop', methods=['POST'])
def stop_bot():
    global stop_event, trade_thread
    if not trade_thread or not trade_thread.is_alive():
        return jsonify({"status": "Bot not running"})
    stop_event.set()
    return jsonify({"status": "Stopping bot"})

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
