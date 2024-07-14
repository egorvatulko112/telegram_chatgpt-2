import telebot
from telebot import apihelper
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
import time
from functools import wraps

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
bot = telebot.TeleBot(TOKEN)

model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def error_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            bot.reply_to(args[0], 'Произошла ошибка, попробуйте позже.')
    return wrapper

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, 'Привет! Я бот, который может генерировать текст с помощью GPT-2.')

@bot.message_handler(func=lambda message: True)
@error_handler
def handle_message(message):
    bot.send_chat_action(message.chat.id, 'typing')
    time.sleep(1)  # Имитация печатания
    input_text = message.text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = model.generate(input_ids, max_length=50, num_return_sequences=1)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    bot.reply_to(message, text)

bot.polling(none_stop=True)
