from typing import Final
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import openai

TOKEN: Final = ''
BOT_USERNAME: Final = ''

#Commands
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Hello! Thanks for chatting with me! I am Phuong Huyen')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Do you need help? Please type something')

async def custom_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('This is a custom command!')

#Responses
def handle_response(text: str) -> str: 
    processed: str = text.lower()
    print("Processed", processed)

    if 'hello' in processed: 
        return 'Hey there!'
    
    if 'how are you' in processed:
        return "I'm good"
    
    if 'who are you' in processed: 
        return "i'm your assistant"
    
    if 'do you like anything' in processed: 
        return 'i like u <3'
    
    if 'you are so beautiful' in processed: 
        return 'Thank you so much <3'
    
    return 'I do not understand what you wrote ...'

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE): 
    message_type: str = update.message.chat.type
    text: str = update.message.text
    
    print(f'User ({update.message.chat.id}) in {message_type}: "{text}"')

    if message_type == 'group': 
        if BOT_USERNAME in text: 
            new_text: str = text.replace(BOT_USERNAME, '').strip()
            response: str = handle_response(new_text)
        else: 
            return
    else: 
        response: str = handle_response(text)

    print('Bot', response)
    await update.message.reply_text(response)

async def error(update: Update, context: ContextTypes.DEFAULT_TYPE): 
    print(f'Update {update} caused error {context.error}')

# ai chat bot
openai.api_key = ""

def chat_with_gpt(prompt): 
    response = openai.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages = [{"role": "user", "content": prompt}]
    )   
    return response.choices[0].message.content.strip()

#get rag response
def get_rag_response(user_input):
    # This is where you embed the user_input, search your DB, etc.
    # context = "Your Excel knowledge base context here..."

    # Combine context with user question
    # prompt = f"Answer the question based on the following context:\n{context}\n\nQuestion: {user_input}"
    prompt = user_input
    return chat_with_gpt(prompt)

async def handle_ai_message(update: Update, context: ContextTypes.DEFAULT_TYPE): 
    message_type: str = update.message.chat.type
    text: str = update.message.text

    print(f'User ({update.message.chat.id}) in {message_type}: "{text}"')

    if message_type == 'group':
        if BOT_USERNAME in text:
            new_text: str = text.replace(BOT_USERNAME, '').strip()
            user_input = new_text
        else:
            return
    else:
        user_input = text

    rag_response = get_rag_response(user_input)

    print('Bot:', rag_response)
    await update.message.reply_text(rag_response)

if __name__ == '__main__': 
    print('Start polling...')
    app = Application.builder().token(TOKEN).build()

    #Command
    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('help', help_command))
    app.add_handler(CommandHandler('custom', custom_command))

    #Messages
    app.add_handler(MessageHandler(filters.TEXT, handle_message))

    #Error
    app.add_error_handler(error)

    #Polls the bot
    print('Polling...')
    app.run_polling(poll_interval=3)
    