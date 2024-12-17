import base64

def get_image_base64(image_path):
    """
    Reads an image file and returns its Base64 encoded string.
    """
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
    
# Paths to your local images
USER_ICON_PATH = "./images/ThinkingMan.png"
BOT_ICON_PATH = "./images/bot.jpg"

# Encode images to Base64
user_icon_base64 = get_image_base64(USER_ICON_PATH)
bot_icon_base64 = get_image_base64(BOT_ICON_PATH)

css = '''
<style>
.chat-container {
    max-height: 500px;
    overflow-y: auto;
    padding: 10px;
    background-color: #F5F5F5;
}
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
}
.chat-message.user {
    background-color: #2b313e;
}
.chat-message.bot {
    background-color: #475063;
}
.chat-message .avatar {
    width: 10%;
}
.chat-message .avatar img {
    max-width: 30px;
    max-height: 30px;
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .message {
    width: 90%;
    padding: 0 1rem;
    color: #fff;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="data:image/png;base64,{bot_icon}" alt="User Icon"/>
    </div>
    <div class="message">{message}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="data:image/png;base64,{user_icon}" alt="Bot Icon"/>
    </div>    
    <div class="message">{message}</div>
</div>
'''
