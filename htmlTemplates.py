css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2f2b3e
}
.chat-message.bot {
    background-color: #6d7b99
}
.chat-message .message {
  padding: 0 1.5rem;
  color: #fff;
}
'''
# TODO is there better streamlit feature ???

bot_template = '''
<div class="chat-message bot">
    <div class="message">{{MSG}}</div>
</div>
'''
user_template = '''
<div class="chat-message user"> 
    <div class="message">{{MSG}}</div>
</div>
'''
