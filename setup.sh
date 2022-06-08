mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"suriadivajrakaruna432@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\

[theme]\n\
backgroundColor = '#FFFFFF'\n\
textColor = '#000000'\n\
font = 'sans serif'\n\
" > ~/.streamlit/config.toml