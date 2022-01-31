from flask import Flask
import server

app = Flask(__name__)

# URLs
app.add_url_rule('/', 'extract_nc', server.extract_nc, methods=['GET', 'POST'])

if __name__ == "__main__":
    app.run(debug = True)
