from flask import Flask
import click
from cli import cli

app = Flask(__name__)
app.cli.add_command(cli)

@app.route('/')
def index():
    return "Welcome to the YouTube Dubbing Application!"

if __name__ == '__main__':
    app.run(debug=True)