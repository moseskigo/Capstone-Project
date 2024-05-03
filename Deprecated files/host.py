from flask import Flask

from threading import Thread

app = Flask('')


app.route('/')

def main():
  return "Bot is online."


def run():
  app.run(host = "127.0.0.1", port = 5000)


def keep_alive():
  server = Thread(target = run)
  server.start()