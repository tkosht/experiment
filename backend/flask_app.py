from __future__ import annotations

import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from slackeventsapi import SlackEventAdapter

from app.component.slack_event_handler import event_handler

load_dotenv(".env")

app = Flask(__name__)
slack_events_adapter = SlackEventAdapter(
    os.environ["SLACK_SIGNING_SECRET"], "/slack/events", app
)


@app.route("/")
def api_hello_world():
    return "Hello from Flask!"


@slack_events_adapter.on("reaction_added")
def event_reaction_added(event_data: dict):
    event_handler(event_data)
    return jsonify({})


@slack_events_adapter.on("message")
def event_message(event_data: dict):
    event_handler(event_data)
    return jsonify({})
