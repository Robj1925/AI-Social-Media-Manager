# Uses Gemini via OpenAI-compatible endpoint
# Includes both non-streaming and streaming endpoints
# Serves a simple HTML UI + JSON API

import os
import asyncio
from flask import Flask, request, jsonify, render_template, Response
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
from agents import Agent, Runner, trace, OpenAIChatCompletionsModel

load_dotenv(override=True)

# ────────────────────────────────────────────────
# Gemini client setup (sync + async versions)
# ────────────────────────────────────────────────
sync_client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

async_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

gemini_model = OpenAIChatCompletionsModel(
    model=os.getenv("MODEL", "gemini-1.5-flash"),  # fallback if .env missing
    openai_client=async_client,  # agents lib uses async
)

# ────────────────────────────────────────────────
# Agent definition (same as your original)
# ────────────────────────────────────────────────
instructions = """
You are a Social Media Manager for a BJJ/MMA-focused YouTube channel. Your task is to write a concise, respectful, and personalized Instagram or Twitter/X Direct Message to a specific BJJ, MMA, or UFC athlete with the goal of inviting them on for an interview.

Use the following provided recent accomplishment or news about the athlete at the beginning of the message (be very specific and use 2–3 key points if multiple are provided). Show genuine awareness and respect for their career. Keep this opening natural and not overly flattering:

{athlete_accomplishment}

After the opening, smoothly transition into introducing the channel. The channel is a BJJ and MMA-focused YouTube channel that creates educational breakdowns, technical insights, and short-form, fast-paced content centered around grappling, MMA, and high-level combat sports. The tone of the channel is authentic, technical, and practitioner-focused, aimed at fans and athletes who genuinely love the sport.

Clearly state that the purpose of the message is to invite the athlete on for an interview or conversation. Emphasize that the interview would focus on their experience, journey, mindset, and insights into BJJ and/or MMA, and that it would be scheduled at their earliest convenience, with flexibility around their availability.

Keep the overall message friendly, professional, and brief—do not sound corporate, spammy, or overly promotional. Avoid buzzwords and marketing language. The message should feel like it’s coming from a real fan who respects the athlete and the sport.

At the very end of the message, include a soft call-to-action by linking the channel so they can check it out if they’re interested:
https://www.youtube.com/@Rob-J-BJJ

Do not include hashtags, emojis (unless very subtle), or excessive formatting. Talk only in first person using "I" never using "We". My name is Robert. The final output should be a single, polished direct message ready to send. Keep the message at 1,000 CHARACTERS MAX.
"""

social_media_manager = Agent(
    name="Social Media Manager",
    instructions=instructions,
    model=gemini_model
)

# ────────────────────────────────────────────────
# Flask app
# ────────────────────────────────────────────────
app = Flask(__name__)

# ────────────────────────────────────────────────
# Non-streaming generation (sync wrapper around async agent)
# ────────────────────────────────────────────────
def generate_dm(athlete_name: str, accomplishment: str) -> str:
    if not accomplishment:
        accomplishment = "No specific recent accomplishment provided — mention their general skill / reputation in the sport."

    user_message = f"""
Athlete: {athlete_name}

Recent accomplishment/news:
{accomplishment}

Send a DM to {athlete_name} inviting them for an interview.
"""

    # Run async agent in sync context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        with trace("Social Media Manager"):
            result = loop.run_until_complete(Runner.run(social_media_manager, user_message))
        return result.final_output.strip()
    finally:
        loop.close()

# ────────────────────────────────────────────────
# Streaming version (yields chunks as agent generates)
# ────────────────────────────────────────────────
async def generate_dm_stream(athlete_name: str, accomplishment: str):
    if not accomplishment:
        accomplishment = "No specific recent accomplishment provided — mention their general skill / reputation in the sport."

    user_message = f"""
Athlete: {athlete_name}

Recent accomplishment/news:
{accomplishment}

Send a DM to {athlete_name} inviting them for an interview.
"""

    # We'll simulate streaming by yielding the final result in chunks (since agents lib doesn't natively stream tokens yet)
    # For real token-by-token streaming you'd need to wrap the underlying client directly
    full_dm = await Runner.run(social_media_manager, user_message)
    text = full_dm.final_output.strip()

    # Yield word-by-word for demo effect
    words = text.split()
    for i, word in enumerate(words):
        yield word + (" " if i < len(words)-1 else "")
        await asyncio.sleep(0.05)  # simulate typing effect

# ────────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', dm='', athlete_name='', accomplishment='')

@app.route('/generate_dm_ui', methods=['POST'])
def generate_dm_ui():
    athlete_name = request.form.get('athlete_name', '').strip()
    accomplishment = request.form.get('accomplishment', '').strip()

    if not athlete_name:
        return render_template('index.html', dm="Please provide an athlete name.", athlete_name='', accomplishment='')

    dm = generate_dm(athlete_name, accomplishment)
    return render_template('index.html', dm=dm, athlete_name=athlete_name, accomplishment=accomplishment)

@app.route('/generate_dm', methods=['POST'])
def generate_dm_endpoint():
    data = request.get_json()
    if not data or 'athlete_name' not in data:
        return jsonify({"error": "Missing 'athlete_name' in request body"}), 400

    athlete_name = data['athlete_name'].strip()
    accomplishment = data.get('accomplishment', '').strip()

    dm = generate_dm(athlete_name, accomplishment)
    return jsonify({"dm": dm})

@app.route('/stream_dm', methods=['POST'])
def stream_dm():
    data = request.get_json()
    if not data or 'athlete_name' not in data:
        return jsonify({"error": "Missing 'athlete_name' in request body"}), 400

    athlete_name = data['athlete_name'].strip()
    accomplishment = data.get('accomplishment', '').strip()

    async def stream_response():
        yield "data: Generating DM...\n\n"
        async for chunk in generate_dm_stream(athlete_name, accomplishment):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return Response(
        stream_response(),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
